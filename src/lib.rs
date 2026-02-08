//! Asynchronous handle for rusqlite library.
//!
//! # Guide
//!
//! This library provides [`Connection`] struct. [`Connection`] struct is a handle
//! to call functions in background thread and can be cloned cheaply.
//! [`Connection::call`] method calls provided function in the background thread
//! and returns its result asynchronously.
//!
//! # Design
//!
//! A thread is spawned for each opened connection handle. When `call` method
//! is called: provided function is boxed, sent to the thread through mpsc
//! channel and executed. Return value is then sent by oneshot channel from
//! the thread and then returned from function.
//!
//! # Example
//!
//! ```rust,no_run
//! use tokio_rusqlite::{params, Connection, Result};
//!
//! #[derive(Debug)]
//! struct Person {
//!     id: i32,
//!     name: String,
//!     data: Option<Vec<u8>>,
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let conn = Connection::open_in_memory().await?;
//!
//!     let people = conn
//!         .call(|conn| {
//!             conn.execute(
//!                 "CREATE TABLE person (
//!                     id    INTEGER PRIMARY KEY,
//!                     name  TEXT NOT NULL,
//!                     data  BLOB
//!                 )",
//!                 [],
//!             )?;
//!
//!             let steven = Person {
//!                 id: 1,
//!                 name: "Steven".to_string(),
//!                 data: None,
//!             };
//!
//!             conn.execute(
//!                 "INSERT INTO person (name, data) VALUES (?1, ?2)",
//!                 params![steven.name, steven.data],
//!             )?;
//!
//!             let mut stmt = conn.prepare("SELECT id, name, data FROM person")?;
//!             let people = stmt
//!                 .query_map([], |row| {
//!                     Ok(Person {
//!                         id: row.get(0)?,
//!                         name: row.get(1)?,
//!                         data: row.get(2)?,
//!                     })
//!                 })?
//!                 .collect::<std::result::Result<Vec<Person>, rusqlite::Error>>()?;
//!
//!             Ok(people)
//!         })
//!         .await?;
//!
//!     for person in people {
//!         println!("Found person {:?}", person);
//!     }
//!
//!     Ok(())
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(
    clippy::await_holding_lock,
    clippy::cargo_common_metadata,
    clippy::dbg_macro,
    clippy::empty_enums,
    clippy::enum_glob_use,
    clippy::inefficient_to_string,
    clippy::mem_forget,
    clippy::mutex_integer,
    clippy::needless_continue,
    clippy::todo,
    clippy::unimplemented,
    clippy::wildcard_imports,
    future_incompatible,
    missing_docs,
    missing_debug_implementations,
    unreachable_pub
)]

#[cfg(test)]
mod tests;

use crossfire::{mpsc, oneshot, MTx, Rx, SendError};
use std::{
    fmt::{self, Debug, Display},
    path::Path,
    thread,
};

pub use rusqlite::{self, *};

const BUG_TEXT: &str = "bug in tokio-rusqlite, please report";

#[derive(Debug)]
/// Represents the errors specific for this library.
#[non_exhaustive]
pub enum Error<E = rusqlite::Error> {
    /// The connection to the SQLite has been closed and cannot be queried any more.
    ConnectionClosed,

    /// An error occured while closing the SQLite connection.
    /// This `Error` variant contains the [`Connection`], which can be used to retry the close operation
    /// and the underlying [`rusqlite::Error`] that made it impossile to close the database.
    Close((Connection, rusqlite::Error)),

    /// An application-specific error occured.
    Error(E),
}

impl<E: Display> Display for Error<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ConnectionClosed => write!(f, "ConnectionClosed"),
            Error::Close((_, e)) => write!(f, "Close((Connection, \"{e}\"))"),
            Error::Error(e) => write!(f, "Error(\"{e}\")"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for Error<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::ConnectionClosed => None,
            Error::Close((_, e)) => Some(e),
            Error::Error(e) => Some(e),
        }
    }
}

impl From<rusqlite::Error> for Error {
    fn from(value: rusqlite::Error) -> Self {
        Error::Error(value)
    }
}

/// The result returned on method calls in this crate.
pub type Result<T> = std::result::Result<T, Error>;

type MessageSender = MTx<mpsc::List<Message>>;
type MessageReceiver = Rx<mpsc::List<Message>>;
type CloseSender = oneshot::TxOneshot<std::result::Result<(), rusqlite::Error>>;
type TaskFn = Box<dyn Task>;

trait Task: Send + 'static {
    fn run(self: Box<Self>, conn: &mut rusqlite::Connection);
    fn fail(self: Box<Self>);
}

struct ExecuteTask<F, R> {
    func: F,
    reply: oneshot::TxOneshot<R>,
}

impl<F, R> Task for ExecuteTask<F, R>
where
    F: FnOnce(&mut rusqlite::Connection) -> R + Send + 'static,
    R: Send + 'static,
{
    fn run(self: Box<Self>, conn: &mut rusqlite::Connection) {
        let ExecuteTask { func, reply } = *self;
        let value = func(conn);
        let _ = reply.send(value);
    }

    fn fail(self: Box<Self>) {
        drop(self);
    }
}

enum Message {
    Execute(TaskFn),
    Close(CloseSender),
}

/// A handle to call functions in background thread.
#[derive(Clone)]
pub struct Connection {
    sender: MessageSender,
}

impl Connection {
    /// Open a new connection to a SQLite database.
    ///
    /// `Connection::open(path)` is equivalent to
    /// `Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE |
    /// OpenFlags::SQLITE_OPEN_CREATE)`.
    ///
    /// # Failure
    ///
    /// Will return `Err` if `path` cannot be converted to a C-compatible
    /// string or if the underlying SQLite open call fails.
    pub async fn open<P: AsRef<Path>>(path: P) -> std::result::Result<Self, rusqlite::Error> {
        let path = path.as_ref().to_owned();
        start(move || rusqlite::Connection::open(path)).await
    }

    /// Open a new connection to an in-memory SQLite database.
    ///
    /// # Failure
    ///
    /// Will return `Err` if the underlying SQLite open call fails.
    pub async fn open_in_memory() -> std::result::Result<Self, rusqlite::Error> {
        start(rusqlite::Connection::open_in_memory).await
    }

    /// Open a new connection to a SQLite database.
    ///
    /// [Database Connection](http://www.sqlite.org/c3ref/open.html) for a
    /// description of valid flag combinations.
    ///
    /// # Failure
    ///
    /// Will return `Err` if `path` cannot be converted to a C-compatible
    /// string or if the underlying SQLite open call fails.
    pub async fn open_with_flags<P: AsRef<Path>>(
        path: P,
        flags: OpenFlags,
    ) -> std::result::Result<Self, rusqlite::Error> {
        let path = path.as_ref().to_owned();
        start(move || rusqlite::Connection::open_with_flags(path, flags)).await
    }

    /// Open a new connection to a SQLite database using the specific flags
    /// and vfs name.
    ///
    /// [Database Connection](http://www.sqlite.org/c3ref/open.html) for a
    /// description of valid flag combinations.
    ///
    /// # Failure
    ///
    /// Will return `Err` if either `path` or `vfs` cannot be converted to a
    /// C-compatible string or if the underlying SQLite open call fails.
    pub async fn open_with_flags_and_vfs<P: AsRef<Path>>(
        path: P,
        flags: OpenFlags,
        vfs: &str,
    ) -> std::result::Result<Self, rusqlite::Error> {
        let path = path.as_ref().to_owned();
        let vfs = vfs.to_owned();
        start(move || rusqlite::Connection::open_with_flags_and_vfs(path, flags, &*vfs)).await
    }

    /// Open a new connection to an in-memory SQLite database.
    ///
    /// [Database Connection](http://www.sqlite.org/c3ref/open.html) for a
    /// description of valid flag combinations.
    ///
    /// # Failure
    ///
    /// Will return `Err` if the underlying SQLite open call fails.
    pub async fn open_in_memory_with_flags(
        flags: OpenFlags,
    ) -> std::result::Result<Self, rusqlite::Error> {
        start(move || rusqlite::Connection::open_in_memory_with_flags(flags)).await
    }

    /// Open a new connection to an in-memory SQLite database using the
    /// specific flags and vfs name.
    ///
    /// [Database Connection](http://www.sqlite.org/c3ref/open.html) for a
    /// description of valid flag combinations.
    ///
    /// # Failure
    ///
    /// Will return `Err` if `vfs` cannot be converted to a C-compatible
    /// string or if the underlying SQLite open call fails.
    pub async fn open_in_memory_with_flags_and_vfs(
        flags: OpenFlags,
        vfs: &str,
    ) -> std::result::Result<Self, rusqlite::Error> {
        let vfs = vfs.to_owned();
        start(move || rusqlite::Connection::open_in_memory_with_flags_and_vfs(flags, &*vfs)).await
    }

    /// Call a function in background thread and get the result
    /// asynchronously.
    ///
    /// # Failure
    ///
    /// Will return `Err` if the database connection has been closed.
    /// Will return `Error::Error` wrapping the inner error if `function` failed.
    pub async fn call<F, R, E>(&self, function: F) -> std::result::Result<R, Error<E>>
    where
        F: FnOnce(&mut rusqlite::Connection) -> std::result::Result<R, E> + 'static + Send,
        R: Send + 'static,
        E: Send + 'static,
    {
        self.call_raw(function)
            .await
            .map_err(|_| Error::ConnectionClosed)
            .and_then(|result| result.map_err(Error::Error))
    }

    /// Call a function in background thread and get the result
    /// asynchronously.
    ///
    /// # Failure
    ///
    /// Will return `Err` if the database connection has been closed.
    pub async fn call_raw<F, R>(&self, function: F) -> Result<R>
    where
        F: FnOnce(&mut rusqlite::Connection) -> R + 'static + Send,
        R: Send + 'static,
    {
        let (sender, receiver) = oneshot::oneshot::<R>();
        let task = ExecuteTask {
            func: function,
            reply: sender,
        };

        self.sender
            .send(Message::Execute(Box::new(task)))
            .map_err(|_| Error::ConnectionClosed)?;

        receiver.await.map_err(|_| Error::ConnectionClosed)
    }

    /// Call a function in background thread and get the result
    /// asynchronously.
    ///
    /// This method can cause a `panic` if the underlying database connection is closed.
    /// it is a more user-friendly alternative to the [`Connection::call`] method.
    /// It should be safe if the connection is never explicitly closed (using the [`Connection::close`] call).
    ///
    /// Calling this on a closed connection will cause a `panic`.
    pub async fn call_unwrap<F, R>(&self, function: F) -> R
    where
        F: FnOnce(&mut rusqlite::Connection) -> R + Send + 'static,
        R: Send + 'static,
    {
        let (sender, receiver) = oneshot::oneshot::<R>();
        let task = ExecuteTask {
            func: function,
            reply: sender,
        };

        self.sender
            .send(Message::Execute(Box::new(task)))
            .expect("database connection should be open");

        receiver.await.expect(BUG_TEXT)
    }

    /// Close the database connection.
    ///
    /// This is functionally equivalent to the `Drop` implementation for
    /// `Connection`. It consumes the `Connection`, but on error returns it
    /// to the caller for retry purposes.
    ///
    /// If successful, any following `close` operations performed
    /// on `Connection` copies will succeed immediately.
    ///
    /// On the other hand, any calls to [`Connection::call`] will return a [`Error::ConnectionClosed`],
    /// and any calls to [`Connection::call_unwrap`] will cause a `panic`.
    ///
    /// # Failure
    ///
    /// Will return `Err` if the underlying SQLite close call fails.
    pub async fn close(self) -> Result<()> {
        let (sender, receiver) = oneshot::oneshot::<std::result::Result<(), rusqlite::Error>>();

        if let Err(SendError(_)) = self.sender.send(Message::Close(sender)) {
            // If the channel is closed on the other side, it means the connection closed successfully
            // This is a safeguard against calling close on a `Copy` of the connection
            return Ok(());
        }

        let result = receiver.await;

        if result.is_err() {
            // If we get a RecvError at this point, it also means the channel closed in the meantime
            // we can assume the connection is closed
            return Ok(());
        }

        result.unwrap().map_err(|e| Error::Close((self, e)))
    }
}

impl Debug for Connection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Connection").finish()
    }
}

impl From<rusqlite::Connection> for Connection {
    fn from(conn: rusqlite::Connection) -> Self {
        let (sender, receiver) = mpsc::unbounded_blocking::<Message>();
        thread::spawn(move || event_loop(conn, receiver));

        Self { sender }
    }
}

async fn start<F>(open: F) -> rusqlite::Result<Connection>
where
    F: FnOnce() -> rusqlite::Result<rusqlite::Connection> + Send + 'static,
{
    let (sender, receiver) = mpsc::unbounded_blocking::<Message>();
    let (result_sender, result_receiver) = oneshot::oneshot();

    thread::spawn(move || {
        let conn = match open() {
            Ok(c) => c,
            Err(e) => {
                result_sender.send(Err(e));
                return;
            }
        };

        result_sender.send(Ok(()));

        event_loop(conn, receiver);
    });

    result_receiver
        .await
        .expect(BUG_TEXT)
        .map(|_| Connection { sender })
}

fn event_loop(mut conn: rusqlite::Connection, receiver: MessageReceiver) {
    while let Ok(message) = receiver.recv() {
        match message {
            Message::Execute(task) => task.run(&mut conn),
            Message::Close(s) => {
                let result = conn.close();

                match result {
                    Ok(v) => {
                        s.send(Ok(v));
                        // drain the channel to make sure all pending tasks are dropped
                        loop {
                            match receiver.try_recv() {
                                Ok(message) => match message {
                                    Message::Execute(task) => task.fail(),
                                    Message::Close(sender) => {
                                        sender.send(Ok(()));
                                    }
                                },
                                Err(_) => break,
                            }
                        }
                        break;
                    }
                    Err((c, e)) => {
                        conn = c;
                        s.send(Err(e));
                    }
                }
            }
        }
    }
}
