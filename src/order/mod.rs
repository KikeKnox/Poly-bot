//! Order module: unified executor plus the router implementations.

mod audit;
pub mod clob_router;
pub mod executor;
pub mod router;
pub mod sim_router;

pub use executor::{Executor, UnknownAttempt};
pub use router::{OrderKind, OrderRequest, OrderRouter, RouterResult, TimeInForce};
