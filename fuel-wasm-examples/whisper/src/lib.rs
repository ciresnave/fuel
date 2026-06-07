pub const WITH_TIMER: bool = true;

mod app;
pub mod languages;
pub mod worker;
pub use app::App;
pub use worker::Worker;
