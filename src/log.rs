/// Re-export everything from `log` crate
pub use log::*;

use anyhow::Error;

pub fn error_details(err: &Error) {
    log::error!("{err}");

    err.chain().enumerate().rev().for_each(|(i, cause)| {
        log::error!("  {i}: {cause}");
    });
}
