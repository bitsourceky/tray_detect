# Vendored dependencies

## nokhwa-bindings-macos 0.2.4 (patched)

Vendored copy of the crates.io release with one fix in `set_all`
(`src/lib.rs`): upstream selects the AVFoundation format and the frame-rate
range independently — it matches formats by resolution only, keeps looping
after a match, and never resets the previously chosen range. On cameras that
expose the same resolution in more than one pixel format with different
frame-rate limits (e.g. a Logitech Brio 100 offering 1920x1080 as 420v at
30 fps and yuvs at 5 fps), the mismatched pair makes AVFoundation throw an
uncaught Objective-C exception and the process crashes on startup.

The patch selects the format and its frame-rate range as a pair: a format is
only chosen when one of its own ranges matches the requested frame rate.

Drop this directory once upstream nokhwa ships an equivalent fix; the
`[patch.crates-io]` entry in the root `Cargo.toml` is the only wiring.
