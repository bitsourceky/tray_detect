//! Tray Detect — live school-lunch waste sorting.
//!
//! Usage:
//!   tray_detect                      run live on the default webcam
//!   tray_detect --camera 1           pick a different camera index
//!   tray_detect --conf 0.35          confidence threshold (default 0.25)
//!   tray_detect --image tray.jpg     run once on a still image (writes *_out.png)

mod bins;
mod draw;
mod model;

use anyhow::{Context, Result};
use draw::Overlay;
use model::Model;
use std::time::Instant;

const MODEL_PATH: &str = "best.onnx";

struct Args {
    camera: u32,
    conf: f32,
    image: Option<String>,
}

fn parse_args() -> Args {
    let mut args = Args { camera: 0, conf: 0.25, image: None };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--camera" => args.camera = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--conf" => args.conf = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.25),
            "--image" => args.image = it.next(),
            "--help" | "-h" => {
                eprintln!("tray_detect [--camera N] [--conf 0.25] [--image path]");
                std::process::exit(0);
            }
            _ => {}
        }
    }
    args
}

fn main() -> Result<()> {
    let args = parse_args();
    let mut model = Model::load(MODEL_PATH)
        .with_context(|| format!("loading model from {MODEL_PATH}"))?;
    let overlay = Overlay::new();

    match args.image {
        Some(path) => run_image(&mut model, &overlay, &path, args.conf),
        None => run_camera(&mut model, &overlay, args.camera, args.conf),
    }
}

/// One-shot inference on a still image — for testing without a webcam.
fn run_image(model: &mut Model, overlay: &Overlay, path: &str, conf: f32) -> Result<()> {
    let mut frame = image::open(path)
        .with_context(|| format!("opening image {path}"))?
        .to_rgb8();

    let t = Instant::now();
    let dets = model.detect(&frame, conf)?;
    println!("Detected {} item(s) in {:?}:", dets.len(), t.elapsed());
    for d in &dets {
        let bin = bins::bin_for_class(d.class_id);
        println!(
            "  [{}] {} {:.0}%  box=({:.0},{:.0})-({:.0},{:.0})",
            bin.label(),
            bins::CLASS_NAMES.get(d.class_id).copied().unwrap_or("?"),
            d.score * 100.0,
            d.x1, d.y1, d.x2, d.y2
        );
    }

    overlay.draw(&mut frame, &dets);
    let out = format!(
        "{}_out.png",
        path.rsplit_once('.').map(|(s, _)| s).unwrap_or(path)
    );
    frame.save(&out).with_context(|| format!("writing {out}"))?;
    println!("Wrote annotated image to {out}");
    Ok(())
}

/// Live webcam loop with an on-screen window.
fn run_camera(model: &mut Model, overlay: &Overlay, camera: u32, conf: f32) -> Result<()> {
    use minifb::{Key, Window, WindowOptions};
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
    use nokhwa::Camera;

    let index = CameraIndex::Index(camera);
    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut cam = Camera::new(index, format)
        .with_context(|| format!("opening camera {camera}"))?;
    cam.open_stream().context("starting camera stream")?;

    let mut window: Option<Window> = None;
    let mut smoothed_fps = 0.0f32;

    loop {
        let t = Instant::now();

        let buf = cam.frame().context("grabbing camera frame")?;
        let res = buf.resolution();
        let (w, h) = (res.width(), res.height());
        let decoded = buf.decode_image::<RgbFormat>().context("decoding frame")?;
        let mut frame = image::RgbImage::from_raw(w, h, decoded.into_raw())
            .context("wrapping frame buffer")?;

        let dets = model.detect(&frame, conf)?;
        overlay.draw(&mut frame, &dets);

        let dt = t.elapsed().as_secs_f32();
        let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
        smoothed_fps = if smoothed_fps == 0.0 { fps } else { smoothed_fps * 0.9 + fps * 0.1 };
        overlay.draw_hud(
            &mut frame,
            &format!("{:.0} FPS   {} item(s)", smoothed_fps, dets.len()),
        );

        // Lazily create the window once we know the camera resolution.
        let win = match window.as_mut() {
            Some(win) => win,
            None => {
                let win = Window::new(
                    "Tray Detect  —  press ESC to quit",
                    w as usize,
                    h as usize,
                    WindowOptions::default(),
                )
                .context("creating display window")?;
                window = Some(win);
                window.as_mut().unwrap()
            }
        };

        if !win.is_open() || win.is_key_down(Key::Escape) {
            break;
        }
        let argb: Vec<u32> = frame
            .pixels()
            .map(|p| (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32)
            .collect();
        win.update_with_buffer(&argb, w as usize, h as usize)
            .context("presenting frame")?;
    }

    Ok(())
}
