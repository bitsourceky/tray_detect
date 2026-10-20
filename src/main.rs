//! Tray Detect — live school-lunch waste sorting.
//!
//! Usage:
//!   tray_detect                      run live on the default webcam
//!   tray_detect --camera 1           pick a different camera index
//!   tray_detect --conf 0.35          confidence threshold (default 0.47)
//!   tray_detect --interval 2         seconds between inferences (default 1.25)
//!   tray_detect --model path.onnx    model file (default: best.onnx next to
//!                                    the executable, then in the working dir)
//!   tray_detect --image tray.jpg     run once on a still image (writes *_out.png)

mod bins;
mod draw;
mod model;

use anyhow::{bail, Context, Result};
use draw::Overlay;
use model::{Detection, Model};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const MODEL_FILE: &str = "best.onnx";
const DEFAULT_CONF: f32 = 0.47;
const DEFAULT_INTERVAL: f32 = 1.25;

struct Args {
    camera: u32,
    conf: f32,
    interval: f32,
    model: Option<PathBuf>,
    image: Option<String>,
}

fn parse_args() -> Args {
    let mut args = Args {
        camera: 0,
        conf: DEFAULT_CONF,
        interval: DEFAULT_INTERVAL,
        model: None,
        image: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--camera" => args.camera = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--conf" => {
                args.conf = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(DEFAULT_CONF)
            }
            "--interval" => {
                args.interval = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .filter(|v: &f32| v.is_finite() && *v >= 0.0)
                    .unwrap_or(DEFAULT_INTERVAL)
            }
            "--model" => args.model = it.next().map(PathBuf::from),
            "--image" => args.image = it.next(),
            "--help" | "-h" => {
                eprintln!(
                    "tray_detect [--camera N] [--conf 0.47] [--interval 1.25] \
                     [--model path] [--image path]"
                );
                std::process::exit(0);
            }
            _ => {}
        }
    }
    args
}

/// `--model` if given; otherwise `best.onnx` next to the executable, then in
/// the working directory.
fn resolve_model_path(explicit: Option<PathBuf>) -> Result<PathBuf> {
    let candidates: Vec<PathBuf> = match explicit {
        Some(p) => vec![p],
        None => [
            std::env::current_exe()
                .ok()
                .and_then(|exe| exe.parent().map(|d| d.join(MODEL_FILE))),
            std::env::current_dir().ok().map(|d| d.join(MODEL_FILE)),
        ]
        .into_iter()
        .flatten()
        .collect(),
    };
    if let Some(found) = candidates.iter().find(|p| p.is_file()) {
        return Ok(found.clone());
    }
    let tried: Vec<String> = candidates
        .iter()
        .map(|p| format!("  {}", p.display()))
        .collect();
    bail!(
        "model file not found; tried:
{}
pass --model <path> to point at best.onnx",
        tried.join(
            "
"
        )
    )
}

fn main() -> Result<()> {
    let args = parse_args();
    let model_path = resolve_model_path(args.model)?;
    let mut model = Model::load(&model_path)
        .with_context(|| format!("loading model from {}", model_path.display()))?;
    let overlay = Overlay::new();

    match args.image {
        Some(path) => run_image(&mut model, &overlay, &path, args.conf),
        None => run_camera(
            &mut model,
            &overlay,
            args.camera,
            args.conf,
            Duration::from_secs_f32(args.interval),
        ),
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
            bins::class_name(d.class_id),
            d.score * 100.0,
            d.x1,
            d.y1,
            d.x2,
            d.y2
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

/// Live webcam loop with an on-screen window. The preview renders every
/// frame; inference runs at most once per `interval`, and the latest results
/// stay overlaid in between.
fn run_camera(
    model: &mut Model,
    overlay: &Overlay,
    camera: u32,
    conf: f32,
    interval: Duration,
) -> Result<()> {
    use minifb::{Key, Window, WindowOptions};
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
    use nokhwa::Camera;

    let index = CameraIndex::Index(camera);
    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut cam = Camera::new(index, format).with_context(|| format!("opening camera {camera}"))?;
    cam.open_stream().context("starting camera stream")?;

    let mut window: Option<Window> = None;
    let mut smoothed_fps = 0.0f32;
    let mut dets: Vec<Detection> = Vec::new();
    let mut last_inference: Option<Instant> = None;

    loop {
        let t = Instant::now();

        let buf = cam.frame().context("grabbing camera frame")?;
        let res = buf.resolution();
        let (w, h) = (res.width(), res.height());
        let decoded = buf.decode_image::<RgbFormat>().context("decoding frame")?;
        let mut frame =
            image::RgbImage::from_raw(w, h, decoded.into_raw()).context("wrapping frame buffer")?;

        if last_inference.is_none_or(|at| at.elapsed() >= interval) {
            last_inference = Some(Instant::now());
            dets = model.detect(&frame, conf)?;
        }
        overlay.draw(&mut frame, &dets);

        let dt = t.elapsed().as_secs_f32();
        let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
        smoothed_fps = if smoothed_fps == 0.0 {
            fps
        } else {
            smoothed_fps * 0.9 + fps * 0.1
        };
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
