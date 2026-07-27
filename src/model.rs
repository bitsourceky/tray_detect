//! YOLO26-nano ONNX inference: letterbox preprocess -> ort -> decode.
//!
//! The model was exported end-to-end (`end2end=True`, NMS baked in), so
//! `output0` is already `[1, 300, 6]` where each row is
//! `[x1, y1, x2, y2, score, class_id]` in 640x640 letterboxed pixel space,
//! sorted by score descending. No anchor decode or NMS needed here — we just
//! threshold and map the coordinates back to the original frame.

use anyhow::{anyhow, Result};
use image::RgbImage;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

/// `ort`'s error type carries the (non-Send) builder/session on failure so it
/// can be recovered, which means it won't auto-convert into `anyhow::Error`.
/// We never recover, so collapse it to a message.
fn e(err: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("onnxruntime: {err}")
}

const INPUT_SIZE: usize = 640;
const PAD_VALUE: f32 = 114.0 / 255.0; // standard YOLO letterbox gray

#[derive(Clone, Copy, Debug)]
pub struct Detection {
    /// Box in original-frame pixel coordinates.
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class_id: usize,
}

pub struct Model {
    session: Session,
}

impl Model {
    pub fn load(path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(e)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(e)?
            .with_intra_threads(4)
            .map_err(e)?
            .commit_from_file(path)
            .map_err(e)?;
        Ok(Self { session })
    }

    /// Run detection on an RGB frame. Returns boxes in the frame's own pixel
    /// coordinates, filtered to `conf_threshold`.
    pub fn detect(&mut self, frame: &RgbImage, conf_threshold: f32) -> Result<Vec<Detection>> {
        let (ow, oh) = (frame.width() as f32, frame.height() as f32);

        // --- letterbox: scale keeping aspect ratio, pad to 640x640 ---
        let r = (INPUT_SIZE as f32 / ow).min(INPUT_SIZE as f32 / oh);
        let new_w = (ow * r).round() as u32;
        let new_h = (oh * r).round() as u32;
        let pad_x = (INPUT_SIZE as u32 - new_w) / 2;
        let pad_y = (INPUT_SIZE as u32 - new_h) / 2;

        let resized = image::imageops::resize(
            frame,
            new_w,
            new_h,
            image::imageops::FilterType::Triangle,
        );

        // NCHW float tensor, RGB, normalized 0..1, gray padding.
        let mut input = vec![PAD_VALUE; 3 * INPUT_SIZE * INPUT_SIZE];
        let plane = INPUT_SIZE * INPUT_SIZE;
        for y in 0..new_h {
            for x in 0..new_w {
                let px = resized.get_pixel(x, y);
                let dst = (pad_y + y) as usize * INPUT_SIZE + (pad_x + x) as usize;
                input[dst] = px[0] as f32 / 255.0; // R
                input[plane + dst] = px[1] as f32 / 255.0; // G
                input[2 * plane + dst] = px[2] as f32 / 255.0; // B
            }
        }

        let shape = vec![1i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64];
        let tensor = Tensor::from_array((shape, input)).map_err(e)?;
        let outputs = self.session.run(ort::inputs!["images" => tensor]).map_err(e)?;
        let (out_shape, data) = outputs["output0"].try_extract_tensor::<f32>().map_err(e)?;

        // Expect [1, N, 6]. Walk rows.
        let cols = *out_shape.last().unwrap() as usize; // 6
        let rows = data.len() / cols;

        let mut dets = Vec::new();
        for i in 0..rows {
            let o = i * cols;
            let score = data[o + 4];
            // Rows are score-sorted; once we drop below threshold we can stop.
            if score < conf_threshold {
                break;
            }
            // Undo letterbox back into original-frame coordinates.
            let x1 = ((data[o] - pad_x as f32) / r).clamp(0.0, ow);
            let y1 = ((data[o + 1] - pad_y as f32) / r).clamp(0.0, oh);
            let x2 = ((data[o + 2] - pad_x as f32) / r).clamp(0.0, ow);
            let y2 = ((data[o + 3] - pad_y as f32) / r).clamp(0.0, oh);
            dets.push(Detection {
                x1,
                y1,
                x2,
                y2,
                score,
                class_id: data[o + 5] as usize,
            });
        }
        Ok(dets)
    }
}
