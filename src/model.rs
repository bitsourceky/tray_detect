//! YOLO26s-seg ONNX inference: letterbox preprocess -> ort -> decode + NMS.
//!
//! The model is *not* end-to-end: `output0` is `[1, 4 + NC + 32, N]`
//! (channel-first), one column per anchor holding `cx, cy, w, h` in 640x640
//! letterboxed pixels, `NC` already-sigmoided class scores, and 32 mask
//! coefficients. `output1` (mask protos) is ignored — we draw boxes only. So
//! here we take the best class per anchor, threshold, run class-wise NMS, and
//! map the boxes back to the original frame.

use crate::bins::CLASS_NAMES;
use anyhow::{anyhow, bail, Result};
use image::RgbImage;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

/// `ort`'s error type carries the (non-Send) builder/session on failure so it
/// can be recovered, which means it won't auto-convert into `anyhow::Error`.
/// We never recover, so collapse it to a message.
fn e(err: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("onnxruntime: {err}")
}

const INPUT_SIZE: usize = 640;
const PAD_VALUE: f32 = 114.0 / 255.0; // standard YOLO letterbox gray
const NUM_CLASSES: usize = CLASS_NAMES.len();
const NUM_MASK_COEFFS: usize = 32;
const IOU_THRESHOLD: f32 = 0.7; // ultralytics default
const MAX_DETECTIONS: usize = 300;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    /// Box in original-frame pixel coordinates.
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class_id: usize,
}

/// How a frame was fitted into the 640x640 input, so boxes can be mapped back.
#[derive(Clone, Copy, Debug)]
struct Letterbox {
    r: f32,
    pad_x: f32,
    pad_y: f32,
    ow: f32,
    oh: f32,
}

impl Letterbox {
    fn new(ow: u32, oh: u32) -> Self {
        let (ow, oh) = (ow as f32, oh as f32);
        let r = (INPUT_SIZE as f32 / ow).min(INPUT_SIZE as f32 / oh);
        let new_w = (ow * r).round() as u32;
        let new_h = (oh * r).round() as u32;
        Self {
            r,
            pad_x: ((INPUT_SIZE as u32 - new_w) / 2) as f32,
            pad_y: ((INPUT_SIZE as u32 - new_h) / 2) as f32,
            ow,
            oh,
        }
    }

    fn new_size(&self) -> (u32, u32) {
        (
            (self.ow * self.r).round() as u32,
            (self.oh * self.r).round() as u32,
        )
    }

    /// Undo the letterbox: 640-space box -> original-frame coordinates.
    fn unmap(&self, d: Detection) -> Detection {
        Detection {
            x1: ((d.x1 - self.pad_x) / self.r).clamp(0.0, self.ow),
            y1: ((d.y1 - self.pad_y) / self.r).clamp(0.0, self.oh),
            x2: ((d.x2 - self.pad_x) / self.r).clamp(0.0, self.ow),
            y2: ((d.y2 - self.pad_y) / self.r).clamp(0.0, self.oh),
            ..d
        }
    }
}

pub struct Model {
    session: Session,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
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
    /// coordinates, filtered to `conf_threshold` and NMS'd.
    pub fn detect(&mut self, frame: &RgbImage, conf_threshold: f32) -> Result<Vec<Detection>> {
        // --- letterbox: scale keeping aspect ratio, pad to 640x640 ---
        let lb = Letterbox::new(frame.width(), frame.height());
        let (new_w, new_h) = lb.new_size();
        let (pad_x, pad_y) = (lb.pad_x as u32, lb.pad_y as u32);

        let resized =
            image::imageops::resize(frame, new_w, new_h, image::imageops::FilterType::Triangle);

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
        let outputs = self
            .session
            .run(ort::inputs!["images" => tensor])
            .map_err(e)?;
        let (out_shape, data) = outputs["output0"].try_extract_tensor::<f32>().map_err(e)?;

        decode(out_shape, data, conf_threshold, &lb)
    }
}

/// Decode raw `output0` (`[1, 4 + NC + 32, N]`, channel-first) into
/// frame-space detections: per-anchor argmax class, threshold, class-wise NMS,
/// cap, un-letterbox.
fn decode(shape: &[i64], data: &[f32], conf: f32, lb: &Letterbox) -> Result<Vec<Detection>> {
    let channels = 4 + NUM_CLASSES + NUM_MASK_COEFFS;
    let n = match shape {
        [1, c, n] if *c as usize == channels && *n > 0 => *n as usize,
        _ => bail!(
            "unexpected output0 shape {shape:?}: expected [1, {channels}, N] \
             (4 box + {NUM_CLASSES} classes + {NUM_MASK_COEFFS} mask coeffs) — \
             is this the right best.onnx?"
        ),
    };
    if data.len() != channels * n {
        bail!(
            "output0 has {} values, expected {} for shape {shape:?}",
            data.len(),
            channels * n
        );
    }
    let at = |c: usize, i: usize| data[c * n + i];

    let mut candidates = Vec::new();
    for i in 0..n {
        let (class_id, score) =
            (0..NUM_CLASSES)
                .map(|c| (c, at(4 + c, i)))
                .fold(
                    (0, f32::MIN),
                    |best, cur| if cur.1 > best.1 { cur } else { best },
                );
        if score < conf {
            continue;
        }
        let (cx, cy, w, h) = (at(0, i), at(1, i), at(2, i), at(3, i));
        candidates.push(Detection {
            x1: cx - w / 2.0,
            y1: cy - h / 2.0,
            x2: cx + w / 2.0,
            y2: cy + h / 2.0,
            score,
            class_id,
        });
    }

    let mut kept = nms(candidates, IOU_THRESHOLD);
    kept.truncate(MAX_DETECTIONS);
    Ok(kept.into_iter().map(|d| lb.unmap(d)).collect())
}

/// Greedy class-wise NMS. Returns survivors sorted by score descending.
fn nms(mut dets: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for d in dets {
        if kept
            .iter()
            .all(|k| k.class_id != d.class_id || iou(k, &d) <= iou_threshold)
        {
            kept.push(d);
        }
    }
    kept
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let iw = (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0.0);
    let ih = (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0.0);
    let inter = iw * ih;
    let area = |d: &Detection| (d.x2 - d.x1).max(0.0) * (d.y2 - d.y1).max(0.0);
    let union = area(a) + area(b) - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const C: usize = 4 + NUM_CLASSES + NUM_MASK_COEFFS;

    /// Build a channel-first `[1, C, n]` tensor from `(cx, cy, w, h, class, score)` anchors.
    fn tensor(anchors: &[(f32, f32, f32, f32, usize, f32)]) -> (Vec<i64>, Vec<f32>) {
        let n = anchors.len();
        let mut data = vec![0.0; C * n];
        for (i, &(cx, cy, w, h, class, score)) in anchors.iter().enumerate() {
            data[i] = cx;
            data[n + i] = cy;
            data[2 * n + i] = w;
            data[3 * n + i] = h;
            data[(4 + class) * n + i] = score;
        }
        (vec![1, C as i64, n as i64], data)
    }

    /// Identity letterbox: a 640x640 frame maps 1:1.
    fn square() -> Letterbox {
        Letterbox::new(640, 640)
    }

    #[test]
    fn rejects_wrong_shape() {
        let lb = square();
        // Old end-to-end layout.
        assert!(decode(&[1, 300, 6], &[0.0; 1800], 0.5, &lb).is_err());
        // Wrong class count.
        assert!(decode(&[1, 4 + 7 + 32, 1], &[0.0; 43], 0.5, &lb).is_err());
        // Data length disagrees with shape.
        assert!(decode(&[1, C as i64, 2], &[0.0; C], 0.5, &lb).is_err());
        let err = decode(&[1, 300, 6], &[0.0; 1800], 0.5, &lb).unwrap_err();
        assert!(err.to_string().contains("[1, 73, N]"), "{err}");
    }

    #[test]
    fn thresholds_and_converts_cxcywh() {
        let (shape, data) = tensor(&[
            (100.0, 200.0, 40.0, 20.0, 5, 0.9),
            (300.0, 300.0, 10.0, 10.0, 2, 0.3), // below conf
        ]);
        let dets = decode(&shape, &data, 0.47, &square()).unwrap();
        assert_eq!(dets.len(), 1);
        let d = dets[0];
        assert_eq!(d.class_id, 5);
        assert_eq!((d.x1, d.y1, d.x2, d.y2), (80.0, 190.0, 120.0, 210.0));
        assert!((d.score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn picks_argmax_class() {
        let (shape, mut data) = tensor(&[(100.0, 100.0, 10.0, 10.0, 3, 0.6)]);
        data[4 + 20] = 0.8; // n == 1, so channel c is at index c
        let dets = decode(&shape, &data, 0.5, &square()).unwrap();
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 20);
    }

    #[test]
    fn nms_suppresses_same_class_overlap() {
        let (shape, data) = tensor(&[
            (100.0, 100.0, 50.0, 50.0, 2, 0.8),
            (102.0, 101.0, 50.0, 50.0, 2, 0.9), // heavy overlap, higher score
            (400.0, 400.0, 50.0, 50.0, 2, 0.7), // same class, far away
        ]);
        let dets = decode(&shape, &data, 0.5, &square()).unwrap();
        assert_eq!(dets.len(), 2);
        assert!((dets[0].score - 0.9).abs() < 1e-6);
        assert!((dets[1].score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_different_class_overlap() {
        let (shape, data) = tensor(&[
            (100.0, 100.0, 50.0, 50.0, 2, 0.9),
            (100.0, 100.0, 50.0, 50.0, 7, 0.8),
        ]);
        let dets = decode(&shape, &data, 0.5, &square()).unwrap();
        assert_eq!(dets.len(), 2);
    }

    #[test]
    fn caps_detections() {
        let anchors: Vec<_> = (0..400)
            .map(|i| {
                (
                    (i % 20) as f32 * 30.0 + 15.0,
                    (i / 20) as f32 * 30.0 + 15.0,
                    10.0,
                    10.0,
                    0,
                    0.9,
                )
            })
            .collect();
        let (shape, data) = tensor(&anchors);
        let dets = decode(&shape, &data, 0.5, &square()).unwrap();
        assert_eq!(dets.len(), MAX_DETECTIONS);
    }

    #[test]
    fn unletterboxes_to_frame_coords() {
        // 1280x720 -> r = 0.5, resized 640x360, pad_y = 140.
        let lb = Letterbox::new(1280, 720);
        assert_eq!((lb.r, lb.pad_x, lb.pad_y), (0.5, 0.0, 140.0));
        let (shape, data) = tensor(&[(320.0, 320.0, 100.0, 50.0, 0, 0.9)]);
        let d = decode(&shape, &data, 0.5, &lb).unwrap()[0];
        assert_eq!((d.x1, d.y1, d.x2, d.y2), (540.0, 310.0, 740.0, 410.0));

        // Boxes spilling into the padding clamp to the frame.
        let (shape, data) = tensor(&[(10.0, 150.0, 40.0, 40.0, 0, 0.9)]);
        let d = decode(&shape, &data, 0.5, &lb).unwrap()[0];
        assert_eq!((d.x1, d.y1), (0.0, 0.0));
    }

    /// Loads the real `best.onnx` and runs one inference on a synthetic frame.
    #[test]
    fn real_model_loads_and_decodes() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("best.onnx");
        let mut model = Model::load(&path).expect("load best.onnx");
        let frame = RgbImage::from_pixel(1280, 720, image::Rgb([128, 128, 128]));
        let dets = model.detect(&frame, 0.47).expect("detect");
        assert!(dets.len() <= MAX_DETECTIONS);
    }
}
