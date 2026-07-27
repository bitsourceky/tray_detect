//! Overlay rendering: draw detection boxes + bin labels onto an RGB frame.

use crate::bins::{bin_for_class, Bin, CLASS_NAMES};
use crate::model::Detection;
use ab_glyph::{FontRef, PxScale};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut, text_size};
use imageproc::rect::Rect;

// Lato (SIL Open Font License) bundled in assets/. Embedded so the shipped
// binary is fully self-contained.
static FONT_BYTES: &[u8] = include_bytes!("../assets/font.ttf");

pub struct Overlay<'a> {
    font: FontRef<'a>,
}

impl Overlay<'_> {
    pub fn new() -> Self {
        let font = FontRef::try_from_slice(FONT_BYTES).expect("bundled font is valid");
        Self { font }
    }

    /// Draw all detections in place. Box + label colored by waste bin.
    pub fn draw(&self, frame: &mut RgbImage, dets: &[Detection]) {
        let scale = PxScale::from((frame.height() as f32 * 0.03).clamp(14.0, 32.0));
        for d in dets {
            let bin = bin_for_class(d.class_id);
            let color = Rgb(bin.color());

            let x = d.x1 as i32;
            let y = d.y1 as i32;
            let w = (d.x2 - d.x1).max(1.0) as u32;
            let h = (d.y2 - d.y1).max(1.0) as u32;
            let rect = Rect::at(x, y).of_size(w, h);

            // 3px box for visibility.
            for i in 0..3 {
                if w > 2 * i && h > 2 * i {
                    draw_hollow_rect_mut(
                        frame,
                        Rect::at(x + i as i32, y + i as i32)
                            .of_size(w - 2 * i, h - 2 * i),
                        color,
                    );
                }
            }
            let _ = rect;

            // Label: "BIN  ClassName 87%"
            let label = format!(
                "{}  {} {:.0}%",
                bin.label(),
                class_name(d.class_id),
                d.score * 100.0,
            );
            self.draw_label(frame, x, y, scale, color, bin, &label);
        }
    }

    fn draw_label(
        &self,
        frame: &mut RgbImage,
        x: i32,
        y: i32,
        scale: PxScale,
        color: Rgb<u8>,
        bin: Bin,
        text: &str,
    ) {
        let (tw, th) = text_size(scale, &self.font, text);
        let pad = 4i32;
        // Sit the label bar just above the box; tuck it inside if near the top.
        let bar_h = th as i32 + pad * 2;
        let bar_y = if y - bar_h >= 0 { y - bar_h } else { y };
        let bar_w = tw as i32 + pad * 2;

        let bar_x = x.max(0);
        let bar_y = bar_y.max(0);
        if bar_w > 0 && bar_h > 0 {
            draw_filled_rect_mut(
                frame,
                Rect::at(bar_x, bar_y).of_size(bar_w as u32, bar_h as u32),
                color,
            );
        }
        // White text reads on all three bin colors.
        let _ = bin;
        draw_text_mut(
            frame,
            Rgb([255, 255, 255]),
            bar_x + pad,
            bar_y + pad,
            scale,
            &self.font,
            text,
        );
    }

    /// Draw a small HUD line (e.g. FPS / detection count) top-left.
    pub fn draw_hud(&self, frame: &mut RgbImage, text: &str) {
        let scale = PxScale::from(18.0);
        let (tw, th) = text_size(scale, &self.font, text);
        draw_filled_rect_mut(
            frame,
            Rect::at(0, 0).of_size(tw + 8, th + 6),
            Rgb([0, 0, 0]),
        );
        draw_text_mut(frame, Rgb([255, 255, 255]), 4, 3, scale, &self.font, text);
    }
}

fn class_name(id: usize) -> &'static str {
    CLASS_NAMES.get(id).copied().unwrap_or("?")
}
