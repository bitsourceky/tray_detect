//! Overlay rendering: draw detection boxes, bin labels, and the per-bin
//! summary panel onto an RGB frame.

use crate::bins::{bin_for_class, class_name, Bin};
use crate::model::Detection;
use ab_glyph::{FontRef, PxScale};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut, text_size};
use imageproc::rect::Rect;

// Lato (SIL Open Font License) bundled in assets/. Embedded so the shipped
// binary is fully self-contained.
static FONT_BYTES: &[u8] = include_bytes!("../assets/font.ttf");

const WHITE: Rgb<u8> = Rgb([255, 255, 255]);
const BLACK: Rgb<u8> = Rgb([0, 0, 0]);

pub struct Overlay<'a> {
    font: FontRef<'a>,
}

impl Overlay<'_> {
    pub fn new() -> Self {
        let font = FontRef::try_from_slice(FONT_BYTES).expect("bundled font is valid");
        Self { font }
    }

    /// Draw all detections in place (box + label colored by waste bin), then
    /// the summary panel top-left.
    pub fn draw(&self, frame: &mut RgbImage, dets: &[Detection]) {
        let scale = PxScale::from((frame.height() as f32 * 0.03).clamp(14.0, 32.0));
        for d in dets {
            let bin = bin_for_class(d.class_id);
            let color = Rgb(bin.color());

            let x = d.x1 as i32;
            let y = d.y1 as i32;
            let w = (d.x2 - d.x1).max(1.0) as u32;
            let h = (d.y2 - d.y1).max(1.0) as u32;

            // 3px box for visibility.
            for i in 0..3 {
                if w > 2 * i && h > 2 * i {
                    draw_hollow_rect_mut(
                        frame,
                        Rect::at(x + i as i32, y + i as i32).of_size(w - 2 * i, h - 2 * i),
                        color,
                    );
                }
            }

            // Label: "BIN  ClassName 87%"
            let label = format!(
                "{}  {} {:.0}%",
                bin.label(),
                class_name(d.class_id),
                d.score * 100.0,
            );
            self.draw_label(frame, x, y, scale, color, &label);
        }
        self.draw_summary(frame, dets);
    }

    fn draw_label(
        &self,
        frame: &mut RgbImage,
        x: i32,
        y: i32,
        scale: PxScale,
        color: Rgb<u8>,
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
        draw_text_mut(
            frame,
            text_color_on(color),
            bar_x + pad,
            bar_y + pad,
            scale,
            &self.font,
            text,
        );
    }

    /// Translucent top-left panel: one line per bin listing
    /// `Name: NN.N%` for each detection, or `None`.
    fn draw_summary(&self, frame: &mut RgbImage, dets: &[Detection]) {
        let scale = PxScale::from(20.0);
        let pad = 8i32;
        let lines: Vec<(Bin, String)> = [Bin::Trash, Bin::Recycle, Bin::Compost]
            .into_iter()
            .map(|bin| {
                let items: Vec<String> = dets
                    .iter()
                    .filter(|d| bin_for_class(d.class_id) == bin)
                    .map(|d| format!("{}: {:.1}%", class_name(d.class_id), d.score * 100.0))
                    .collect();
                let items = if items.is_empty() {
                    "None".to_string()
                } else {
                    items.join(", ")
                };
                (bin, items)
            })
            .collect();

        let heads: Vec<String> = lines
            .iter()
            .map(|(bin, _)| format!("{}: ", bin_title(*bin)))
            .collect();
        let line_h = text_size(scale, &self.font, "Ag").1 as i32 + 6;
        let width = lines
            .iter()
            .zip(&heads)
            .map(|((_, items), head)| {
                text_size(scale, &self.font, head).0 + text_size(scale, &self.font, items).0
            })
            .max()
            .unwrap_or(0) as i32
            + pad * 2;
        let height = line_h * lines.len() as i32 + pad * 2;
        darken(frame, width as u32, height as u32);

        for (i, ((bin, items), head)) in lines.iter().zip(&heads).enumerate() {
            let y = pad + i as i32 * line_h;
            draw_text_mut(frame, Rgb(bin.color()), pad, y, scale, &self.font, head);
            let hx = pad + text_size(scale, &self.font, head).0 as i32;
            draw_text_mut(frame, WHITE, hx, y, scale, &self.font, items);
        }
    }

    /// Draw a small HUD line (e.g. FPS / detection count) bottom-left, clear
    /// of the summary panel.
    pub fn draw_hud(&self, frame: &mut RgbImage, text: &str) {
        let scale = PxScale::from(18.0);
        let (tw, th) = text_size(scale, &self.font, text);
        let y = frame.height().saturating_sub(th + 6) as i32;
        draw_filled_rect_mut(frame, Rect::at(0, y).of_size(tw + 8, th + 6), BLACK);
        draw_text_mut(frame, WHITE, 4, y + 3, scale, &self.font, text);
    }
}

fn bin_title(bin: Bin) -> &'static str {
    match bin {
        Bin::Compost => "Compost",
        Bin::Recycle => "Recycle",
        Bin::Trash => "Trash",
        Bin::Human => "Human",
        Bin::NeedsReview => "Needs Review",
    }
}

/// Black text on bright bars (green, orange), white otherwise.
fn text_color_on(bg: Rgb<u8>) -> Rgb<u8> {
    let [r, g, b] = bg.0.map(f32::from);
    if 0.299 * r + 0.587 * g + 0.114 * b > 150.0 {
        BLACK
    } else {
        WHITE
    }
}

/// Halve the brightness of the top-left `w`x`h` region (translucent black).
fn darken(frame: &mut RgbImage, w: u32, h: u32) {
    for y in 0..h.min(frame.height()) {
        for x in 0..w.min(frame.width()) {
            let p = frame.get_pixel_mut(x, y);
            p.0 = p.0.map(|c| c / 2);
        }
    }
}
