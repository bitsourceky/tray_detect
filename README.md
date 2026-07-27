# Tray Detect

Live school-lunch waste sorting. Opens a webcam, runs the trained YOLO26-nano
model on each frame, and draws boxes labeled with the waste **bin** each item
belongs to: **Compost**, **Recycle**, or **Trash**.

Single self-contained Rust binary — the ONNX Runtime and the overlay font are
bundled, so target machines need nothing installed.

## Run

```sh
cargo run --release                 # live on the default webcam
cargo run --release -- --camera 1   # a different camera
cargo run --release -- --conf 0.35  # confidence threshold (default 0.25)
cargo run --release -- --image tray.jpg   # one-shot on a still image → tray_out.png
```

Press **ESC** to quit the live window.

The `best.onnx` model file must sit next to the binary (or in the working
directory). On first launch the `ort` crate downloads a matching ONNX Runtime
build automatically.

### macOS camera permission

The first live run triggers a camera-permission prompt (TCC). If the app was
launched from a terminal, grant Terminal/iTerm camera access under
**System Settings → Privacy & Security → Camera**, then re-run.

## Model

- **Input:** `images`, `[1,3,640,640]`, RGB, normalized 0–1, letterboxed (gray padding).
- **Output:** `output0`, `[1,300,6]` — rows of `[x1,y1,x2,y2,score,class_id]` in
  640-space, already sorted and **NMS-free** (`end2end=True`). Postprocessing is
  just: threshold on score, undo the letterbox, map class → bin.
- **Classes (7):** Chips, Compost, Napkin, Plastic-Cups, Plastic-Wrap, Recycle, Utensils.

### Bin mapping

The 7 model classes collapse into 3 bins in `src/bins.rs::bin_for_class`. This
is the district-specific sorting policy — **review it** against your cafeteria's
actual waste contract. Current defaults:

| Class | Bin |
|---|---|
| Compost, Napkin | Compost |
| Recycle, Plastic-Cups | Recycle |
| Chips, Plastic-Wrap, Utensils | Trash |

## Layout

- `src/model.rs` — letterbox preprocess, ORT inference, output decode.
- `src/bins.rs` — class names + class→bin mapping and colors.
- `src/draw.rs` — box/label overlay rendering.
- `src/main.rs` — arg parsing, camera loop, still-image mode.
- `assets/font.ttf` — Lato Regular (SIL Open Font License), embedded for labels.

## License notes

`assets/font.ttf` is Lato, licensed under the SIL Open Font License 1.1.
Ultralytics YOLO models are AGPL-3.0 unless you hold a commercial license —
confirm your licensing before distributing the packaged app.
