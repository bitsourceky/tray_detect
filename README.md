# Tray Detect

Live school-lunch waste sorting. Opens a webcam, runs the trained YOLO26s-seg
model (every 1.25 s by default), and draws boxes labeled with the waste **bin**
each item belongs to: **Compost**, **Recycle**, or **Trash**. A panel in the
top-left lists what's currently detected per bin.

Single self-contained Rust binary — the ONNX Runtime and the overlay font are
bundled, so target machines need nothing installed.

## Run

```sh
cargo run --release                 # live on the default webcam
cargo run --release -- --camera 1   # a different camera
cargo run --release -- --conf 0.35  # confidence threshold (default 0.47)
cargo run --release -- --interval 2 # seconds between inferences (default 1.25)
cargo run --release -- --model path/to/best.onnx   # explicit model file
cargo run --release -- --image tray.jpg   # one-shot on a still image → tray_out.png
```

Press **ESC** to quit the live window.

The camera preview renders every frame; inference runs once per `--interval`
and the latest boxes stay on screen in between.

Without `--model`, the app looks for `best.onnx` next to the executable first,
then in the current working directory, and lists the paths it tried if neither
exists. On first launch the `ort` crate downloads a matching ONNX Runtime
build automatically.

### macOS camera permission

The first live run triggers a camera-permission prompt (TCC). If the app was
launched from a terminal, grant Terminal/iTerm camera access under
**System Settings → Privacy & Security → Camera**, then re-run.

## Model

- **Input:** `images`, `[1,3,640,640]`, RGB, normalized 0–1, letterboxed (gray padding).
- **Output:** `output0`, `[1,73,8400]`, channel-first — per anchor: `cx,cy,w,h`
  in 640-space, 37 class scores (already sigmoid), 32 mask coefficients.
  `output1` (mask protos) is ignored; only boxes are drawn. The graph has **no
  NMS**, so `src/model.rs` does it: argmax class per anchor, threshold on
  `--conf`, class-wise NMS (IoU 0.7), cap at 300, undo the letterbox.
- **Classes (37):** Bread, Burrito, Can, Cardboard-Tray, Carton, Cheese,
  Chicken, Chip-Bag, Condiments, Corndog, Fries, Fruit, Generic-Wrapper, Gravy,
  Hamburger, Human, Lasagna, Macaroni, Mashed Potatoes, Meat, Napkin, Pizza,
  Plastic-Tray, Plastic-Wrapper, Quesadilla, Rice, Salad, Sauce, Snack-Cup,
  Spaghetti, Styrofoam-Cup, Styrofoam-Tray, Tortilla, Utensil, Vegetables,
  Waffle, Water-Bottle.

### Bin mapping

Classes map to bins by name in `src/bins.rs::bin_for_name`. This is the
district-specific sorting policy — **review it** against your cafeteria's
actual waste contract. Current defaults:

| Bin | Classes |
|---|---|
| Recycle (blue) | Plastic-Tray, Can, Snack-Cup, Water-Bottle, Chip-Bag, Carton, Cardboard-Tray |
| Trash (red) | Styrofoam-Tray, Utensil, Plastic-Wrapper, Napkin, Generic-Wrapper, Condiments, Styrofoam-Cup, Gravy, Sauce |
| Compost (green) | Hamburger, Fries, Chicken, Waffle, Salad, Fruit, Quesadilla, Bread, Vegetables, Burrito, Meat, Cheese, Spaghetti, Pizza, Corndog, Mashed Potatoes, Lasagna, Macaroni, Tortilla, Rice |
| Human (black) | Human — drawn, never counted toward a bin |

Any class id the mapping doesn't know is boxed in orange as **Needs Review**
rather than silently counted as Trash.

## Layout

- `src/model.rs` — letterbox preprocess, ORT inference, output decode + NMS.
- `src/bins.rs` — class names + class→bin mapping and colors.
- `src/draw.rs` — box/label overlay and per-bin summary panel.
- `src/main.rs` — arg parsing, camera loop, still-image mode.
- `assets/font.ttf` — Lato Regular (SIL Open Font License), embedded for labels.

## Releasing

Pushing a tag like `v0.1.0` triggers `.github/workflows/release.yml`, which
builds natively on macOS (Apple Silicon), Windows, and Linux runners — camera
(`nokhwa`) and window (`minifb`) libs bind to OS-native frameworks, so this
can't be cross-compiled from a single host — and attaches a zip per platform
(binary + `best.onnx` + README) to a GitHub Release for that tag:

```sh
git tag v0.1.0
git push origin v0.1.0
```

The workflow can also be run manually (Actions tab → Release → Run workflow)
to sanity-check a build without publishing a release; it only creates/updates
a GitHub Release when triggered by a tag push.

The download site (tray_detect_site) links to
`releases/latest/download/tray_detect-<platform>.zip`, so asset names must
stay `tray_detect-macos-arm64.zip`, `tray_detect-windows-x64.zip`,
`tray_detect-linux-x64.zip`, and `tray_detect-linux-arm64.zip`.

## License notes

`assets/font.ttf` is Lato, licensed under the SIL Open Font License 1.1.
Ultralytics YOLO models are AGPL-3.0 unless you hold a commercial license —
confirm your licensing before distributing the packaged app.
