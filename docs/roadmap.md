# Millow Roadmap

**Last updated**: 2026-07-05 | **Current version**: v0.3.0

## Current State (v0.3.0)

Millow is a zero-FFI, cross-platform image-processing library for MoonBit.
It operates on in-memory RGBA8 buffers and runs on every backend: `wasm-gc`,
`wasm`, `js`, and `native`.

### Completed Features

- **Core image type** — `Image` with construction, pixel access, cloning,
  channel split/merge, and sub-images.
- **Color** — grayscale (flat & weighted), invert, tint, BGR, alpha flatten,
  HSV/YCbCr conversions, LUT application, and alpha compositing.
- **Geometry** — crop, flips, 90/180/270 rotation, arbitrary rotation,
  translation, affine transform, shear, **perspective transform**
  (8-DOF homography + 4-point convenience), resize (nearest / bilinear /
  bicubic), rescale, fit/cover, **`resize_to_model_input`**, **`letterbox`**,
  thumbnails, and padding.
- **Enhancement** — brightness, contrast, gamma, normalize, auto-contrast,
  standardize, sharpen, and unsharp mask.
- **Threshold & histogram** — fixed threshold, Otsu, Sauvola, Niblack,
  **Bernsen**, **Wolf**, **adaptive threshold (mean / Gaussian)**, histogram
  (gray & color), equalization, CLAHE, and histogram matching.
- **Filters & edges** — convolution, box/Gaussian blur, median/min/max,
  bilateral filter, **guided filter** (He et al. 2010, O(n) via integral
  images), **Wiener filter** (adaptive MMSE denoiser), **difference of
  Gaussians**, Sobel, Scharr, Prewitt, Laplacian, and Canny.
- **Morphology** — erode, dilate, open, close, gradient, top-hat, black-hat,
  skeletonize, hit-or-miss, `remove_small_objects` / `remove_small_holes`,
  and `remove_lines` (directional table-line removal).
- **Feature detection** — LBP, HOG, Harris corner, and Shi-Tomasi corner.
- **Measurement** — connected components, find contours, moments, Hu moments,
  region properties, pixel counting, **Hough line transform**, **distance
  transform**, **template matching** (full map + best match), and **K-means
  color quantization**.
- **Data augmentation** — random crop, flip, rotate, brightness/contrast/gamma
  adjustment, Gaussian/salt-pepper noise, color jitter, and composable
  pipelines with weighted random choice.
- **Metrics** — MSE, PSNR, SSIM.
- **Drawing** — pixels, lines, rectangles, circles, ellipses, polygons, and
  flood fill.
- **I/O** — PPM/PGM serialization plus a pluggable `Encoder`/`Decoder` registry.
- **Border handling** — `Replicate`, `Reflect`, `Wrap`, and `Constant` modes
  for affine transforms and interpolation.
- **OCR preprocessing** — projection profiles (horizontal/vertical, luma/binary),
  skew detection and deskew (projection-profile variance maximization with
  two-stage search), auto-crop (blank border removal), small object removal
  and hole filling.

### Test Coverage

- 158 tests across blackbox, alignment, and whitebox suites — all passing.
- 60+ functions verified against Python reference implementations
  (skimage / Pillow / numpy).
- Alignment tests use exact byte matching for integer operations and tight
  tolerances (±1 or ±2) for floating-point operations.

---

## Future Plans

### Algorithm Correctness

- **SSIM** — refactor to use local windowed statistics (currently computes
  global statistics). Requires an integral-image utility.
- **Integral image API** — extract the internal integral-image computation
  into a public API. This is foundational for several performance and
  algorithm improvements (Sauvola, Niblack, guided filter).
- **`regionprops`** — extend beyond the current 4 fields (label/area/centroid/
  bbox) to include perimeter, eccentricity, solidity, and orientation needed
  for document analysis.

### Performance Optimization

- **Median/min/max filters** — replace the current O(n·w²) windowed approach
  with sliding histogram (median) or monotonic deque (min/max) for O(n)
  performance.
- **RNG seeding** — add optional `seed` parameters to all `random_*`
  functions for reproducible augmentation, and introduce an `Augmenter`
  struct to share RNG state across a pipeline.

### OCR Preprocessing

- ~~**Projection profiles** — horizontal and vertical projection for text-line
  detection and skew estimation.~~ ✅ Done (v0.3.0)
- ~~**Deskew** — detect and correct document skew using projection-profile
  variance maximization.~~ ✅ Done (v0.3.0)
- ~~**Auto-crop** — remove blank borders from scanned documents.~~ ✅ Done (v0.3.0)
- ~~**Small object removal** — filter out noise components by area threshold
  after binarization.~~ ✅ Done (v0.3.0, includes `remove_small_holes`)
- ~~**Perspective transform** — 8-parameter homography for document
  trapezoid correction.~~ ✅ Done (v0.3.0)
- ~~**Niblack threshold** — adaptive local thresholding
  (`T = mean + k * std`), reusing the integral-image infrastructure.~~ ✅ Done (v0.3.0)

### API Cleanup

- **Unify `PadMode` and `BorderMode`** — the two enums are identical; merge
  into a single `BorderMode` (breaking change, deferred to a major version).
- **Brightness semantics** — align `adjust_brightness` with Pillow's additive
  semantics (currently multiplicative).
- **Color function cleanup** — remove redundant grayscale variants and
  consolidate the color conversion API surface.

### Advanced Features

- ~~**Guided filter** — O(n) edge-preserving filter for background normalization.~~ ✅ Done (v0.3.0)
- ~~**Line removal** — detect and remove table/divider lines in document images.~~ ✅ Done (v0.3.0)
- ~~**Model input preprocessing** — `resize_to_model_input` and `letterbox`
  utilities for fixed-size model input with aspect-ratio preservation.~~ ✅ Done (v0.3.0)

---

## Versioning

Millow follows semantic versioning. During the 0.x series, breaking changes
may occur between minor versions. The first stable 1.0 release will lock the
public API surface.
