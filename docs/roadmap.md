# Millow Roadmap

**Last updated**: 2026-07-11 | **Current version**: v0.3.1

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
  translation, affine transform, shear, resize (nearest / bilinear / bicubic),
  rescale, fit/cover, thumbnails, and padding.
- **Enhancement** — brightness, contrast, gamma, normalize, auto-contrast,
  standardize, sharpen, and unsharp mask.
- **Threshold & histogram** — fixed threshold, Otsu, Sauvola, histogram
  (gray & color), equalization, CLAHE, and histogram matching.
- **Filters & edges** — convolution, box/Gaussian blur, median/min/max,
  bilateral filter, Sobel, Scharr, Prewitt, Laplacian, and Canny.
- **Morphology** — erode, dilate, open, close, gradient, top-hat, black-hat,
  skeletonize, and hit-or-miss.
- **Feature detection** — LBP, HOG, Harris corner, and Shi-Tomasi corner.
- **Measurement** — connected components, find contours, moments, Hu moments,
  region properties, and pixel counting.
- **Data augmentation** — random crop, flip, rotate, brightness/contrast/gamma
  adjustment, Gaussian/salt-pepper noise, color jitter, and composable
  pipelines with weighted random choice.
- **Metrics** — MSE, PSNR, SSIM.
- **Drawing** — pixels, lines, rectangles, circles, ellipses, polygons, and
  flood fill.
- **I/O** — PPM/PGM serialization plus a pluggable `Encoder`/`Decoder` registry.
- **Border handling** — `Replicate`, `Reflect`, `Wrap`, and `Constant` modes
  for affine transforms and interpolation.

### Test Coverage

- 122 tests across blackbox, alignment, and whitebox suites — all passing.
- 48 functions verified against Python reference implementations
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

- **Projection profiles** — horizontal and vertical projection for text-line
  detection and skew estimation.
- **Deskew** — detect and correct document skew using projection-profile
  variance maximization.
- **Auto-crop** — remove blank borders from scanned documents.
- **Small object removal** — filter out noise components by area threshold
  after binarization.
- **Perspective transform** — 8-parameter homography for document
  trapezoid correction.
- **Niblack threshold** — adaptive local thresholding
  (`T = mean + k * std`), reusing the integral-image infrastructure.

### API Cleanup

- **Unify `PadMode` and `BorderMode`** — the two enums are identical; merge
  into a single `BorderMode` (breaking change, deferred to a major version).
- **Brightness semantics** — align `adjust_brightness` with Pillow's additive
  semantics (currently multiplicative).
- **Color function cleanup** — remove redundant grayscale variants and
  consolidate the color conversion API surface.

### Advanced Features

- **Guided filter** — O(n) edge-preserving filter for background normalization.
- **Line removal** — detect and remove table/divider lines in document images.
- **Model input preprocessing** — `resize_to_model_input` and `letterbox`
  utilities for fixed-size model input with aspect-ratio preservation.

---

## Versioning

Millow follows semantic versioning. During the 0.x series, breaking changes
may occur between minor versions. The first stable 1.0 release will lock the
public API surface.
