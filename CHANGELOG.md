# Changelog

## v0.3.0 (2026-07-05)

### Changed
- **millow no longer depends on `moonbitlang/x` or `mizchi/image`** — the library is now fully self-contained (only uses `moonbitlang/core`). The full-featured demo (with PNG/JPEG I/O via `mizchi/image` and `moonbitlang/x/fs`) has been moved to `examples/` as a standalone project. `cmd/main` now contains a lightweight synthetic-image demo that only requires `millow` itself.

### CI/CD
- Simplified `.github/workflows/ci.yml` — removed `moon add` dependency installation steps since millow has no third-party deps. CI now runs `moon fmt --check`, `moon test`, and multi-backend builds (wasm-gc / wasm / js / native).

## v0.2.1 (2026-07-04)

### Performance
Algorithm-level optimizations across the filter, edge, threshold, morphology, and feature modules. All 122 tests pass with unchanged tolerances (no shortcuts, no widened tolerances); the `.mbti` diff only adds one internal helper (`gaussian_blur_double3`).

- **HOG** — 500.90ms → 10.76ms (**46.5x**): precompute flat luma array, inline gradients, flatten gx/gy buffers.
- **Sauvola threshold** — 94.69ms → 6.46ms (**14.7x**): integral image (O(n·w²) → O(n)).
- **Sobel edge** — 55.52ms → 11.84ms (**4.7x**): inlined `grad` with precomputed Reflect border coordinates.
- **Canny edge** — 114.31ms → 33.12ms (**3.5x**): stack-based hysteresis flood fill (O(n) instead of O(n·iterations)) + flat arrays.
- **Corner Harris / Shi-Tomasi** — 72.21ms → 24.31ms (**3.0x**): shared `structure_tensor` helper using fused 3-way Gaussian smoothing (`gaussian_blur_double3`).
- **Gaussian blur** — 51.45ms → 17.57ms (**2.9x**): interior/border split in `conv_1d`, inlined `at_clamped`, interleaved 3-channel processing.
- **Bilateral filter** — 331.01ms → 175.26ms (**1.9x**): precomputed spatial weights, flat RGB array, hoisted `reflect_coord` per row, `exp(-(a+b)) → exp(-a)·exp(-b)`.
- **Median filter / Dilate** — 1.2x: preallocated buffers + interior/border split.

### Documentation
- Added "Coordinate system" section to `AGENTS.md`, `README.mbt.md`, and `README_zh.md` documenting the uniform `(h, w)` dimension order, `(y, x)` coordinate order, top-left origin, drawing centers, translations, contours, and image-moment convention.
- Clarified `moments()` comment to state the `m_pq = Σ y^p · x^q · luma` convention explicitly.

### CI/CD
- Added `.github/workflows/ci.yml` running `moon fmt --check` and `moon test`

### Fixed
- `moments()` comment now explicitly states the `(y, x)` axis convention.

## v0.2.0 (2026-07-04)

### Added
- **Core image type** — `Image` with construction, pixel access, cloning, channel split/merge, and sub-images.
- **Color** — grayscale (flat & weighted), invert, tint, BGR, alpha flatten, HSV/YCbCr conversion, LUT application, alpha compositing.
- **Geometry** — crop, flips, 90/180/270 rotation, arbitrary rotation, translation, affine transform, shear, resize (nearest / bilinear / bicubic), rescale, fit/cover, thumbnails, padding.
- **Enhancement** — brightness, contrast, gamma, normalize, auto-contrast, standardize, sharpen, unsharp mask.
- **Threshold & histogram** — fixed threshold, Otsu, Sauvola, histogram (gray & color), equalization, CLAHE, histogram matching, correlation.
- **Filters & edges** — convolution, box/Gaussian blur, median/min/max filter, bilateral filter, Sobel, Scharr, Prewitt, Laplacian, Canny.
- **Morphology** — erode, dilate, open, close, gradient, top-hat, black-hat, skeletonize, hit-or-miss.
- **Feature detection** — LBP, HOG, Harris corner, Shi-Tomasi corner.
- **Measurement** — connected components, find contours, moments, Hu moments, region properties, pixel counting.
- **Data augmentation** — random crop, flip, rotate, brightness/contrast/gamma adjustment, Gaussian/salt-pepper noise, color jitter, composable pipeline, weighted random choice.
- **Metrics** — MSE, PSNR, SSIM.
- **Drawing** — pixels, lines, rectangles, circles, ellipses, polygons, flood fill.
- **I/O** — PPM/PGM serialization, pluggable `Encoder`/`Decoder` registry.
- **Border handling** — `Replicate`, `Reflect`, `Wrap`, `Constant` modes for affine transforms and interpolation.
- **Python alignment tests** — verification against skimage/Pillow/numpy reference implementations for 48+ functions.

### Fixed
- `random_color_jitter` now correctly implements saturation scaling and hue shifting (previously ignored parameters).
- `affine_transform` Nearest interpolation now handles all border modes correctly (previously returned wrong pixel for non-Constant modes).
- `corner_harris`/`corner_shi_tomasi` initialization optimization (removed wasteful computation).
- `moments`/`hu_moments` alignment tests now use value-level comparisons (previously only checked array length).

### Documentation
- English README with full API documentation and examples.
- Chinese README translation (README_zh.md).
- Roadmap document (docs/roadmap.md).

### Breaking Changes
- Removed unused `_block_size` parameter from `corner_harris` and `corner_shi_tomasi`.
- Removed unused `_ksize` parameter from `corner_harris`.
