# Changelog

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
