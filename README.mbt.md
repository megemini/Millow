# millow

A **zero-FFI, cross-platform image-processing library** for MoonBit. `millow`
works entirely on in-memory RGBA8 buffers (`Array[Byte]`, laid out `H × W × 4`)
and builds on every backend: `wasm-gc`, `wasm`, `js`, and `native`.

## Features

- **Core image type** — `Image` with construction, pixel access, cloning,
  channel split/merge, and sub-images.
- **Color** — grayscale (flat & weighted), invert, tint, BGR, alpha flatten,
  and `over` compositing.
- **Geometry** — crop, flips, 90/180/270 rotation, resize (nearest / bilinear /
  bicubic), rescale, fit/cover, thumbnails, and padding.
- **Enhancement** — brightness, contrast, gamma, normalize, auto-contrast,
  sharpen, and unsharp mask.
- **Threshold & histogram** — fixed threshold, Otsu, histogram (gray & color),
  and equalization.
- **Filters & edges** — convolution, box/Gaussian blur, median/min/max, Sobel,
  Scharr, Prewitt, and Laplacian.
- **Morphology** — erode, dilate, open, close, gradient, top-hat, black-hat.
- **Drawing** — pixels, lines, rectangles, and circles.
- **I/O** — PPM/PGM serialization plus a pluggable `Encoder`/`Decoder` registry.

## Project layout

```
millow/
├── src/              # implementation package (megemini/millow/src)
├── millow.mbt        # root facade: re-exports the public API (megemini/millow)
├── test/             # blackbox test package exercising the public API
├── test_alignment/   # alignment tests against Python (skimage/Pillow) reference
└── cmd/main/         # runnable end-to-end demo
```

The root package is a thin **facade** over `src`, so downstream users just
`import "megemini/millow"` and reach the whole API through `@millow`.

## Installation

```
moon add megemini/millow
```

Then import it in your package's `moon.pkg`:

```
import {
  "megemini/millow" @millow,
}
```

## Quick start

```mbt nocheck
///|
test "build, transform and inspect an image" {
  // A 64×64 canvas with a filled rectangle drawn on it.
  let base = Image::from_pixel(64, 64, 30, 60, 90, 255)
  let canvas = draw_rect(base, 8, 8, 40, 40, 220, 40, 40, 255, true, 1)

  // Grayscale → blur → edges.
  let gray = to_grayscale(canvas)
  let blurred = gaussian_blur(gray, 1.5)
  let edges = sobel(blurred)
  assert_eq(edges.shape(), (64, 64))

  // Otsu adaptive threshold.
  let (_, binary) = threshold_otsu(blurred)
  assert_eq(binary.shape(), (64, 64))

  // Downscale and serialize to PPM.
  let thumb = resize(canvas, 16, 16, Nearest)
  let ppm = to_ppm(thumb)
  assert_eq(ppm[0], 'P'.to_int().to_byte())
}
```

> In the examples above the API is called unqualified because they run inside
> the `millow` package itself. From another module, prefix each name with the
> import alias, e.g. `@millow.to_grayscale(img)`.

## Backends

`millow` contains no foreign function calls. It is verified to build on
`wasm-gc`, `wasm`, `js`, and `native`, and the test suite passes on each.

## Testing

```
moon test                 # run every test
moon test --target native # pick a backend
moon run cmd/main         # run the demo pipeline
```

### Alignment tests

`test_alignment/` verifies millow's output against a Python reference
(numpy / skimage / Pillow) that implements the same algorithms. The workflow
is:

1. `test_alignment/generate_fixtures.py` computes expected output bytes for
   small test images and writes them as `Array[Int]` literals in
   `fixtures_test.mbt`.
2. The MoonBit tests construct `Image`s from those fixtures, run each millow
   operation, and compare byte-for-byte (exact for integer ops, ±1 tolerance
   for floating-point rounding).

Regenerate the fixtures after changing an algorithm:

```
source /home/shun/venv310/bin/activate
python test_alignment/generate_fixtures.py
moon test
```

## License

Apache-2.0. See [LICENSE](LICENSE).