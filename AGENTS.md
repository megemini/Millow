# Project Agents.md Guide

This is a [MoonBit](https://docs.moonbitlang.com) project.

You can browse and install extra skills here:
<https://github.com/moonbitlang/skills>

## Project Structure

- MoonBit packages are organized per directory; each directory contains a
  `moon.pkg` file listing its dependencies. Each package has its files and
  blackbox test files (ending in `_test.mbt`) and whitebox test files (ending
  in `_wbtest.mbt`).

- In the toplevel directory, there is a `moon.mod` file listing module
  metadata.

## Coding convention

- MoonBit code is organized in block style, each block is separated by `///|`,
  the order of each block is irrelevant. In some refactorings, you can process
  block by block independently.

- Try to keep deprecated blocks in file called `deprecated.mbt` in each
  directory.

## Tooling

- `moon fmt` is used to format your code properly.

- `moon ide` provides project navigation helpers like `peek-def`, `outline`, and
  `find-references`. See $moonbit-agent-guide for details.

- `moon info` is used to update the generated interface of the package, each
  package has a generated interface file `.mbti`, it is a brief formal
  description of the package. If nothing in `.mbti` changes, this means your
  change does not bring the visible changes to the external package users, it is
  typically a safe refactoring.

- In the last step, run `moon info && moon fmt` to update the interface and
  format the code. Check the diffs of `.mbti` file to see if the changes are
  expected.

- Run `moon test` to check tests pass. MoonBit supports snapshot testing; when
  changes affect outputs, run `moon test --update` to refresh snapshots.

- Prefer `assert_eq` or `assert_true(pattern is Pattern(...))` for results that
  are stable or very unlikely to change. For snapshot tests that record
  structured debugging output, derive `Debug` and use `debug_inspect`, rather
  than deriving `Show` for debugging. For solid, well-defined results (e.g.
  scientific computations), prefer assertion tests. You can use
  `moon coverage analyze > uncovered.log` to see which parts of your code are
  not covered by tests.

---

## millow project conventions

The sections below are specific to the **millow** image-processing library.
Follow them when adding features or modifying algorithms so that new code
stays consistent with the existing API and the alignment test suite.

### Architecture

```
millow/
├── src/              # implementation package (megemini/millow/src)
├── millow.mbt        # root facade: re-exports the public API
├── test/             # blackbox tests for the public API
├── test_alignment/   # alignment tests against Python reference
├── cmd/main/         # lightweight synthetic-image demo (millow-only)
└── examples/         # standalone full demo (PNG/JPEG I/O via mizchi/image)
```

- The root package (`megemini/millow`) is a **thin facade** that re-exports
  `@src` via `pub using`. All public API lives in `src/`. When adding a new
  public function, define it in `src/` and re-export it from `millow.mbt`.
- Downstream users import `megemini/millow` and access everything through
  `@millow`.

### Image format

- Images are stored internally as **RGBA8** in a flat `Array[Byte]` of length
  `h * w * 4`, row-major. The byte offset of pixel `(y, x)` channel `c` is
  `(y * w + x) * 4 + c` (use `Image::offset(y, x)` for the pixel base).
- `Image::from_data(data, h, w)` validates `data.length() == h * w * 4` and
  raises `ImageError` on mismatch.
- Filters operate on **RGB channels (0..2)** only; the alpha channel is copied
  unchanged from the source pixel unless the operation explicitly transforms
  it (e.g. `flatten_alpha`, `composite_over`).

### Coordinate system

millow uses a **single, uniform coordinate convention** across the entire API.
Violating it is the most common source of off-by-one and axis-swap bugs.

- **Dimension order**: always `(h, w)` — height first, width second.
  Examples: `Image::new(h, w)`, `Image::from_pixel(h, w, ...)`,
  `Image::from_data(data, h, w)`, `resize(img, dst_h, dst_w, interp)`,
  `crop(img, y, x, h, w)`, `Image::shape() -> (h, w)`.

- **Coordinate order**: always `(y, x)` — row first, column second.
  `y` is the vertical axis (row), `x` is the horizontal axis (column).
  Examples: `Image::offset(y, x)`, `Image::pixel_at(y, x)`,
  `draw_pixel(img, y, x, ...)`, `luma_at(img, y, x)`, `flood_fill(img, y, x, ...)`.

- **Origin**: top-left corner. Pixel `(0, 0)` is the top-left of the image;
  `y` increases downward, `x` increases rightward.

- **Valid ranges**: `y ∈ [0, h-1]`, `x ∈ [0, w-1]`.

- **Drawing centers**: circle/ellipse centers use `(cy, cx)` order.
  Example: `draw_circle(img, cy, cx, radius, ...)`,
  `draw_ellipse(img, cy, cx, ry, rx, ...)`.

- **Translations**: use `(dy, dx)` order.
  Example: `translate(img, dy, dx, interp)`.

- **Contours**: `find_contours` returns arrays of `(y, x)` tuples.

- **Image moments**: use the `(y, x)` convention where
  `m_pq = Σ y^p · x^q · luma`. The first index `p` is the row (y) exponent,
  the second `q` is the column (x) exponent. This differs from some textbook
  conventions (`m_pq = Σ x^p · y^q`), but Hu moments are invariant to the
  axis swap, so `hu_moments()` results are unaffected.

- **Loop pattern**: always `for y = 0; y < h; y = y + 1` (outer) /
  `for x = 0; x < w; x = x + 1` (inner).

- **Note on `rotate_90`/`rotate_270`**: these correctly swap the output
  dimensions — `Image::new(img.w, img.h)` — because a 90°/270° rotation
  transposes the image.

- **Note on PPM/PGM format**: the PPM header writes `width height` (e.g.
  `P6\n640 480\n255\n`) per the Netpbm specification, which is the opposite
  of millow's internal `(h, w)` order. This is a format requirement, not an
  inconsistency.

### Numeric conventions

- **Luma** uses integer BT.601: `(R*77 + G*150 + B*29) >> 8`. Use the
  `luma_at(img, y, x)` helper rather than re-deriving.
- **Rounding** is round-half-away-from-zero via `@math.round`, then clamped to
  `[0, 255]` by `round_byte`. When porting a reference implementation, do NOT
  use banker's rounding (numpy `np.round`) — use
  `sign(x) * floor(abs(x) + 0.5)`.
- **Clamping** helpers: `clampi` (int), `clampd` (double), `clamp_byte`
  (to `Byte` in `[0, 255]`).

### Border handling

- The default border mode for convolution, filters, gradients, and morphology
  is **replicate** (a.k.a. clamp / `mode='edge'` in numpy). Use `at_clamped`
  / `luma_at` which clamp coordinates to `[0, h-1] × [0, w-1]`.
- `pad` supports four `PadMode`s:
  - `Constant(r, g, b, a)` — fill with the given color.
  - `Replicate` — copy the nearest edge pixel (= numpy `mode='edge'`).
  - `Reflect` — mirror with **edge duplication** (= OpenCV `BORDER_REFLECT` =
    numpy `mode='symmetric'`, **not** numpy `mode='reflect'`).
  - `Wrap` — tile (= numpy `mode='wrap'`).
- `box_blur` **shrinks** border windows rather than clamping coordinates: the
  window bounds are `clamp(y-r, 0, h-1) .. clamp(y+r, 0, h-1)`, so edge pixels
  average fewer samples. Match this when writing a reference.

### Filter & resize conventions

- `gaussian_blur(img, sigma)`: kernel size = `clampi(ceil(sigma*3)*2+1, 3, 99)`,
  weights `exp(-d²/(2σ²))` normalised, applied as a separable 1D×1D pass. This
  differs from skimage's default `truncate=4.0`.
- `resize`:
  - **Nearest**: `sy = clampi(y * h / dst_h, 0, h-1)` (integer division).
  - **Bilinear**: Pillow-style separable convolution. Weights are precomputed
    per axis via `precompute_bilinear_weights` (triangle filter `1 - |x|` on a
    support of `max(1, scale)`), then applied as a horizontal pass followed by
    a vertical pass. This matches Pillow's `Image.resize` bilinear output.
  - **Bicubic**: centre-aligned coordinates
    `fy = clampd((y + 0.5) * h / dst_h - 0.5, 0, h-1)`, then interpolate with
    replicate border using the cubic kernel (a = -0.5).
- `convolve(img, kernel, normalize)`: `normalize=true` divides by the kernel
  sum (if non-zero); `normalize=false` divides by 1.0. The kernel anchor is
  the centre (`kh/2, kw/2`).
- Edge detectors (`sobel`, `prewitt`, `scharr`, `laplacian`) compute the
  gradient on **luma** using **`Reflect`** border handling (an exception to the
  default replicate rule above), then **max-normalize** the magnitude
  (`scale = 255 / max_val`) before emitting to RGB. `sobel`/`prewitt`/`scharr`
  use `sqrt(gx² + gy²)`; `laplacian` uses `|value|`.
- Morphology (`erode`/`dilate`) takes min/max over the structuring element on
  each RGB channel independently. `Kernel::Cross(n)` is a plus shape,
  `Kernel::Square(n)` is the full `(2r+1)²` box.

### Alignment testing (`test_alignment/`)

`test_alignment/` verifies millow's output against a Python reference
(numpy / skimage / Pillow) that implements the same algorithms.

**Workflow** (run after changing any algorithm):

```bash
source $HOME/venv310/bin/activate   # project Python env
python test_alignment/generate_fixtures.py  # regenerates fixtures_test.mbt
moon test                                   # all tests must pass
```

- `generate_fixtures.py` holds the **reference implementations** — when you
  change a millow algorithm, update the matching Python function there too,
  regenerate, and check that `fixtures_test.mbt` diffs are expected.
- Tests use `assert_img_eq` (exact byte match) for integer operations and
  `assert_img_close(img, expected, 1)` for floating-point operations where
  `@math.exp`/`pow`/`sqrt` may round differently.
- **Do not** widen a tolerance to make a failing test pass without
  understanding the root cause — investigate the off-by-one first.

### When adding a new operation

1. Implement it in the appropriate `src/*.mbt` file.
2. Re-export it from `millow.mbt` (`pub using @src { your_fn }`).
3. Add a reference implementation to `generate_fixtures.py` and an expected
   output to `generate()`; regenerate `fixtures_test.mbt`.
4. Add an alignment test in the matching `*_alignment_test.mbt` file.
5. Run `moon info && moon fmt && moon test` and confirm the `.mbti` diff is
   intentional.
