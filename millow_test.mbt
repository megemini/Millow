// Blackbox smoke tests for the root facade: they exercise the public API that
// `millow.mbt` re-exports from the `src` package, confirming the facade is
// correctly wired. The exhaustive suite lives in the `test/` package.

///|
test "facade exposes a full processing pipeline" {
  let base = Image::from_pixel(8, 8, 30, 60, 90, 255)
  let drawn = draw_rect(base, 1, 1, 5, 5, 220, 40, 40, 255, true, 1)
  let gray = to_grayscale(drawn)
  let blurred = gaussian_blur(gray, 1.0)
  @debug.assert_eq(blurred.shape(), (8, 8))
  let edges = sobel(blurred)
  @debug.assert_eq(edges.shape(), (8, 8))
  let (thr, binary) = threshold_otsu(blurred)
  assert_true(thr.to_int() >= 0 && thr.to_int() <= 255)
  @debug.assert_eq(binary.shape(), (8, 8))
}

///|
test "facade re-exports types and I/O" {
  let img = resize(Image::from_pixel(4, 4, 10, 20, 30, 255), 2, 2, Nearest)
  @debug.assert_eq(img.shape(), (2, 2))
  let ppm = to_ppm(img)
  assert_eq(ppm[0], 'P'.to_int().to_byte())
}
