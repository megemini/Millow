// Whitebox tests run within the root package scope. Since this package is a
// thin facade over `src`, they mainly assert that the re-exported helpers and
// type aliases resolve to the intended implementations.

///|
test "re-exported numeric helpers behave as in src" {
  assert_eq(clampi(15, 0, 10), 10)
  assert_eq(clampi(-3, 0, 10), 0)
  assert_eq(clampd(1.5, 0.0, 1.0), 1.0)
  assert_eq(clamp_byte(300), 255)
  assert_eq(round_byte(1.6), 2)
}

///|
test "Image alias is usable for construction and access" {
  let img = Image::from_pixel(2, 2, 4, 5, 6, 255)
  @debug.assert_eq(img.shape(), (2, 2))
  @debug.assert_eq(img.pixel_at(0, 0).unwrap(), (4, 5, 6, 255))
}
