///|
test "threshold_sauvola preserves dimensions" {
  let img = Image::from_pixel(5, 5, 128, 128, 128, 255)
  let out = threshold_sauvola(img, 3, 0.15)
  assert_eq(out.height(), 5)
  assert_eq(out.width(), 5)
}
