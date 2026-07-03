///|
test "canny preserves dimensions" {
  let img = Image::from_pixel(5, 5, 128, 128, 128, 255)
  let out = canny(img, 50.0, 150.0)
  assert_eq(out.height(), 5)
  assert_eq(out.width(), 5)
}
