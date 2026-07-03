///|
test "bilateral_filter preserves dimensions" {
  let img = Image::from_pixel(5, 5, 128, 128, 128, 255)
  let out = bilateral_filter(img, 5, 50.0, 5.0)
  assert_eq(out.height(), 5)
  assert_eq(out.width(), 5)
}
