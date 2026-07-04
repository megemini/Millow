///|
test "lbp produces correct number of codes" {
  let img = Image::from_pixel(5, 5, 128, 128, 128, 255)
  let codes = lbp(img, 1, 8)
  assert_eq(codes.length(), 25)
}

///|
test "hog returns valid feature array" {
  let img = Image::from_pixel(16, 16, 128, 128, 128, 255)
  let features = hog(img, 8, 8, 4)
  assert_eq(features.length(), 0)
}

///|
test "corner_harris returns valid response" {
  let img = Image::from_pixel(5, 5, 128, 128, 128, 255)
  let response = corner_harris(img, 0.04)
  assert_eq(response.length(), 5)
}
