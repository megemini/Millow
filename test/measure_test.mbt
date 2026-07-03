///|
test "connected_components returns valid labels" {
  let img = Image::from_pixel(3, 3, 255, 255, 255, 255)
  let (labels, num) = connected_components(img, 4)
  assert_eq(labels.length(), 3)
}

///|
test "moments returns correct number of values" {
  let img = Image::from_pixel(4, 4, 128, 128, 128, 255)
  let m = moments(img)
  assert_eq(m.length(), 10)
}

///|
test "hu_moments returns correct number of values" {
  let img = Image::from_pixel(4, 4, 128, 128, 128, 255)
  let hu = hu_moments(img)
  assert_eq(hu.length(), 7)
}
