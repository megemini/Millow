///|
test "rotate_any produces valid dimensions" {
  let img = Image::from_pixel(5, 5, 128, 128, 128, 255)
  let out = rotate_any(img, 45.0, Bilinear)
  assert_true(out.height() > 0)
  assert_true(out.width() > 0)
}

///|
test "translate preserves dimensions" {
  let img = Image::from_pixel(5, 5, 128, 128, 128, 255)
  let out = translate(img, 1.0, 1.0, Bilinear)
  assert_eq(out.height(), 5)
  assert_eq(out.width(), 5)
}

///|
test "affine_transform produces specified dimensions" {
  let img = Image::from_pixel(5, 5, 128, 128, 128, 255)
  let matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
  let out = affine_transform(img, matrix, 5, 5, Bilinear)
  assert_eq(out.height(), 5)
  assert_eq(out.width(), 5)
}
