///|
test "random_noise_gaussian preserves dimensions" {
  let img = Image::from_pixel(5, 5, 128, 128, 128, 255)
  let out = random_noise_gaussian(img, 10.0)
  assert_eq(out.height(), 5)
  assert_eq(out.width(), 5)
}

///|
test "random_noise_salt_pepper preserves dimensions" {
  let img = Image::from_pixel(5, 5, 128, 128, 128, 255)
  let out = random_noise_salt_pepper(img, 0.05)
  assert_eq(out.height(), 5)
  assert_eq(out.width(), 5)
}
