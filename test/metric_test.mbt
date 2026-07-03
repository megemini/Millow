///|
test "ssim returns valid score for identical images" {
  let img1 = Image::from_pixel(5, 5, 128, 128, 128, 255)
  let img2 = Image::from_pixel(5, 5, 128, 128, 128, 255)
  let score = ssim(img1, img2)
  assert_true(score >= 0.0 && score <= 1.0)
}
