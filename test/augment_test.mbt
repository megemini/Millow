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

///|
test "augment_pipeline applies a sequence of augmentations" {
  let img = Image::from_pixel(8, 8, 30, 60, 90, 255)
  let out = augment_pipeline(img, [
    FlipHorizontal,
    FlipVertical,
    Brightness(1.2),
    Contrast(1.3),
    Gamma(0.9),
    NoiseGaussian(5.0),
    NoiseSaltPepper(0.02),
    ColorJitter(0.1, 0.1, 0.0, 0.0),
  ])
  assert_eq(out.height(), 8)
  assert_eq(out.width(), 8)
}

///|
test "augment_pipeline resize changes dimensions" {
  let img = Image::from_pixel(8, 8, 30, 60, 90, 255)
  let out = augment_pipeline(img, [Resize(16, 4)])
  assert_eq(out.height(), 16)
  assert_eq(out.width(), 4)
}

///|
test "augment_pipeline rotate preserves dimensions" {
  let img = Image::from_pixel(8, 8, 30, 60, 90, 255)
  let out = augment_pipeline(img, [Rotate(15.0)])
  assert_true(out.height() > 0 && out.width() > 0)
}

///|
test "augment_pipeline crop adjusts dimensions" {
  let img = Image::from_pixel(8, 8, 30, 60, 90, 255)
  let out = augment_pipeline(img, [Crop(0, 0, 4, 4)])
  assert_eq(out.height(), 4)
  assert_eq(out.width(), 4)
}

///|
test "augment_pipeline raises on invalid crop" {
  let img = Image::from_pixel(8, 8, 30, 60, 90, 255)
  assert_true(
    raises(() => augment_pipeline(img, [Crop(0, 0, 20, 20)]) |> ignore),
  )
}

///|
test "augment_random_choice returns valid dimensions" {
  let img = Image::from_pixel(8, 8, 30, 60, 90, 255)
  let out = augment_random_choice(img, [
    (0.4, FlipVertical),
    (0.4, Gamma(0.8)),
    (0.2, ColorJitter(0.2, 0.2, 0.0, 0.0)),
  ])
  assert_eq(out.height(), 8)
  assert_eq(out.width(), 8)
}

///|
test "augment_random_choice empty list returns a clone" {
  let img = Image::from_pixel(4, 4, 30, 60, 90, 255)
  let out = augment_random_choice(img, [])
  assert_eq(out.height(), 4)
  assert_eq(out.width(), 4)
}

///|
test "process_batch applies a function to each image" {
  let imgs = [
    Image::from_pixel(2, 2, 10, 20, 30, 255),
    Image::from_pixel(2, 2, 40, 50, 60, 255),
  ]
  let out = process_batch(imgs, fn(img) { invert(img) })
  assert_eq(out.length(), 2)
  assert_eq(out[0].height(), 2)
  assert_eq(out[0].width(), 2)
  assert_eq(out[1].height(), 2)
  assert_eq(out[1].width(), 2)
}
