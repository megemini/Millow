///|
test "resize nearest and bilinear keep size" {
  let img = Image::from_pixel(4, 4, 80, 80, 80, 255)
  let up = resize(img, 8, 8, Nearest)
  @debug.assert_eq(up.shape(), (8, 8))
  assert_eq(up.pixel_at(3, 3).unwrap().0, 80)
  let down = resize(img, 2, 2, Bilinear)
  @debug.assert_eq(down.shape(), (2, 2))
  assert_eq(down.pixel_at(0, 0).unwrap().0, 80)
}

///|
test "rescale and resize_to_fit preserve ratio" {
  let img = Image::from_pixel(10, 20, 5, 5, 5, 255)
  @debug.assert_eq(rescale(img, 0.5, Nearest).shape(), (5, 10))
  let fit = resize_to_fit(img, 5, 5, Nearest)
  assert_eq(fit.width(), 5)
  assert_true(fit.height() <= 5)
}

///|
test "resize_to_cover crops to exact size" {
  let img = Image::from_pixel(10, 20, 5, 5, 5, 255)
  @debug.assert_eq(resize_to_cover(img, 8, 8, Nearest).shape(), (8, 8))
}

///|
test "pad constant, replicate, reflect and wrap" {
  let img = Image::from_pixel(2, 2, 100, 0, 0, 255)
  let p = pad(img, 1, 1, 1, 1, Constant(7, 8, 9, 10))
  @debug.assert_eq(p.shape(), (4, 4))
  @debug.assert_eq(p.pixel_at(0, 0).unwrap(), (7, 8, 9, 10))
  assert_eq(p.pixel_at(1, 1).unwrap().0, 100)
  assert_eq(pad(img, 1, 0, 0, 1, Replicate).pixel_at(0, 0).unwrap().0, 100)
  @debug.assert_eq(pad(img, 1, 1, 1, 1, Reflect).shape(), (4, 4))
  @debug.assert_eq(pad(img, 1, 1, 1, 1, Wrap).shape(), (4, 4))
}

///|
test "pad_to_size centers image" {
  let img = Image::from_pixel(2, 2, 1, 1, 1, 255)
  @debug.assert_eq(pad_to_size(img, 4, 4, Replicate).shape(), (4, 4))
}

///|
test "brightness, contrast and gamma" {
  let img = Image::from_pixel(1, 1, 100, 100, 100, 255)
  assert_eq(adjust_brightness(img, 50).pixel_at(0, 0).unwrap().0, 150)
  assert_eq(adjust_brightness(img, 500).pixel_at(0, 0).unwrap().0, 255)
  assert_eq(adjust_contrast(img, 2.0).pixel_at(0, 0).unwrap().0, 72)
  assert_eq(adjust_gamma(img, 1.0).pixel_at(0, 0).unwrap().0, 100)
}

///|
test "normalize and auto_contrast stretch range" {
  let img = Image::new(1, 2)
  img.pixel_set(0, 0, 50, 50, 50, 255)
  img.pixel_set(0, 1, 150, 150, 150, 255)
  let norm = normalize(img, 0, 255)
  assert_eq(norm.pixel_at(0, 0).unwrap().0, 0)
  assert_eq(norm.pixel_at(0, 1).unwrap().0, 255)
  assert_eq(auto_contrast(img, 0).pixel_at(0, 0).unwrap().0, 0)
}

///|
test "threshold and threshold_inv" {
  let img = Image::new(1, 2)
  img.pixel_set(0, 0, 10, 10, 10, 255)
  img.pixel_set(0, 1, 240, 240, 240, 255)
  let t = threshold(img, 128)
  assert_eq(t.pixel_at(0, 0).unwrap().0, 0)
  assert_eq(t.pixel_at(0, 1).unwrap().0, 255)
  let ti = threshold_inv(img, 128)
  assert_eq(ti.pixel_at(0, 0).unwrap().0, 255)
  assert_eq(ti.pixel_at(0, 1).unwrap().0, 0)
}

///|
test "otsu splits bimodal image" {
  let img = Image::new(1, 4)
  img.pixel_set(0, 0, 0, 0, 0, 255)
  img.pixel_set(0, 1, 0, 0, 0, 255)
  img.pixel_set(0, 2, 255, 255, 255, 255)
  img.pixel_set(0, 3, 255, 255, 255, 255)
  let (thr, out) = threshold_otsu(img)
  assert_true(thr.to_int() > 0 && thr.to_int() < 255)
  assert_eq(out.pixel_at(0, 0).unwrap().0, 0)
  assert_eq(out.pixel_at(0, 3).unwrap().0, 255)
}

///|
test "histogram counts and equalize" {
  let img = Image::from_pixel(2, 2, 0, 0, 0, 255)
  assert_eq(histogram(img)[0], 4)
  let hc = histogram_color(img)
  assert_eq(hc.length(), 3)
  assert_eq(hc[0][0], 4)
  @debug.assert_eq(equalize_histogram(img).shape(), (2, 2))
}
