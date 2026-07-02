///|
/// A 3x3 image with a single bright center pixel on a black background.
fn dot_image() -> Image raise {
  let img = Image::from_pixel(3, 3, 0, 0, 0, 255)
  img.pixel_set(1, 1, 255, 255, 255, 255)
  img
}

///|
test "convolve identity preserves image" {
  let img = dot_image()
  let k = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
  let out = convolve(img, k, false)
  assert_eq(out.pixel_at(1, 1).unwrap().0, 255)
  assert_eq(out.pixel_at(0, 0).unwrap().0, 0)
}

///|
test "box_blur spreads the center dot" {
  let out = box_blur(dot_image(), 1)
  assert_true(out.pixel_at(0, 0).unwrap().0.to_int() > 0)
  @debug.assert_eq(out.shape(), (3, 3))
}

///|
test "gaussian_blur keeps shape and alpha" {
  let out = gaussian_blur(dot_image(), 1.0)
  @debug.assert_eq(out.shape(), (3, 3))
  assert_eq(out.pixel_at(0, 0).unwrap().3, 255)
}

///|
test "median filter removes lone pixel" {
  assert_eq(median_filter(dot_image(), 3).pixel_at(1, 1).unwrap().0, 0)
}

///|
test "min and max filters" {
  assert_eq(max_filter(dot_image(), 3).pixel_at(0, 0).unwrap().0, 255)
  assert_eq(min_filter(dot_image(), 3).pixel_at(1, 1).unwrap().0, 0)
}

///|
test "sharpen and unsharp keep shape" {
  @debug.assert_eq(sharpen(dot_image(), 1.0).shape(), (3, 3))
  @debug.assert_eq(unsharp_mask(dot_image(), 1.0, 1.0, 0).shape(), (3, 3))
}

///|
test "edge detectors produce a magnitude image" {
  let img = dot_image()
  @debug.assert_eq(sobel(img).shape(), (3, 3))
  @debug.assert_eq(prewitt(img).shape(), (3, 3))
  @debug.assert_eq(scharr(img).shape(), (3, 3))
  @debug.assert_eq(laplacian(img).shape(), (3, 3))
  assert_eq(sobel_x(img).length(), 3)
  assert_eq(sobel_y(img)[0].length(), 3)
}

///|
test "dilate and erode with cross and square" {
  let img = dot_image()
  assert_eq(dilate(img, Cross(3)).pixel_at(0, 1).unwrap().0, 255)
  assert_eq(erode(img, Square(3)).pixel_at(1, 1).unwrap().0, 0)
}

///|
test "morphology compositions and custom kernel" {
  let img = dot_image()
  @debug.assert_eq(morph_open(img, Cross(3)).shape(), (3, 3))
  @debug.assert_eq(morph_close(img, Cross(3)).shape(), (3, 3))
  @debug.assert_eq(morph_gradient(img, Cross(3)).shape(), (3, 3))
  @debug.assert_eq(top_hat(img, Cross(3)).shape(), (3, 3))
  @debug.assert_eq(black_hat(img, Cross(3)).shape(), (3, 3))
  let mask = [[false, true, false], [true, true, true], [false, true, false]]
  @debug.assert_eq(dilate(img, Custom(mask)).shape(), (3, 3))
}

///|
test "draw pixel, line, rect and circle" {
  let img = Image::from_pixel(5, 5, 0, 0, 0, 255)
  let p = draw_pixel(img, 2, 2, 255, 0, 0, 255)
  assert_eq(p.pixel_at(2, 2).unwrap().0, 255)
  let l = draw_line(img, 0, 0, 4, 4, 0, 255, 0, 255, 1)
  assert_eq(l.pixel_at(0, 0).unwrap().1, 255)
  assert_eq(l.pixel_at(4, 4).unwrap().1, 255)
  let r = draw_rect(img, 1, 1, 3, 3, 0, 0, 255, 255, true, 1)
  assert_eq(r.pixel_at(2, 2).unwrap().2, 255)
  let c = draw_circle(img, 2, 2, 2, 255, 255, 0, 255, true, 1)
  assert_eq(c.pixel_at(2, 2).unwrap().0, 255)
}

///|
test "util info, equals, mse and psnr" {
  let a = dot_image()
  let b = a.clone()
  assert_true(equals(a, b))
  assert_eq(mse(a, b), 0.0)
  assert_true(psnr(a, b) >= 99.0)
  let c = invert(a)
  assert_false(equals(a, c))
  assert_true(mse(a, c) > 0.0)
  assert_true(info(a).length() > 0)
}

///|
test "to_ppm, to_pgm and display bytes" {
  let img = Image::from_pixel(1, 2, 10, 20, 30, 255)
  let ppm = to_ppm(img)
  assert_eq(ppm[0], 'P'.to_int().to_byte())
  assert_eq(ppm[1], '6'.to_int().to_byte())
  // The trailing bytes are the raw RGB payload (2 pixels, 3 bytes each).
  assert_eq(ppm[ppm.length() - 6], 10)
  assert_eq(ppm[ppm.length() - 4], 30)
  assert_eq(to_pgm(img)[1], '5'.to_int().to_byte())
  let disp = to_display_bytes(img)
  assert_eq(disp.length(), 1 * 2 * 3)
  assert_eq(disp[0], 10)
}

///|
test "io registry encode via PpmEncoder" {
  let img = Image::from_pixel(1, 1, 1, 2, 3, 255)
  register_encoder("ppm", PpmEncoder::{  })
  let bytes = encode(img, "ppm")
  assert_eq(bytes[0], 'P'.to_int().to_byte())
  assert_true(raises(() => encode(img, "zzz") |> ignore))
}
