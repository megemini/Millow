// Behavioral tests for BorderMode. Since `get_pixel` is not part of the
// public facade, border modes are exercised through `affine_transform` with
// a translation matrix that pushes sampling out of bounds.

///|
test "BorderMode type exists" {
  let _ : BorderMode? = None
}

///|
test "BorderMode variants are constructible" {
  let _r : BorderMode = Replicate
  let _f : BorderMode = Reflect
  let _w : BorderMode = Wrap
  let _c : BorderMode = Constant(0, 0, 0, 255)
}

///|
test "Constant border fills out-of-bounds with the given color" {
  // 3×3 white image; translation matrix [1,0,2, 0,1,0] shifts the source
  // right by 2, so destination columns 0 and 1 sample out of bounds.
  let img = Image::from_pixel(3, 3, 255, 255, 255, 255)
  let matrix = [1.0, 0.0, 2.0, 0.0, 1.0, 0.0]
  let out = affine_transform(
    img,
    matrix,
    3,
    3,
    Nearest,
    mode=Constant(0, 0, 0, 255),
  )
  // Columns 0 and 1 are out of bounds → constant black.
  let (r0, g0, b0, _) = out.pixel_at(1, 0).unwrap()
  assert_eq(r0, 0)
  assert_eq(g0, 0)
  assert_eq(b0, 0)
  let (r1, g1, b1, _) = out.pixel_at(1, 1).unwrap()
  assert_eq(r1, 0)
  assert_eq(g1, 0)
  assert_eq(b1, 0)
  // Column 2 is in bounds → white.
  let (r2, g2, b2, _) = out.pixel_at(1, 2).unwrap()
  assert_eq(r2, 255)
  assert_eq(g2, 255)
  assert_eq(b2, 255)
}

///|
test "Replicate border copies the nearest edge pixel" {
  let img = Image::from_pixel(3, 3, 255, 255, 255, 255)
  let matrix = [1.0, 0.0, 2.0, 0.0, 1.0, 0.0]
  let out = affine_transform(img, matrix, 3, 3, Nearest, mode=Replicate)
  // All columns replicate the left edge (white).
  for x = 0; x < 3; x = x + 1 {
    let (r, g, b, _) = out.pixel_at(1, x).unwrap()
    assert_eq(r, 255)
    assert_eq(g, 255)
    assert_eq(b, 255)
  }
}

///|
test "Reflect and Wrap stay within the image" {
  let img = Image::from_pixel(3, 3, 200, 100, 50, 255)
  let matrix = [1.0, 0.0, 2.0, 0.0, 1.0, 0.0]
  let out_reflect = affine_transform(img, matrix, 3, 3, Nearest, mode=Reflect)
  let out_wrap = affine_transform(img, matrix, 3, 3, Nearest, mode=Wrap)
  // Both modes map out-of-bounds columns back into the image, so every
  // output pixel should carry the original colour (no constant fill).
  for x = 0; x < 3; x = x + 1 {
    let (r, g, b, _) = out_reflect.pixel_at(1, x).unwrap()
    assert_eq(r, 200)
    assert_eq(g, 100)
    assert_eq(b, 50)
    let (r, g, b, _) = out_wrap.pixel_at(1, x).unwrap()
    assert_eq(r, 200)
    assert_eq(g, 100)
    assert_eq(b, 50)
  }
}

///|
test "BorderMode modes produce different outputs" {
  let img = Image::from_pixel(3, 3, 255, 255, 255, 255)
  let matrix = [1.0, 0.0, 2.0, 0.0, 1.0, 0.0]
  let const_out = affine_transform(
    img,
    matrix,
    3,
    3,
    Nearest,
    mode=Constant(0, 0, 0, 255),
  )
  let repl_out = affine_transform(img, matrix, 3, 3, Nearest, mode=Replicate)
  // Constant fills with black, Replicate fills with white → differ at col 0.
  let (cr, _, _, _) = const_out.pixel_at(0, 0).unwrap()
  let (rr, _, _, _) = repl_out.pixel_at(0, 0).unwrap()
  assert_true(cr != rr)
}
