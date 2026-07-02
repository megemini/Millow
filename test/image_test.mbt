///|
/// True if calling `f` raises an error.
fn raises(f : () -> Unit raise) -> Bool {
  try {
    f()
    false
  } catch {
    _ => true
  }
}

///|
/// Build a small 2x3 image whose red channel encodes `y * w + x`.
fn sample_image() -> Image raise {
  let img = Image::new(2, 3)
  for y = 0; y < 2; y = y + 1 {
    for x = 0; x < 3; x = x + 1 {
      img.pixel_set(y, x, (y * 3 + x).to_byte(), 100, 200, 255)
    }
  }
  img
}

///|
test "image new, shape and accessors" {
  let img = Image::new(3, 4)
  assert_eq(img.height(), 3)
  assert_eq(img.width(), 4)
  @debug.assert_eq(img.shape(), (3, 4))
  assert_eq(img.len(), 3 * 4 * 4)
  assert_false(img.is_empty())
  assert_true(Image::new(0, 5).is_empty())
}

///|
test "from_pixel and pixel access" {
  let img = Image::from_pixel(2, 2, 10, 20, 30, 40)
  @debug.assert_eq(img.pixel_at(0, 0), Some((10, 20, 30, 40)))
  @debug.assert_eq(img.pixel_at(-1, 0), None)
  @debug.assert_eq(img.pixel_at(2, 2), None)
  img.pixel_set(1, 1, 1, 2, 3, 4)
  @debug.assert_eq(img.pixel_at(1, 1), Some((1, 2, 3, 4)))
}

///|
test "from_data validates length" {
  let ok = Image::from_data(Array::make(2 * 2 * 4, (7 : Byte)), 2, 2)
  assert_eq(ok.pixel_at(0, 0).unwrap().0, 7)
  assert_true(
    raises(() => Image::from_data(Array::make(3, (0 : Byte)), 2, 2) |> ignore),
  )
}

///|
test "clone is independent" {
  let a = sample_image()
  let b = a.clone()
  b.pixel_set(0, 0, 9, 9, 9, 9)
  assert_eq(a.pixel_at(0, 0).unwrap().0, 0)
  assert_eq(b.pixel_at(0, 0).unwrap().0, 9)
}

///|
test "channel split, set and merge" {
  let img = sample_image()
  let (r, g, b, a) = img.split_channels()
  assert_eq(r.length(), 6)
  assert_eq(r[4], 4)
  assert_eq(g[0], 100)
  let img2 = img.set_channel(0, Array::make(6, (5 : Byte)))
  assert_eq(img2.channel_r()[3], 5)
  let merged = merge_channels(r, g, b, Some(a))
  assert_eq(merged.width(), 6)
  assert_eq(merged.height(), 1)
  assert_true(raises(() => merge_channels(r, [], b, None) |> ignore))
}

///|
test "grayscale, invert and bgr" {
  let img = Image::from_pixel(1, 1, 10, 20, 30, 128)
  let (gr, gg, gb, ga) = to_grayscale(img).pixel_at(0, 0).unwrap()
  assert_eq(gr, gg)
  assert_eq(gg, gb)
  assert_eq(ga, 128)
  assert_eq(invert(img).pixel_at(0, 0).unwrap().0, 245)
  @debug.assert_eq(to_bgr(img).pixel_at(0, 0).unwrap(), (30, 20, 10, 128))
}

///|
test "flatten_alpha over black" {
  let img = Image::from_pixel(1, 1, 200, 200, 200, 0)
  let flat = flatten_alpha(img, 0, 0, 0)
  @debug.assert_eq(flat.pixel_at(0, 0).unwrap(), (0, 0, 0, 255))
}

///|
test "flips and rotations" {
  let img = sample_image()
  assert_eq(flip_horizontal(img).pixel_at(0, 0).unwrap().0, 2)
  assert_eq(flip_vertical(img).pixel_at(0, 0).unwrap().0, 3)
  @debug.assert_eq(rotate_90(img).shape(), (3, 2))
  assert_eq(rotate_180(img).pixel_at(0, 0).unwrap().0, 5)
  @debug.assert_eq(rotate_270(img).shape(), (3, 2))
}

///|
test "crop bounds" {
  let img = sample_image()
  let c = crop(img, 0, 1, 2, 2)
  @debug.assert_eq(c.shape(), (2, 2))
  assert_eq(c.pixel_at(0, 0).unwrap().0, 1)
  assert_true(raises(() => crop(img, 0, 0, 5, 5) |> ignore))
}
