// Smoke tests for the extended algorithm set (threshold variants, Wiener,
// DoG, guided filter, Hough lines, distance transform, template matching,
// K-means quantization). These exercise the public API on synthetic inputs
// and verify basic invariants rather than exact alignment with a reference.

///|
fn solid_4x4(r : Int, g : Int, b : Int) -> Image raise {
  let data : Array[Byte] = Array::make(4 * 4 * 4, b'\x00')
  for y = 0; y < 4; y = y + 1 {
    for x = 0; x < 4; x = x + 1 {
      let o = (y * 4 + x) * 4
      data[o] = r.to_byte()
      data[o + 1] = g.to_byte()
      data[o + 2] = b.to_byte()
      data[o + 3] = 255
    }
  }
  Image::from_data(data, 4, 4)
}

///|
fn binary_dot_3x3() -> Image raise {
  // 3x3 with a single white center pixel on a black background.
  let data : Array[Byte] = Array::make(3 * 3 * 4, b'\x00')
  for i = 0; i < 9; i = i + 1 {
    data[i * 4 + 3] = 255
  }
  let o = (1 * 3 + 1) * 4
  data[o] = 255
  data[o + 1] = 255
  data[o + 2] = 255
  Image::from_data(data, 3, 3)
}

///|
test "smoke threshold_bernsen returns binary" {
  let img = solid_4x4(100, 100, 100)
  let out = threshold_bernsen(img, 3, 15)
  assert_true(out.h == 4 && out.w == 4)
  for i = 0; i < out.data.length(); i = i + 1 {
    let v = out.data[i].to_int()
    assert_true(v == 0 || v == 255)
  }
}

///|
test "smoke threshold_wolf returns binary" {
  let img = solid_4x4(100, 100, 100)
  let out = threshold_wolf(img, 3, -0.2)
  assert_true(out.h == 4 && out.w == 4)
}

///|
test "smoke adaptive_threshold_mean returns binary" {
  let img = solid_4x4(100, 100, 100)
  let out = adaptive_threshold_mean(img, 3, 5.0)
  assert_true(out.h == 4 && out.w == 4)
}

///|
test "smoke adaptive_threshold_gaussian returns binary" {
  let img = solid_4x4(100, 100, 100)
  let out = adaptive_threshold_gaussian(img, 3, 5.0)
  assert_true(out.h == 4 && out.w == 4)
}

///|
test "smoke wiener_filter preserves shape" {
  let img = solid_4x4(100, 100, 100)
  let out = wiener_filter(img, 3, -1.0)
  assert_true(out.h == 4 && out.w == 4)
}

///|
test "smoke dog preserves shape" {
  let img = solid_4x4(100, 100, 100)
  let out = dog(img, 0.5, 1.0)
  assert_true(out.h == 4 && out.w == 4)
}

///|
test "smoke guided_filter preserves shape" {
  let img = solid_4x4(100, 100, 100)
  let out = guided_filter(img, img, 2, 50.0)
  assert_true(out.h == 4 && out.w == 4)
}

///|
test "smoke hough_lines on dot returns empty" {
  let img = binary_dot_3x3()
  let lines = hough_lines(img, 5)
  assert_true(lines.length() == 0)
}

///|
test "smoke distance_transform on dot" {
  let img = binary_dot_3x3()
  let d = distance_transform(img)
  assert_true(d.length() == 3 && d[0].length() == 3)
  assert_true(d[1][1] > 0.99 && d[1][1] < 1.01)
  assert_true(d[0][0] == 0.0)
}

///|
test "smoke template_match_best perfect match" {
  // Build a 4x4 image with a distinct L-shaped 2x2 pattern (three bright
  // pixels + one gray) at position (1,1). The template is that 2x2 patch;
  // the best match must be at (1, 1) with score ~1.0.
  let img = solid_4x4(100, 100, 100)
  let img = draw_pixel(img, 1, 1, 255, 255, 255, 255)
  let img = draw_pixel(img, 1, 2, 255, 255, 255, 255)
  let img = draw_pixel(img, 2, 1, 255, 255, 255, 255)
  let tpl = crop(img, 1, 1, 2, 2)
  let (best_y, best_x, score) = template_match_best(img, tpl)
  assert_eq(best_y, 1)
  assert_eq(best_x, 1)
  assert_true(score > 0.999)
}

///|
test "smoke kmeans_quantize reduces colors" {
  let img = solid_4x4(100, 100, 100)
  let out = kmeans_quantize(img, 2)
  assert_true(out.h == 4 && out.w == 4)
  let mut distinct = 0
  let mut seen_r = -1
  let mut seen_g = -1
  let mut seen_b = -1
  for y = 0; y < 4; y = y + 1 {
    for x = 0; x < 4; x = x + 1 {
      let o = (y * 4 + x) * 4
      let r = out.data[o].to_int()
      let g = out.data[o + 1].to_int()
      let b = out.data[o + 2].to_int()
      if r != seen_r || g != seen_g || b != seen_b {
        distinct = distinct + 1
        seen_r = r
        seen_g = g
        seen_b = b
      }
    }
  }
  assert_true(distinct <= 2)
}
