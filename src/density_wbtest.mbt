///|
test "density_scaling basic usage" {
  let img = Image::from_pixel(100, 100, 255, 255, 255, 255)
  for y = 20; y < 80; y = y + 1 {
    for x = 20; x < 80; x = x + 1 {
      img.data[img.offset(y, x)] = 0
      img.data[img.offset(y, x) + 1] = 0
      img.data[img.offset(y, x) + 2] = 0
    }
  }
  let scaled = density_scaling(img, char_count=Some(5))
  assert_true(scaled.h >= 32 && scaled.w >= 32)
}

///|
test "density_scaling estimated char count" {
  let img = Image::from_pixel(100, 100, 255, 255, 255, 255)
  for y = 30; y < 40; y = y + 1 {
    for x = 30; x < 40; x = x + 1 {
      img.data[img.offset(y, x)] = 0
      img.data[img.offset(y, x) + 1] = 0
      img.data[img.offset(y, x) + 2] = 0
    }
  }
  for y = 50; y < 60; y = y + 1 {
    for x = 50; x < 60; x = x + 1 {
      img.data[img.offset(y, x)] = 0
      img.data[img.offset(y, x) + 1] = 0
      img.data[img.offset(y, x) + 2] = 0
    }
  }
  let scaled = density_scaling(img)
  assert_true(scaled.h >= 32 && scaled.w >= 32)
}

///|
test "density_scaling light text on dark background" {
  let img = Image::from_pixel(100, 100, 0, 0, 0, 255)
  for y = 20; y < 80; y = y + 1 {
    for x = 20; x < 80; x = x + 1 {
      img.data[img.offset(y, x)] = 255
      img.data[img.offset(y, x) + 1] = 255
      img.data[img.offset(y, x) + 2] = 255
    }
  }
  let scaled = density_scaling(img, char_count=Some(5), dark_text=false)
  assert_true(scaled.h >= 32 && scaled.w >= 32)
}

///|
test "density_scaling with bbox protection" {
  let img = Image::from_pixel(100, 100, 255, 255, 255, 255)
  for y = 40; y < 60; y = y + 1 {
    for x = 40; x < 60; x = x + 1 {
      img.data[img.offset(y, x)] = 0
      img.data[img.offset(y, x) + 1] = 0
      img.data[img.offset(y, x) + 2] = 0
    }
  }
  let bboxes = [(40, 40, 42, 60)]
  let scaled = density_scaling(img, char_count=Some(1), bboxes=Some(bboxes))
  assert_true(scaled.h >= 32 && scaled.w >= 32)
}

///|
test "density_scaling min_scale clamping" {
  let img = Image::from_pixel(100, 100, 255, 255, 255, 255)
  let scaled = density_scaling(img, char_count=Some(1), min_scale=0.8)
  assert_true(scaled.h >= 32 && scaled.w >= 32)
}

///|
test "density_scaling size rounded to 32" {
  let img = Image::from_pixel(100, 100, 255, 255, 255, 255)
  for y = 20; y < 80; y = y + 1 {
    for x = 20; x < 80; x = x + 1 {
      img.data[img.offset(y, x)] = 0
      img.data[img.offset(y, x) + 1] = 0
      img.data[img.offset(y, x) + 2] = 0
    }
  }
  let scaled = density_scaling(img, char_count=Some(100))
  assert_true(scaled.h % 32 == 0)
  assert_true(scaled.w % 32 == 0)
}

///|
test "density_scaling empty image" {
  let img = Image::new(0, 0)
  let scaled = density_scaling(img)
  assert_true(scaled.h == 0)
  assert_true(scaled.w == 0)
}

///|
test "density_scaling all white image" {
  let img = Image::from_pixel(100, 100, 255, 255, 255, 255)
  let scaled = density_scaling(img)
  assert_true(scaled.h >= 32 && scaled.w >= 32)
}

///|
test "density_scaling all black image" {
  let img = Image::from_pixel(100, 100, 0, 0, 0, 255)
  let scaled = density_scaling(img, char_count=Some(1))
  assert_true(scaled.h >= 32 && scaled.w >= 32)
}

///|
test "density_scaling no bboxes" {
  let img = Image::from_pixel(100, 100, 255, 255, 255, 255)
  for y = 20; y < 80; y = y + 1 {
    for x = 20; x < 80; x = x + 1 {
      img.data[img.offset(y, x)] = 0
      img.data[img.offset(y, x) + 1] = 0
      img.data[img.offset(y, x) + 2] = 0
    }
  }
  let scaled = density_scaling(img, char_count=Some(5), bboxes=None)
  assert_true(scaled.h >= 32 && scaled.w >= 32)
}
