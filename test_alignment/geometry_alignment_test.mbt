// Geometry alignment tests — crop, flips, rotations, padding. These are
// pure pixel-copy operations, so exact byte match is expected.

///|
test "align crop" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(crop(img, 0, 1, 2, 2), exp_crop)
}

///|
test "align flip_horizontal" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(flip_horizontal(img), exp_flip_h)
}

///|
test "align flip_vertical" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(flip_vertical(img), exp_flip_v)
}

///|
test "align rotate_90" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(rotate_90(img), exp_rotate_90)
}

///|
test "align rotate_180" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(rotate_180(img), exp_rotate_180)
}

///|
test "align rotate_270" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(rotate_270(img), exp_rotate_270)
}

///|
test "align pad Constant" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(pad(img, 1, 1, 1, 1, Constant(7, 8, 9, 10)), exp_pad_const)
}

///|
test "align pad Replicate" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(pad(img, 1, 1, 1, 1, Replicate), exp_pad_replicate)
}

///|
test "align pad Reflect (= numpy symmetric)" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(pad(img, 1, 1, 1, 1, Reflect), exp_pad_reflect)
}

///|
test "align pad Wrap" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(pad(img, 1, 1, 1, 1, Wrap), exp_pad_wrap)
}
