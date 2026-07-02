// Color operations alignment tests — verify millow matches the Python
// reference (numpy implementing the same BT.601 / integer formulas).

///|
test "align invert" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(invert(img), exp_invert)
}

///|
test "align to_bgr" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(to_bgr(img), exp_to_bgr)
}

///|
test "align to_grayscale (BT.601)" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(to_grayscale(img), exp_to_grayscale)
}

///|
test "align tint" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(tint(img, 200, 100, 50), exp_tint)
}

///|
test "align flatten_alpha" {
  let img = img_from_ints(2, 2, in_alpha_2x2)
  assert_img_eq(flatten_alpha(img, 10, 20, 30), exp_flatten_alpha)
}
