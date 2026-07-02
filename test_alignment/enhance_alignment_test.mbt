// Enhancement alignment tests. Brightness is integer (exact); contrast and
// gamma use floating point, tolerance 1 for rounding edges.

///|
test "align adjust_brightness +50" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(adjust_brightness(img, 50), exp_brightness_50)
}

///|
test "align adjust_contrast x1.5" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_close(adjust_contrast(img, 1.5), exp_contrast_1_5, 1)
}

///|
test "align adjust_gamma 2.0" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_close(adjust_gamma(img, 2.0), exp_gamma_2_0, 1)
}

///|
test "align normalize 0-255" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_close(normalize(img, 0, 255), exp_normalize, 1)
}
