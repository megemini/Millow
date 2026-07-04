// Feature detection alignment tests — corner detection, HOG, LBP.
// All references below are numpy implementations matching millow's behavior
// (millow uses Constant(0) border for gradients and Gaussian smoothing, which
// differs from skimage's Gaussian-weighted structure tensors).
// hog/lbp: custom implementation (differs from skimage).

///|
test "align corner_harris" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_arr2_close_double(corner_harris(img, 0.04), exp_corner_harris, 1.0)
}

///|
test "align corner_shi_tomasi" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_arr2_close_double(corner_shi_tomasi(img), exp_corner_shi_tomasi, 1.0)
}

///|
test "align hog" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let actual = hog(img, 2, 1, 9)
  assert_arr_close_double(actual, exp_hog, 1.0e-6)
}

///|
test "align lbp" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let actual = lbp(img, 1, 8)
  assert_arr_eq(actual, exp_lbp)
}

///|
test "align lbp_histogram" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let actual = lbp_histogram(img, 1, 8)
  assert_arr_eq(actual, exp_lbp_hist)
}
