// Feature detection alignment tests — corner detection, HOG, LBP.
// corner_harris/corner_shi_tomasi: Reference: skimage.feature
// hog/lbp: custom implementation (differs from skimage)

///|
test "align corner_harris" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_arr2_close_double(
    corner_harris(img, 2, 3, 0.04),
    exp_corner_harris,
    1.0,
  )
}

///|
test "align corner_shi_tomasi" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_arr2_close_double(
    corner_shi_tomasi(img, 2, 3),
    exp_corner_shi_tomasi,
    1.0,
  )
}

///|
test "align hog" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let actual = hog(img, 2, 1, 9)
  assert_eq(actual.length(), exp_hog.length())
}

///|
test "align lbp" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let actual = lbp(img, 1, 8)
  assert_eq(actual.length(), exp_lbp.length())
}

///|
test "align lbp_histogram" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let actual = lbp_histogram(img, 1, 8)
  assert_arr_eq(actual, exp_lbp_hist)
}
