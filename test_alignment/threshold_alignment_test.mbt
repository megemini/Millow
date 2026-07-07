// Threshold alignment tests — integer luma comparison, exact match.

///|
test "align threshold 128" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(threshold(img, 128), exp_threshold_128)
}

///|
test "align threshold_inv 128" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(threshold_inv(img, 128), exp_threshold_inv_128)
}

///|
test "align threshold_otsu on bimodal" {
  let img = img_from_ints(1, 4, in_bimodal_1x4)
  let (thr, out) = threshold_otsu(img)
  assert_eq(thr.to_int(), exp_otsu_threshold)
  assert_img_eq(out, exp_otsu_img)
}

///|
test "align threshold_niblack" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let result = threshold_niblack(img, 3, -0.2)
  assert_img_eq(result, exp_threshold_niblack_3)
}

///|
test "align threshold_bernsen" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let result = threshold_bernsen(img, 3, 15)
  assert_img_eq(result, exp_bernsen_3_15)
}

///|
test "align threshold_wolf" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let result = threshold_wolf(img, 3, -0.2)
  assert_img_eq(result, exp_wolf_3_neg02)
}

///|
test "align adaptive_threshold_mean" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let result = adaptive_threshold_mean(img, 3, 5.0)
  assert_img_eq(result, exp_adapt_mean_3_5)
}

///|
test "align adaptive_threshold_gaussian" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let result = adaptive_threshold_gaussian(img, 3, 5.0)
  assert_img_eq(result, exp_adapt_gauss_3_5)
}
