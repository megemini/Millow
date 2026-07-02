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
