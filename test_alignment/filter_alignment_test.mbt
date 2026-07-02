// Filter alignment tests. Box blur uses integer integral images (exact);
// gaussian and convolve use floating point (tolerance 1). Median/min/max
// are integer comparisons (exact).

///|
test "align box_blur radius=1" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(box_blur(img, 1), exp_box_blur_1)
}

///|
test "align gaussian_blur sigma=1.0" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_close(gaussian_blur(img, 1.0), exp_gaussian_blur_1, 1)
}

///|
test "align convolve identity kernel" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let k = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
  assert_img_eq(convolve(img, k, false), exp_convolve_identity)
}

///|
test "align median_filter size=3" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(median_filter(img, 3), exp_median_3)
}

///|
test "align min_filter size=3" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(min_filter(img, 3), exp_min_filter_3)
}

///|
test "align max_filter size=3" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(max_filter(img, 3), exp_max_filter_3)
}
