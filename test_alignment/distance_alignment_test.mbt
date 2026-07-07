// Distance transform and template matching alignment tests.

///|
test "align distance_transform on dot" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  let result = distance_transform(img)
  assert_arr2_close_double(result, exp_distance_dot, 0.01)
}

///|
test "align template_match grad/tpl" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let tpl = img_from_ints(2, 2, in_tpl_2x2)
  let result = template_match(img, tpl)
  assert_arr2_close_double(result, exp_template_match_grad, 0.001)
}

///|
test "align template_match_best grad/tpl" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let tpl = img_from_ints(2, 2, in_tpl_2x2)
  // The template is the top-left 2x2 of the image, so the best match should be
  // at (0, 0) with a perfect score of ~1.0.
  let (best_y, best_x, score) = template_match_best(img, tpl)
  assert_eq(best_y, 0)
  assert_eq(best_x, 0)
  assert_true(score > 0.999)
}
