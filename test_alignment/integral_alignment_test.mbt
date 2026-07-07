// Integral image alignment tests — luma summed-area table, exact match.

///|
test "align integral_image" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let actual = integral_image(img)
  assert_arr_close_double(actual, exp_integral_image_grad, 0.001)
}
