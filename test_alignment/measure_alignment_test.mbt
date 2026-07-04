// Measurement alignment tests — connected components, moments, Hu moments.

///|
test "align connected_components" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  let (labels, num) = connected_components(img, 4)
  assert_eq(num, 1)
  assert_arr2_eq(labels, exp_connected_components)
}

///|
test "align moments" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let actual = moments(img)
  assert_arr_close_double(actual, exp_moments, 1.0e-9)
}

///|
test "align hu_moments" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let actual = hu_moments(img)
  assert_arr_close_double(actual, exp_hu_moments, 1.0e-6)
}
