// Projection alignment tests — luma sums and foreground counts, exact match.

///|
test "align project_horizontal" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let actual = project_horizontal(img)
  assert_arr_eq(actual, exp_proj_h_grad)
}

///|
test "align project_vertical" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let actual = project_vertical(img)
  assert_arr_eq(actual, exp_proj_v_grad)
}

///|
test "align project_horizontal_binary" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  let actual = project_horizontal_binary(img)
  assert_arr_eq(actual, exp_proj_h_binary_dot)
}

///|
test "align project_vertical_binary" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  let actual = project_vertical_binary(img)
  assert_arr_eq(actual, exp_proj_v_binary_dot)
}
