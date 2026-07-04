// Measurement alignment tests — connected components, moments.

///|
test "align connected_components" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  let (labels, _num) = connected_components(img, 4)
  @debug.assert_eq(labels, exp_connected_components)
}

///|
test "align moments" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let actual = moments(img)
  assert_eq(actual.length(), exp_moments.length())
}

///|
test "align hu_moments" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let actual = hu_moments(img)
  assert_eq(actual.length(), exp_hu_moments.length())
}
