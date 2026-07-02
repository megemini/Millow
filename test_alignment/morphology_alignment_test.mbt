// Morphology alignment tests — grayscale min/max over structuring elements,
// integer operations, exact match expected.

///|
test "align erode Cross(3)" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  assert_img_eq(erode(img, Cross(3)), exp_erode_cross)
}

///|
test "align dilate Cross(3)" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  assert_img_eq(dilate(img, Cross(3)), exp_dilate_cross)
}

///|
test "align erode Square(3)" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  assert_img_eq(erode(img, Square(3)), exp_erode_square)
}

///|
test "align dilate Square(3)" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  assert_img_eq(dilate(img, Square(3)), exp_dilate_square)
}
