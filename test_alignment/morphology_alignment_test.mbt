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

///|
test "align remove_small_objects" {
  let img = img_from_ints(5, 5, in_dots_5x5)
  let result = remove_small_objects(img, 4)
  assert_img_eq(result, exp_remove_small_objects_4)
}

///|
test "align remove_small_holes" {
  let img = img_from_ints(5, 5, in_holed_5x5)
  let result = remove_small_holes(img, 1)
  assert_img_eq(result, exp_remove_small_holes_1)
}

///|
test "align remove_lines Horizontal" {
  let img = img_from_ints(6, 6, in_lines_6x6)
  let result = remove_lines(img, Horizontal, 1)
  assert_img_eq(result, exp_remove_lines_h)
}

///|
test "align remove_lines Both" {
  let img = img_from_ints(6, 6, in_lines_6x6)
  let result = remove_lines(img, Both, 1)
  assert_img_eq(result, exp_remove_lines_both)
}
