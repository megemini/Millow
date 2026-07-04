// Affine transform alignment tests — rotation, translation.
// Reference: numpy implementation matching millow's rotate_any / translate.

///|
test "align rotate_any 45 degrees" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_close(rotate_any(img, 45.0, Bilinear), exp_rotate_any_45, 1)
}

///|
test "align translate (1, 1)" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_close(translate(img, 1.0, 1.0, Bilinear), exp_translate_1_1, 1)
}
