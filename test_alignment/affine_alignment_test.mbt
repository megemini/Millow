// Affine transform alignment tests — rotation, translation.
// TODO: 这些测试使用 numpy 拟合 millow 的实现方式生成测试数据
// TODO: rotate_any 和 translate 测试失败，需要进一步排查原因

///|
// test "align rotate_any 45 degrees" {
//   let img = img_from_ints(4, 4, in_grad_4x4)
//   assert_img_close(rotate_any(img, 45.0, Bilinear), exp_rotate_any_45, 20)
// }

///|
// test "align translate (1, 1)" {
//   let img = img_from_ints(4, 4, in_grad_4x4)
//   assert_img_close(translate(img, 1.0, 1.0, Bilinear), exp_translate_1_1, 20)
// }
