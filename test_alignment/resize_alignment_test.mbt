// Resize alignment tests — nearest and bilinear interpolation.
// TODO: 这些测试使用 numpy 拟合 millow 的实现方式生成测试数据

///|
test "align resize Nearest 4x4 -> 8x8" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_eq(resize(img, 8, 8, Nearest), exp_resize_nearest_8x8)
}

///|
test "align resize Bilinear 4x4 -> 2x2" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_close(resize(img, 2, 2, Bilinear), exp_resize_bilinear_2x2, 2)
}