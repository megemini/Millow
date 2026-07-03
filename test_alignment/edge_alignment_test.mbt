// Edge detector alignment tests. The gradient magnitude uses sqrt, so
// tolerance 1 for floating-point rounding. sobel_x/sobel_y return float
// arrays compared with a small tolerance.
// TODO: 这些测试使用 numpy 拟合 millow 的实现方式生成测试数据

///|
test "align sobel magnitude" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  assert_img_close(sobel(img), exp_sobel, 2)
}

///|
test "align prewitt magnitude" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  assert_img_close(prewitt(img), exp_prewitt, 2)
}

///|
test "align scharr magnitude" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  assert_img_close(scharr(img), exp_scharr, 2)
}

///|
test "align laplacian" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  assert_img_close(laplacian(img), exp_laplacian, 2)
}

///|
test "align sobel_x gradient array" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  assert_arr2_close(sobel_x(img), exp_sobel_x, 1.0)
}

///|
test "align sobel_y gradient array" {
  let img = img_from_ints(3, 3, in_dot_3x3)
  assert_arr2_close(sobel_y(img), exp_sobel_y, 1.0)
}
