// Histogram alignment tests — histogram computation, equalization.
// TODO: 这些测试使用 numpy 拟合 millow 的实现方式生成测试数据

///|
test "align histogram" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_arr_eq(histogram(img), exp_histogram)
}

///|
test "align histogram_color" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  let actual = histogram_color(img)
  assert_arr_eq(actual[0], exp_hist_color_r)
  assert_arr_eq(actual[1], exp_hist_color_g)
  assert_arr_eq(actual[2], exp_hist_color_b)
}

///|
test "align equalize_histogram" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_close(equalize_histogram(img), exp_equalize, 2)
}
