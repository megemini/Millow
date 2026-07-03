// Bilateral filter alignment tests — edge-preserving smoothing.
// TODO: 这些测试使用 numpy 拟合 millow 的实现方式生成测试数据

///|
test "align bilateral_filter" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_close(
    bilateral_filter(img, 5, 10.0, 10.0),
    exp_bilateral_5_10_10,
    20,
  )
}