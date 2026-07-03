// Bilateral filter alignment tests — edge-preserving smoothing.
// TODO: 这些测试使用 numpy 拟合 millow 的实现方式生成测试数据

///|
test "align bilateral_filter" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_close(
    bilateral_filter(img, 5, 0.1, 10.0),
    exp_bilateral_5_01_10,
    2,
  )
}
