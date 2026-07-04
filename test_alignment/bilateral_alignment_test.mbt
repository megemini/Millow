// Bilateral filter alignment tests — edge-preserving smoothing.
// Reference: numpy implementation matching millow's behavior — millow uses
// Reflect border and RGB normalized to [0,1] for color distance (differs from
// skimage's denoise_bilateral in border handling and sigma_color semantics).

///|
test "align bilateral_filter" {
  let img = img_from_ints(4, 4, in_grad_4x4)
  assert_img_close(
    bilateral_filter(img, 5, 0.1, 10.0),
    exp_bilateral_5_01_10,
    2,
  )
}
