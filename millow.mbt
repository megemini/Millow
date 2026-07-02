// `millow` — a zero-FFI, cross-platform image-processing library for MoonBit.
//
// This root package is a thin facade: the implementation lives in the `src`
// package (`megemini/millow/src`) and is re-exported here so that users can
// simply `import "megemini/millow"` and reach the whole public API through the
// `@millow` alias. See `README.mbt.md` for an overview and examples.

///|
// Core types
pub using @src {
  type Image,
  type Interp,
  type Kernel,
  type PadMode,
  type ImageError,
  type PpmEncoder,

  // I/O traits
  trait Decoder,
  trait Encoder,

  // Channels & color
  to_grayscale,
  to_grayscale_weighted,
  to_bgr,
  invert,
  tint,
  flatten_alpha,
  composite_over,
  merge_channels,

  // Geometry
  crop,
  flip_horizontal,
  flip_vertical,
  rotate_90,
  rotate_180,
  rotate_270,
  resize,
  rescale,
  resize_to_fit,
  resize_to_cover,
  thumbnail,
  pad,
  pad_to_size,

  // Enhancement & tone
  adjust_brightness,
  adjust_contrast,
  adjust_gamma,
  normalize,
  auto_contrast,
  sharpen,
  unsharp_mask,

  // Threshold & histogram
  threshold,
  threshold_inv,
  threshold_otsu,
  histogram,
  histogram_color,
  equalize_histogram,

  // Filters
  convolve,
  box_blur,
  gaussian_blur,
  gaussian_blur_kernel,
  median_filter,
  min_filter,
  max_filter,

  // Edges
  sobel,
  sobel_x,
  sobel_y,
  scharr,
  prewitt,
  laplacian,

  // Morphology
  erode,
  dilate,
  morph_open,
  morph_close,
  morph_gradient,
  top_hat,
  black_hat,

  // Drawing
  draw_pixel,
  draw_line,
  draw_rect,
  draw_circle,

  // Metrics & inspection
  info,
  equals,
  mse,
  psnr,

  // Serialization & I/O registry
  to_ppm,
  to_pgm,
  to_display_bytes,
  encode,
  decode,
  register_encoder,
  register_decoder,

  // Numeric helpers
  clampi,
  clampd,
  clamp_byte,
  round_byte,
}
