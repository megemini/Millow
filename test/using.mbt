// Re-export the `millow` public API from this test package so the blackbox
// suite below can call it unqualified, reading like ordinary library usage.

///|
// Core types (their constructors come along, e.g. `Nearest`, `Cross`, `Wrap`)
pub using @millow {
  type Image,
  type Interp,
  type Kernel,
  type PadMode,
  type PpmEncoder,
  type Augmentation,
  type BorderMode,

  // Channels & color
  to_grayscale,
  to_bgr,
  invert,
  flatten_alpha,
  merge_channels,

  // Geometry
  crop,
  flip_horizontal,
  flip_vertical,
  rotate_90,
  rotate_180,
  rotate_270,
  rotate_any,
  translate,
  affine_transform,
  resize,
  rescale,
  resize_to_fit,
  resize_to_cover,
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
  median_filter,
  min_filter,
  max_filter,
  bilateral_filter,

  // Edges
  sobel,
  sobel_x,
  sobel_y,
  scharr,
  prewitt,
  laplacian,
  canny,

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

  // Feature detection
  lbp,
  hog,
  corner_harris,

  // Measurement
  connected_components,
  moments,
  hu_moments,

  // Augmentation
  augment_pipeline,
  augment_random_choice,
  random_noise_gaussian,
  random_noise_salt_pepper,
  process_batch,

  // Thresholding
  threshold_sauvola,

  // Metrics & inspection
  info,
  equals,
  mse,
  psnr,
  ssim,

  // Serialization & I/O registry
  to_ppm,
  to_pgm,
  to_display_bytes,
  encode,
  register_encoder,
}
