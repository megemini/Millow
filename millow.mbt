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
  type Augmentation,
  type RegionProps,
  type BorderMode,

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
  composite_color,
  merge_channels,
  grayscale_to_rgb,
  grayscale_to_rgba,
  to_rgb,
  to_rgba,
  to_hsv,
  from_hsv,
  to_ycbcr,
  from_ycbcr,

  // Color LUT
  apply_lut,
  apply_lut_uniform,

  // Geometry
  crop,
  flip_horizontal,
  flip_vertical,
  rotate_90,
  rotate_180,
  rotate_270,
  rotate_any,
  affine_transform,
  translate,
  shear,
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
  standardize,
  auto_contrast,
  sharpen,
  unsharp_mask,

  // Threshold & histogram
  threshold,
  threshold_inv,
  threshold_otsu,
  threshold_sauvola,
  histogram,
  histogram_color,
  equalize_histogram,
  clahe,
  match_histogram,
  histogram_correlation,

  // Filters
  convolve,
  box_blur,
  gaussian_blur,
  gaussian_blur_kernel,
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
  skeletonize,
  hit_or_miss,

  // Drawing
  draw_pixel,
  draw_line,
  draw_rect,
  draw_circle,
  draw_ellipse,
  draw_polygon,
  flood_fill,

  // Feature detection
  corner_harris,
  corner_shi_tomasi,
  hog,
  lbp,
  lbp_histogram,

  // Measurement
  connected_components,
  find_contours,
  regionprops,
  count_nonzero,
  count_pixels,
  moments,
  hu_moments,

  // Data augmentation
  random_crop,
  random_flip_horizontal,
  random_rotate,
  random_brightness,
  random_contrast,
  random_gamma,
  random_noise_gaussian,
  random_noise_salt_pepper,
  random_color_jitter,
  augment_pipeline,
  augment_random_choice,
  process_batch,

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
  decode,
  register_encoder,
  register_decoder,

  // Numeric helpers
  clampi,
  clampd,
  clamp_byte,
  round_byte,
}
