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
  type LineDirection,

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
  auto_crop,
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
  resize_to_model_input,
  letterbox,
  thumbnail,
  pad,
  pad_to_size,
  skew_angle,
  deskew,
  perspective_transform,
  perspective_4pt,

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
  threshold_niblack,
  threshold_bernsen,
  threshold_wolf,
  adaptive_threshold_mean,
  adaptive_threshold_gaussian,
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
  guided_filter,
  guided_filter_self,
  wiener_filter,
  dog,

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
  remove_small_objects,
  remove_small_holes,
  remove_lines,

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
  project_horizontal,
  project_vertical,
  project_horizontal_binary,
  project_vertical_binary,
  integral_image,
  integral_array,
  integral_image_sq,
  integral_array_sq,
  query_integral,
  hough_lines,
  distance_transform,
  template_match,
  template_match_best,
  kmeans_quantize,

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
