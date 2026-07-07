///|
/// Comprehensive demonstration of the `millow` image-processing library.
/// Reads an image, applies various operations, and saves results as JPEG.
/// Run with `moon run .` from the examples/ directory.
fn main {
  println("=== millow Demo ===")
  println(" ")

  // Read and decode the input image
  let input_path = "lena_ai_generated.png"
  let img_bytes = @fs.read_file_to_bytes(input_path) catch {
    _ => {
      println("Error: cannot read \{input_path}")
      return
    }
  }
  let img_data = decode_image(img_bytes) catch {
    _ => {
      println("Error: cannot decode image")
      return
    }
  }
  let full = image_data_to_millow(img_data)
  // Resize to 640px wide for smaller output files
  let ratio = 640.0 / full.w.to_double()
  let dst_h = (full.h.to_double() * ratio).to_int()
  let canvas = @millow.resize(full, dst_h, 640, @millow.Interp::Bilinear)
  println(
    "Loaded image: \{@millow.info(full)} -> resized to \{@millow.info(canvas)}",
  )
  println(" ")

  // Create output directory
  @fs.create_dir("demo_output") catch {
    _ => ()
  }

  // === Color operations ===
  let gray = @millow.to_grayscale(canvas)
  save_jpg(gray, "demo_output/01_grayscale.jpg")
  println("1. to_grayscale - saved")

  let inverted = @millow.invert(canvas)
  save_jpg(inverted, "demo_output/02_inverted.jpg")
  println("2. invert - saved")

  let tinted = @millow.tint(canvas, 100, 150, 200)
  save_jpg(tinted, "demo_output/03_tinted.jpg")
  println("3. tint(100,150,200) - saved")
  println(" ")

  // === Enhancement operations ===
  let bright = @millow.adjust_brightness(canvas, 30)
  save_jpg(bright, "demo_output/04_brightness.jpg")
  println("4. adjust_brightness(+30) - saved")

  let contrast = @millow.adjust_contrast(canvas, 1.5)
  save_jpg(contrast, "demo_output/05_contrast.jpg")
  println("5. adjust_contrast(1.5) - saved")

  let gamma = @millow.adjust_gamma(canvas, 1.5)
  save_jpg(gamma, "demo_output/06_gamma.jpg")
  println("6. adjust_gamma(1.5) - saved")

  let sharpened = @millow.sharpen(canvas, 1.0)
  save_jpg(sharpened, "demo_output/07_sharpen.jpg")
  println("7. sharpen - saved")
  println(" ")

  // === Filter operations ===
  let blur = @millow.gaussian_blur(canvas, 2.0)
  save_jpg(blur, "demo_output/08_gaussian_blur.jpg")
  println("8. gaussian_blur(sigma=2.0) - saved")

  let median = @millow.median_filter(canvas, 3)
  save_jpg(median, "demo_output/09_median_filter.jpg")
  println("9. median_filter(kernel=3) - saved")

  let bilateral = @millow.bilateral_filter(gray, 5, 10.0, 10.0)
  save_jpg(bilateral, "demo_output/10_bilateral_filter.jpg")
  println("10. bilateral_filter - saved")
  println(" ")

  // === Edge detection ===
  let sobel = @millow.sobel(gray)
  save_jpg(sobel, "demo_output/11_sobel.jpg")
  println("11. sobel - saved")

  let scharr = @millow.scharr(gray)
  save_jpg(scharr, "demo_output/12_scharr.jpg")
  println("12. scharr - saved")

  let laplacian = @millow.laplacian(gray)
  save_jpg(laplacian, "demo_output/13_laplacian.jpg")
  println("13. laplacian - saved")
  println(" ")

  // === Morphology ===
  let eroded = @millow.erode(gray, @millow.Kernel::Square(3))
  save_jpg(eroded, "demo_output/14_erode.jpg")
  println("14. erode - saved")

  let dilated = @millow.dilate(gray, @millow.Kernel::Square(3))
  save_jpg(dilated, "demo_output/15_dilate.jpg")
  println("15. dilate - saved")

  let morph_open = @millow.morph_open(gray, @millow.Kernel::Square(3))
  save_jpg(morph_open, "demo_output/16_morph_open.jpg")
  println("16. morph_open - saved")

  let morph_close = @millow.morph_close(gray, @millow.Kernel::Square(3))
  save_jpg(morph_close, "demo_output/17_morph_close.jpg")
  println("17. morph_close - saved")
  println(" ")

  // === Geometry ===
  let rotated = @millow.rotate_any(canvas, 45.0, @millow.Interp::Bilinear)
  save_jpg(rotated, "demo_output/18_rotate_45.jpg")
  println("18. rotate_any(45) - saved")

  let flipped_h = @millow.flip_horizontal(canvas)
  save_jpg(flipped_h, "demo_output/19_flip_horizontal.jpg")
  println("19. flip_horizontal - saved")

  let flipped_v = @millow.flip_vertical(canvas)
  save_jpg(flipped_v, "demo_output/20_flip_vertical.jpg")
  println("20. flip_vertical - saved")

  let (oh, ow) = canvas.shape()
  let resized = @millow.resize(canvas, oh / 2, ow / 2, @millow.Interp::Bilinear)
  save_jpg(resized, "demo_output/21_resize_half.jpg")
  println("21. resize(50%) - saved")

  let translated = @millow.translate(
    canvas,
    20.0,
    20.0,
    @millow.Interp::Bilinear,
  )
  save_jpg(translated, "demo_output/22_translate.jpg")
  println("22. translate(20,20) - saved")
  println(" ")

  // === Threshold & Histogram ===
  let (otsu_thr, otsu_binary) = @millow.threshold_otsu(gray)
  save_jpg(otsu_binary, "demo_output/23_threshold_otsu.jpg")
  println("23. threshold_otsu(thr=\{otsu_thr.to_int()}) - saved")

  let equalized = @millow.equalize_histogram(gray)
  save_jpg(equalized, "demo_output/24_equalize_histogram.jpg")
  println("24. equalize_histogram - saved")
  println(" ")

  // === Feature detection ===
  let corners = @millow.corner_harris(gray, 0.04)
  let corners_vis = visualize_corners(canvas, corners)
  save_jpg(corners_vis, "demo_output/25_corner_harris.jpg")
  println("25. corner_harris - saved")

  let lbp_codes = @millow.lbp(gray, 1, 8)
  let lbp_img = lbp_to_image(lbp_codes, gray.h, gray.w)
  save_jpg(lbp_img, "demo_output/26_lbp.jpg")
  println("26. lbp(radius=1, n=8) - saved")
  println(" ")

  // === Measurement ===
  let (_labels, count) = @millow.connected_components(otsu_binary, 4)
  println("27. connected_components: \{count} components")

  let contours = @millow.find_contours(otsu_binary)
  let contour_img = draw_contours(canvas, contours)
  save_jpg(contour_img, "demo_output/28_contours.jpg")
  println("28. find_contours(\{contours.length()} found) - saved")
  println(" ")

  // === Data augmentation ===
  // Compose multiple augmentations into a single pipeline.
  let augmented = @millow.augment_pipeline(canvas, [
    @millow.Augmentation::FlipHorizontal,
    @millow.Augmentation::Rotate(15.0),
    @millow.Augmentation::Brightness(1.2),
    @millow.Augmentation::Contrast(1.3),
    @millow.Augmentation::NoiseGaussian(8.0),
  ]) catch {
    _ => {
      println("Error: augment_pipeline failed")
      return
    }
  }
  save_jpg(augmented, "demo_output/29_augment_pipeline.jpg")
  println("29. augment_pipeline (5 ops) - saved")

  // Randomly pick one augmentation from a weighted list.
  let choice = @millow.augment_random_choice(canvas, [
    (0.4, @millow.Augmentation::FlipVertical),
    (0.4, @millow.Augmentation::Gamma(0.8)),
    (0.2, @millow.Augmentation::ColorJitter(0.2, 0.2, 0.0, 0.0)),
  ]) catch {
    _ => {
      println("Error: augment_random_choice failed")
      return
    }
  }
  save_jpg(choice, "demo_output/30_augment_random_choice.jpg")
  println("30. augment_random_choice - saved")
  println(" ")

  // === Advanced filters ===
  let guided = @millow.guided_filter_self(gray, 5, 0.01)
  save_jpg(guided, "demo_output/31_guided_filter.jpg")
  println("31. guided_filter_self(radius=5, eps=0.01) - saved")

  let wiener = @millow.wiener_filter(gray, 5, -1.0)
  save_jpg(wiener, "demo_output/32_wiener_filter.jpg")
  println("32. wiener_filter(window=5, noise_var=auto) - saved")

  let dog_img = @millow.dog(gray, 1.0, 2.0)
  save_jpg(dog_img, "demo_output/33_dog.jpg")
  println("33. dog(sigma1=1.0, sigma2=2.0) - saved")
  println(" ")

  // === Local thresholding ===
  let bernsen = @millow.threshold_bernsen(gray, 15, 15)
  save_jpg(bernsen, "demo_output/34_threshold_bernsen.jpg")
  println("34. threshold_bernsen(window=15, contrast=15) - saved")

  let wolf = @millow.threshold_wolf(gray, 15, 0.5)
  save_jpg(wolf, "demo_output/35_threshold_wolf.jpg")
  println("35. threshold_wolf(window=15, k=0.5) - saved")

  let adap_mean = @millow.adaptive_threshold_mean(gray, 15, 5.0)
  save_jpg(adap_mean, "demo_output/36_adaptive_threshold_mean.jpg")
  println("36. adaptive_threshold_mean(window=15, C=5) - saved")

  let adap_gauss = @millow.adaptive_threshold_gaussian(gray, 15, 5.0)
  save_jpg(adap_gauss, "demo_output/37_adaptive_threshold_gaussian.jpg")
  println("37. adaptive_threshold_gaussian(window=15, C=5) - saved")
  println(" ")

  // === Hough lines & morphology ===
  let edges = @millow.canny(gray, 50.0, 150.0)
  save_jpg(edges, "demo_output/38_canny.jpg")
  println("38. canny(low=50, high=150) - saved")

  let lines = @millow.hough_lines(edges, 80)
  println("39. hough_lines: \{lines.length()} lines detected (threshold=80)")

  let cleaned = @millow.remove_lines(
    otsu_binary,
    @millow.LineDirection::Both,
    2,
  )
  save_jpg(cleaned, "demo_output/40_remove_lines.jpg")
  println("40. remove_lines(Both, max_thickness=2) - saved")
  println(" ")

  // === Perspective & model input ===
  let (ph, pw) = canvas.shape()
  let src_pts = [(0.0, 0.0), (pw.to_double(), 0.0), (pw.to_double(), ph.to_double()), (0.0, ph.to_double())]
  let dst_pts = [(20.0, 20.0), (pw.to_double() - 20.0, 30.0), (pw.to_double() - 30.0, ph.to_double() - 20.0), (30.0, ph.to_double() - 30.0)]
  let warped = @millow.perspective_4pt(canvas, src_pts, dst_pts, @millow.Interp::Bilinear)
  save_jpg(warped, "demo_output/41_perspective_4pt.jpg")
  println("41. perspective_4pt - saved")

  let model_input = @millow.resize_to_model_input(canvas, 224, 224, interp=@millow.Interp::Bilinear)
  save_jpg(model_input, "demo_output/42_resize_to_model_input.jpg")
  println("42. resize_to_model_input(224x224) - saved")

  let letterboxed = @millow.letterbox(canvas, 224, 224, color=(114, 114, 114, 255))
  save_jpg(letterboxed, "demo_output/43_letterbox.jpg")
  println("43. letterbox(224x224, color=114) - saved")
  println(" ")

  // === Quantization & distance transform ===
  let quant = @millow.kmeans_quantize(canvas, 8, seed=42)
  save_jpg(quant, "demo_output/44_kmeans_quantize.jpg")
  println("44. kmeans_quantize(k=8, seed=42) - saved")

  let dt = @millow.distance_transform(otsu_binary)
  let dt_img = doubles_to_image(dt, otsu_binary.h, otsu_binary.w)
  save_jpg(dt_img, "demo_output/45_distance_transform.jpg")
  println("45. distance_transform - saved")
  println(" ")

  // === Metrics ===
  println("Quality metrics:")
  println("  PSNR(original, blurred) = \{@millow.psnr(canvas, blur)} dB")
  println("  MSE(original, blurred)  = \{@millow.mse(canvas, blur)}")
  println("  SSIM(original, blurred) = \{@millow.ssim(canvas, blur)}")
  println(" ")

  println("=== Demo complete! ===")
  println("Generated 45 output images (JPEG) in demo_output/ directory.")
}

///|
/// Decode an image from bytes, auto-detecting PNG or JPEG format.
fn decode_image(data : Bytes) -> @image.ImageData raise {
  if data.length() >= 4 {
    let b0 = data[0].to_int()
    let b1 = data[1].to_int()
    let b2 = data[2].to_int()
    let b3 = data[3].to_int()
    // PNG: 89 50 4E 47
    if b0 == 0x89 && b1 == 0x50 && b2 == 0x4E && b3 == 0x47 {
      return @image.decode_png(data)
    }
    // JPEG: FF D8
    if b0 == 0xFF && b1 == 0xD8 {
      return @image.decode_jpeg(data)
    }
  }
  // Default: try PNG, then JPEG
  @image.decode_png(data)
}

///|
/// Convert mizchi/image ImageData to millow Image.
fn image_data_to_millow(img_data : @image.ImageData) -> @millow.Image {
  let h = img_data.height
  let w = img_data.width
  let data = Array::make(h * w * 4, (0 : Byte))
  let src = img_data.data
  for i = 0; i < data.length(); i = i + 1 {
    data[i] = src[i]
  }
  { data, h, w }
}

///|
/// Convert millow Image to mizchi/image ImageData.
fn millow_to_image_data(img : @millow.Image) -> @image.ImageData {
  { width: img.w, height: img.h, data: Bytes::from_array(img.data) }
}

///|
/// Save a millow Image as a JPEG file (quality = 90).
fn save_jpg(img : @millow.Image, filename : String) -> Unit {
  let img_data = millow_to_image_data(img)
  let jpg_bytes = @image.encode_jpeg(img_data, quality=90) catch {
    _ => {
      println("Error: cannot encode JPEG for \{filename}")
      return
    }
  }
  @fs.write_bytes_to_file(filename, jpg_bytes) catch {
    _ => println("Error: cannot write \{filename}")
  }
}

///|
fn visualize_corners(
  img : @millow.Image,
  corners : Array[Array[Double]],
) -> @millow.Image {
  let mut result = img
  let threshold = 0.1
  for y = 0; y < corners.length(); y = y + 1 {
    for x = 0; x < corners[y].length(); x = x + 1 {
      if corners[y][x] > threshold {
        result = @millow.draw_circle(result, y, x, 3, 255, 0, 0, 255, true, 1)
      }
    }
  }
  result
}

///|
fn draw_contours(
  img : @millow.Image,
  contours : Array[Array[(Int, Int)]],
) -> @millow.Image {
  let mut result = img
  let colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
  ]
  for i = 0; i < contours.length(); i = i + 1 {
    let (r, g, b) = colors[i % colors.length()]
    for j = 0; j < contours[i].length(); j = j + 1 {
      let (y, x) = contours[i][j]
      result = @millow.draw_pixel(
        result,
        y,
        x,
        r.to_byte(),
        g.to_byte(),
        b.to_byte(),
        255,
      )
    }
  }
  result
}

///|
fn lbp_to_image(codes : Array[Int], h : Int, w : Int) -> @millow.Image {
  let img = @millow.Image::new(h, w)
  for i = 0; i < codes.length(); i = i + 1 {
    let o = i * 4
    let val = codes[i].min(255).to_byte()
    img.data[o] = val
    img.data[o + 1] = val
    img.data[o + 2] = val
    img.data[o + 3] = 255
  }
  img
}

///|
/// Render a 2D `Array[Array[Double]]` (e.g. distance transform output) as a
/// grayscale image by scaling the max value to 255.
fn doubles_to_image(arr : Array[Array[Double]], h : Int, w : Int) -> @millow.Image {
  let img = @millow.Image::new(h, w)
  let mut max_v = 0.0
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      let v = arr[y][x]
      if v > max_v {
        max_v = v
      }
    }
  }
  let scale = if max_v > 0.0 { 255.0 / max_v } else { 0.0 }
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      let o = (y * w + x) * 4
      let val = (arr[y][x] * scale).to_int().min(255).max(0).to_byte()
      img.data[o] = val
      img.data[o + 1] = val
      img.data[o + 2] = val
      img.data[o + 3] = 255
    }
  }
  img
}
