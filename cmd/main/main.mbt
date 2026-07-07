///|
/// Lightweight millow demo that depends only on the millow library itself.
/// Generates a synthetic gradient image and applies various operations,
/// printing info at each step. No external image I/O required.
/// Run with `moon run cmd/main` from the project root.
fn main {
  println("=== millow Demo (synthetic) ===")
  println(" ")

  // Create a synthetic 128x128 gradient image
  let h = 128
  let w = 128
  let img = @millow.Image::new(h, w)
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      let o = (y * w + x) * 4
      img.data[o] = (x * 2 % 256).to_byte()
      img.data[o + 1] = (y * 2 % 256).to_byte()
      img.data[o + 2] = ((x + y) % 256).to_byte()
      img.data[o + 3] = 255
    }
  }
  println("Created synthetic image: \{@millow.info(img)}")
  println(" ")

  // === Color operations ===
  let gray = @millow.to_grayscale(img)
  println("1. to_grayscale: \{@millow.info(gray)}")

  let inverted = @millow.invert(img)
  println("2. invert: \{@millow.info(inverted)}")

  let tinted = @millow.tint(img, 100, 150, 200)
  println("3. tint(100,150,200): \{@millow.info(tinted)}")
  println(" ")

  // === Enhancement ===
  let bright = @millow.adjust_brightness(img, 30)
  println("4. adjust_brightness(+30): \{@millow.info(bright)}")

  let contrast = @millow.adjust_contrast(img, 1.5)
  println("5. adjust_contrast(1.5): \{@millow.info(contrast)}")

  let gamma = @millow.adjust_gamma(img, 1.5)
  println("6. adjust_gamma(1.5): \{@millow.info(gamma)}")

  let sharpened = @millow.sharpen(img, 1.0)
  println("7. sharpen(1.0): \{@millow.info(sharpened)}")
  println(" ")

  // === Filters ===
  let blurred = @millow.gaussian_blur(img, 2.0)
  println("8. gaussian_blur(σ=2.0): \{@millow.info(blurred)}")

  let median = @millow.median_filter(img, 3)
  println("9. median_filter(3): \{@millow.info(median)}")

  let bilateral = @millow.bilateral_filter(gray, 5, 10.0, 10.0)
  println("10. bilateral_filter: \{@millow.info(bilateral)}")
  println(" ")

  // === Edge detection ===
  let sobel = @millow.sobel(gray)
  println("11. sobel: \{@millow.info(sobel)}")

  let scharr = @millow.scharr(gray)
  println("12. scharr: \{@millow.info(scharr)}")

  let laplacian = @millow.laplacian(gray)
  println("13. laplacian: \{@millow.info(laplacian)}")
  println(" ")

  // === Morphology ===
  let eroded = @millow.erode(gray, @millow.Kernel::Square(3))
  println("14. erode: \{@millow.info(eroded)}")

  let dilated = @millow.dilate(gray, @millow.Kernel::Square(3))
  println("15. dilate: \{@millow.info(dilated)}")

  let morph_open = @millow.morph_open(gray, @millow.Kernel::Square(3))
  println("16. morph_open: \{@millow.info(morph_open)}")

  let morph_close = @millow.morph_close(gray, @millow.Kernel::Square(3))
  println("17. morph_close: \{@millow.info(morph_close)}")
  println(" ")

  // === Geometry ===
  let rotated = @millow.rotate_any(img, 45.0, @millow.Interp::Bilinear)
  println("18. rotate_any(45°): \{@millow.info(rotated)}")

  let flipped_h = @millow.flip_horizontal(img)
  println("19. flip_horizontal: \{@millow.info(flipped_h)}")

  let flipped_v = @millow.flip_vertical(img)
  println("20. flip_vertical: \{@millow.info(flipped_v)}")

  let resized = @millow.resize(img, 64, 64, @millow.Interp::Bilinear)
  println("21. resize(64x64): \{@millow.info(resized)}")

  let translated = @millow.translate(img, 20.0, 20.0, @millow.Interp::Bilinear)
  println("22. translate(20,20): \{@millow.info(translated)}")
  println(" ")

  // === Threshold & Histogram ===
  let (otsu_thr, otsu_binary) = @millow.threshold_otsu(gray)
  println(
    "23. threshold_otsu(thr=\{otsu_thr.to_int()}): \{@millow.info(otsu_binary)}",
  )

  let equalized = @millow.equalize_histogram(gray)
  println("24. equalize_histogram: \{@millow.info(equalized)}")

  let sauvola_binary = @millow.threshold_sauvola(gray, 15, 0.2)
  println("25. threshold_sauvola: \{@millow.info(sauvola_binary)}")
  println(" ")

  // === Feature detection ===
  let corners = @millow.corner_harris(gray, 0.04)
  let corner_count = {
    let mut n = 0
    for row in corners {
      for v in row {
        if v > 0.1 {
          n = n + 1
        }
      }
    }
    n
  }
  println("26. corner_harris: \{corner_count} corners above 0.1")

  let lbp_codes = @millow.lbp(gray, 1, 8)
  println("27. lbp(radius=1, n=8): \{lbp_codes.length()} codes")
  println(" ")

  // === Measurement ===
  let (_labels, cc_count) = @millow.connected_components(otsu_binary, 4)
  println("28. connected_components: \{cc_count} components")

  let contours = @millow.find_contours(otsu_binary)
  println("29. find_contours: \{contours.length()} contours found")

  let m = @millow.moments(gray)
  println("30. moments: m00=\{m[0]}, centroid=(\{m[1] / m[0]}, \{m[2] / m[0]})")

  let hu = @millow.hu_moments(gray)
  println("31. hu_moments: \{hu.length()} values (h1=\{hu[0]})")
  println(" ")

  // === Augmentation pipeline ===
  let augmented = @millow.augment_pipeline(img, [
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
  println("32. augment_pipeline (5 ops): \{@millow.info(augmented)}")
  println(" ")

  // === Advanced filters (v0.3.0) ===
  let guided = @millow.guided_filter_self(gray, 5, 0.01)
  println("33. guided_filter_self(5, 0.01): \{@millow.info(guided)}")

  let wiener = @millow.wiener_filter(gray, 5, -1.0)
  println("34. wiener_filter(5, auto): \{@millow.info(wiener)}")

  let dog_img = @millow.dog(gray, 1.0, 2.0)
  println("35. dog(1.0, 2.0): \{@millow.info(dog_img)}")

  let bernsen = @millow.threshold_bernsen(gray, 15, 15)
  println("36. threshold_bernsen(15, 15): \{@millow.info(bernsen)}")

  let wolf = @millow.threshold_wolf(gray, 15, 0.5)
  println("37. threshold_wolf(15, 0.5): \{@millow.info(wolf)}")

  let adap_mean = @millow.adaptive_threshold_mean(gray, 15, 5.0)
  println("38. adaptive_threshold_mean(15, 5): \{@millow.info(adap_mean)}")

  let adap_gauss = @millow.adaptive_threshold_gaussian(gray, 15, 5.0)
  println("39. adaptive_threshold_gaussian(15, 5): \{@millow.info(adap_gauss)}")

  let edges = @millow.canny(gray, 50.0, 150.0)
  let lines = @millow.hough_lines(edges, 80)
  println("40. hough_lines: \{lines.length()} lines (threshold=80)")

  let cleaned = @millow.remove_lines(
    otsu_binary,
    @millow.LineDirection::Both,
    2,
  )
  println("41. remove_lines(Both, 2): \{@millow.info(cleaned)}")

  let (ph, pw) = img.shape()
  let src_pts = [
    (0.0, 0.0),
    (pw.to_double(), 0.0),
    (pw.to_double(), ph.to_double()),
    (0.0, ph.to_double()),
  ]
  let dst_pts = [
    (20.0, 20.0),
    (pw.to_double() - 20.0, 30.0),
    (pw.to_double() - 30.0, ph.to_double() - 20.0),
    (30.0, ph.to_double() - 30.0),
  ]
  let warped = @millow.perspective_4pt(
    img,
    src_pts,
    dst_pts,
    @millow.Interp::Bilinear,
  )
  println("42. perspective_4pt: \{@millow.info(warped)}")

  let model_input = @millow.resize_to_model_input(
    img,
    224,
    224,
    interp=@millow.Interp::Bilinear,
  )
  println("43. resize_to_model_input(224x224): \{@millow.info(model_input)}")

  let letterboxed = @millow.letterbox(img, 224, 224, color=(114, 114, 114, 255))
  println("44. letterbox(224x224, color=114): \{@millow.info(letterboxed)}")

  let quant = @millow.kmeans_quantize(img, 8, seed=42)
  println("45. kmeans_quantize(k=8, seed=42): \{@millow.info(quant)}")

  let dt = @millow.distance_transform(otsu_binary)
  let mut dt_max = 0.0
  for row in dt {
    for v in row {
      if v > dt_max {
        dt_max = v
      }
    }
  }
  println("46. distance_transform: max distance = \{dt_max}")
  println(" ")

  // === Metrics ===
  println("Quality metrics:")
  println("  PSNR(original, blurred) = \{@millow.psnr(img, blurred)} dB")
  println("  MSE(original, blurred)  = \{@millow.mse(img, blurred)}")
  println("  SSIM(original, blurred) = \{@millow.ssim(img, blurred)}")
  println(" ")

  // === PPM serialization ===
  let ppm = @millow.to_ppm(img)
  println("PPM serialization: \{ppm.length()} bytes")
  println(" ")

  println("=== Demo complete! ===")
  println("For full demo with JPEG I/O, see examples/ directory.")
}
