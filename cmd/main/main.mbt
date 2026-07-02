///|
/// Small end-to-end demonstration of the `millow` image-processing library.
/// It builds a synthetic canvas, runs a color/enhancement/edge pipeline and
/// reports a few statistics. Run with `moon run cmd/main` from the project root.
fn main {
  // Start from a solid canvas and draw a few primitives on it.
  let base = @millow.Image::from_pixel(64, 64, 30, 60, 90, 255)
  let a = @millow.draw_rect(base, 8, 8, 24, 40, 220, 40, 40, 255, true, 1)
  let b = @millow.draw_circle(a, 40, 40, 15, 40, 200, 80, 255, true, 1)
  let canvas = @millow.draw_line(b, 0, 0, 63, 63, 255, 255, 255, 255, 2)
  println("canvas : \{@millow.info(canvas)}")

  // Color + enhancement pipeline.
  let gray = @millow.to_grayscale(canvas)
  let bright = @millow.adjust_brightness(gray, 20)
  let blurred = @millow.gaussian_blur(bright, 1.5)
  println("blurred: \{@millow.info(blurred)}")

  // Edge detection and Otsu adaptive threshold.
  let edges = @millow.sobel(blurred)
  let (thr, binary) = @millow.threshold_otsu(blurred)
  println("otsu threshold = \{thr.to_int()}")
  println("edges  : \{@millow.info(edges)}")
  println("binary : \{@millow.info(binary)}")

  // Quality metric between the grayscale source and its blurred version.
  println("psnr(gray, blurred) = \{@millow.psnr(gray, blurred)} dB")

  // Geometric downscale plus PPM serialization size.
  let thumb = @millow.resize(canvas, 16, 16, @millow.Interp::Nearest)
  let ppm = @millow.to_ppm(thumb)
  println("thumb  : \{@millow.info(thumb)}, ppm bytes = \{ppm.length()}")
}
