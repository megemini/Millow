///|
/// Nearest-neighbor resampling.
fn resize_nearest(img : Image, dst_h : Int, dst_w : Int) -> Image {
  let out = Image::new(dst_h, dst_w)
  if img.is_empty() || dst_h == 0 || dst_w == 0 {
    return out
  }
  for y = 0; y < dst_h; y = y + 1 {
    let sy = clampi(y * img.h / dst_h, 0, img.h - 1)
    for x = 0; x < dst_w; x = x + 1 {
      let sx = clampi(x * img.w / dst_w, 0, img.w - 1)
      copy_px(img, img.offset(sy, sx), out, out.offset(y, x))
    }
  }
  out
}

///|
/// Bilinear filter weight: 1 - |x| for |x| < 1, else 0.
fn bilinear_filter(x : Double) -> Double {
  let ax = if x < 0.0 { -x } else { x }
  if ax < 1.0 {
    1.0 - ax
  } else {
    0.0
  }
}

///|
/// Precompute bilinear weights for one axis.
fn precompute_bilinear_weights(
  src_size : Int,
  dst_size : Int,
) -> Array[(Array[Double], Int, Int)] {
  let scale = src_size.to_double() / dst_size.to_double()
  let support = if scale < 1.0 { 1.0 } else { scale }
  let inv_fs = 1.0 / support
  let result = Array::make(dst_size, ([], 0, 0))
  for i = 0; i < dst_size; i = i + 1 {
    let center = (i.to_double() + 0.5) * scale
    let xmin = (center - support + 0.5).floor().to_int().max(0)
    let xmax = (center + support + 0.5).ceil().to_int().min(src_size)
    let n = xmax - xmin
    let weights = Array::make(n, 0.0)
    let mut ww = 0.0
    for j = 0; j < n; j = j + 1 {
      let w = bilinear_filter(((xmin + j).to_double() - center + 0.5) * inv_fs)
      weights[j] = w
      ww = ww + w
    }
    if ww > 0.0 {
      for j = 0; j < n; j = j + 1 {
        weights[j] = weights[j] / ww
      }
    }
    result[i] = (weights, xmin, xmax)
  }
  result
}

///|
/// Bilinear resampling (separable, matches Pillow).
fn resize_bilinear(img : Image, dst_h : Int, dst_w : Int) -> Image {
  let out = Image::new(dst_h, dst_w)
  if img.is_empty() || dst_h == 0 || dst_w == 0 {
    return out
  }
  let h = img.h
  let w = img.w
  // Horizontal pass: src (h, w, 4) -> temp (h, dst_w, 4)
  let x_weights = precompute_bilinear_weights(w, dst_w)
  let temp = Array::make(h * dst_w * 4, 0.0)
  for y = 0; y < h; y = y + 1 {
    for dx = 0; dx < dst_w; dx = dx + 1 {
      let (wts, xmin, xmax) = x_weights[dx]
      for c = 0; c < 4; c = c + 1 {
        let mut acc = 0.0
        for j = 0; j < xmax - xmin; j = j + 1 {
          let sx = xmin + j
          acc = acc + wts[j] * img.data[(y * w + sx) * 4 + c].to_double()
        }
        temp[(y * dst_w + dx) * 4 + c] = acc
      }
    }
  }
  // Vertical pass: temp (h, dst_w, 4) -> final (dst_h, dst_w, 4)
  let y_weights = precompute_bilinear_weights(h, dst_h)
  for dy = 0; dy < dst_h; dy = dy + 1 {
    let (wts, ymin, ymax) = y_weights[dy]
    for x = 0; x < dst_w; x = x + 1 {
      for c = 0; c < 4; c = c + 1 {
        let mut acc = 0.0
        for j = 0; j < ymax - ymin; j = j + 1 {
          let sy = ymin + j
          acc = acc + wts[j] * temp[(sy * dst_w + x) * 4 + c]
        }
        out.data[(dy * dst_w + x) * 4 + c] = round_byte(acc)
      }
    }
  }
  out
}

///|
/// Cubic kernel for bicubic interpolation (a = -0.5).
fn cubic_w(t : Double) -> Double {
  let a = -0.5
  let at = if t < 0.0 { -t } else { t }
  if at <= 1.0 {
    (a + 2.0) * at * at * at - (a + 3.0) * at * at + 1.0
  } else if at < 2.0 {
    a * at * at * at - 5.0 * a * at * at + 8.0 * a * at - 4.0 * a
  } else {
    0.0
  }
}

///|
/// Bicubic resampling using a 4x4 neighborhood with the standard cubic kernel.
fn resize_bicubic(img : Image, dst_h : Int, dst_w : Int) -> Image {
  let out = Image::new(dst_h, dst_w)
  if img.is_empty() || dst_h == 0 || dst_w == 0 {
    return out
  }
  let sh = img.h.to_double()
  let sw = img.w.to_double()
  for y = 0; y < dst_h; y = y + 1 {
    let fy = clampd(
      (y.to_double() + 0.5) * sh / dst_h.to_double() - 0.5,
      0.0,
      sh - 1.0,
    )
    let y0 = @math.floor(fy).to_int()
    let wy = fy - y0.to_double()
    for x = 0; x < dst_w; x = x + 1 {
      let fx = clampd(
        (x.to_double() + 0.5) * sw / dst_w.to_double() - 0.5,
        0.0,
        sw - 1.0,
      )
      let x0 = @math.floor(fx).to_int()
      let wx = fx - x0.to_double()
      let oo = out.offset(y, x)
      for c = 0; c < 4; c = c + 1 {
        let mut acc = 0.0
        for j = -1; j <= 2; j = j + 1 {
          let wyi = cubic_w(wy - j.to_double())
          let sy = clampi(y0 + j, 0, img.h - 1)
          for i = -1; i <= 2; i = i + 1 {
            let wxi = cubic_w(wx - i.to_double())
            let sx = clampi(x0 + i, 0, img.w - 1)
            acc = acc + wyi * wxi * img.data[img.offset(sy, sx) + c].to_double()
          }
        }
        out.data[oo + c] = round_byte(acc)
      }
    }
  }
  out
}

///|
/// Resize the image to `(dst_h, dst_w)` using the given interpolation.
///
/// - `dst_h`, `dst_w`: target dimensions in pixels.
/// - `interp`: resampling method (`Nearest`, `Bilinear`, or `Bicubic`).
pub fn resize(img : Image, dst_h : Int, dst_w : Int, interp : Interp) -> Image {
  match interp {
    Nearest => resize_nearest(img, dst_h, dst_w)
    Bilinear => resize_bilinear(img, dst_h, dst_w)
    Bicubic => resize_bicubic(img, dst_h, dst_w)
  }
}

///|
/// Scale both dimensions by a uniform factor.
///
/// - `scale`: scaling factor (e.g. `0.5` halves, `2.0` doubles).
/// - `interp`: resampling method (`Nearest`, `Bilinear`, or `Bicubic`).
///
/// Output dimensions are `round(h * scale) x round(w * scale)`.
pub fn rescale(img : Image, scale : Double, interp : Interp) -> Image {
  let dh = @math.round(img.h.to_double() * scale).to_int()
  let dw = @math.round(img.w.to_double() * scale).to_int()
  resize(img, dh, dw, interp)
}

///|
/// Scale to fit inside `max_h x max_w`, preserving aspect ratio.
///
/// - `max_h`, `max_w`: maximum bounding dimensions.
/// - `interp`: resampling method (`Nearest`, `Bilinear`, or `Bicubic`).
///
/// The result fits entirely within the box and at least one dimension equals
/// its bound. Returns a clone if the image is empty.
pub fn resize_to_fit(
  img : Image,
  max_h : Int,
  max_w : Int,
  interp : Interp,
) -> Image {
  if img.is_empty() {
    return img.clone()
  }
  let sy = max_h.to_double() / img.h.to_double()
  let sx = max_w.to_double() / img.w.to_double()
  let s = if sy < sx { sy } else { sx }
  rescale(img, s, interp)
}

///|
/// Scale to cover `min_h x min_w`, preserving aspect ratio, cropping overflow.
///
/// - `min_h`, `min_w`: target dimensions to fully cover.
/// - `interp`: resampling method (`Nearest`, `Bilinear`, or `Bicubic`).
///
/// The image is scaled then center-cropped to `(min_h, min_w)`. Returns a
/// clone if the image is empty.
pub fn resize_to_cover(
  img : Image,
  min_h : Int,
  min_w : Int,
  interp : Interp,
) -> Image raise ImageError {
  if img.is_empty() {
    return img.clone()
  }
  let sy = min_h.to_double() / img.h.to_double()
  let sx = min_w.to_double() / img.w.to_double()
  let s = if sy > sx { sy } else { sx }
  let scaled = rescale(img, s, interp)
  let oy = clampi((scaled.h - min_h) / 2, 0, scaled.h)
  let ox = clampi((scaled.w - min_w) / 2, 0, scaled.w)
  crop(scaled, oy, ox, clampi(min_h, 0, scaled.h), clampi(min_w, 0, scaled.w))
}

///|
/// Resize to fit a model's fixed input size while preserving aspect ratio,
/// then center-pad to exactly `(model_h, model_w)`.
///
/// This is the standard preprocessing pipeline for OCR/CV models that require
/// a fixed input shape (e.g. CRNN 32×512, ViT 384×384).
///
/// - `model_h`, `model_w`: exact target dimensions.
/// - `interp`: resampling method (default `Bilinear`).
/// - `pad_color`: RGBA fill color for padding (default white).
pub fn resize_to_model_input(
  img : Image,
  model_h : Int,
  model_w : Int,
  interp? : Interp = Bilinear,
  pad_color? : (Byte, Byte, Byte, Byte) = (255, 255, 255, 255),
) -> Image {
  let fitted = resize_to_fit(img, model_h, model_w, interp)
  pad_to_size(
    fitted,
    model_h,
    model_w,
    Constant(pad_color.0, pad_color.1, pad_color.2, pad_color.3),
  )
}

///|
/// Resize to target size by letterboxing (pad with bars to preserve aspect
/// ratio). Alias of `resize_to_model_input` with a different default name
/// for familiarity with the common letterboxing technique.
///
/// - `target_h`, `target_w`: exact target dimensions.
/// - `color`: RGBA fill color for padding (default white).
pub fn letterbox(
  img : Image,
  target_h : Int,
  target_w : Int,
  color? : (Byte, Byte, Byte, Byte) = (255, 255, 255, 255),
) -> Image {
  let fitted = resize_to_fit(img, target_h, target_w, Bilinear)
  pad_to_size(
    fitted,
    target_h,
    target_w,
    Constant(color.0, color.1, color.2, color.3),
  )
}
