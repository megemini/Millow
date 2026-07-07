///|
/// Guided filter (edge-preserving smoothing) using the algorithm from
/// He et al. (2010). Complexity is O(n) via integral images.
///
/// - `guide`: guidance image (typically the input itself for self-guided).
/// - `img`: input image to filter.
/// - `radius`: filter window radius.
/// - `epsilon`: regularization parameter; larger = smoother.
///
/// Operates on luma; the output is a grayscale image where each pixel is
/// `q = mean_a * I + mean_b` with `a` and `b` computed from local linear
/// correlation between the guide and the input.
pub fn guided_filter(
  guide : Image,
  img : Image,
  radius : Int,
  epsilon : Double,
) -> Image {
  let h = guide.h
  let w = guide.w
  let stride = w * 4
  let w1 = w + 1
  // Extract luma arrays.
  let luma_g = Array::make(h * w, 0.0)
  let luma_p = Array::make(h * w, 0.0)
  for y = 0; y < h; y = y + 1 {
    let row = y * stride
    for x = 0; x < w; x = x + 1 {
      let o = row + x * 4
      let idx = y * w + x
      luma_g[idx] = luma(guide.data[o], guide.data[o + 1], guide.data[o + 2]).to_double()
      luma_p[idx] = luma(img.data[o], img.data[o + 1], img.data[o + 2]).to_double()
    }
  }
  // Build integral images for mean_I, mean_p, mean_Ip, mean_II.
  let integ_I = Array::make((h + 1) * w1, 0.0)
  let integ_p = Array::make((h + 1) * w1, 0.0)
  let integ_Ip = Array::make((h + 1) * w1, 0.0)
  let integ_II = Array::make((h + 1) * w1, 0.0)
  for y = 0; y < h; y = y + 1 {
    let mut sI = 0.0
    let mut sp = 0.0
    let mut sIp = 0.0
    let mut sII = 0.0
    for x = 0; x < w; x = x + 1 {
      let idx = y * w + x
      let ig = luma_g[idx]
      let ip = luma_p[idx]
      sI = sI + ig
      sp = sp + ip
      sIp = sIp + ig * ip
      sII = sII + ig * ig
      let oi = (y + 1) * w1 + (x + 1)
      integ_I[oi] = integ_I[y * w1 + (x + 1)] + sI
      integ_p[oi] = integ_p[y * w1 + (x + 1)] + sp
      integ_Ip[oi] = integ_Ip[y * w1 + (x + 1)] + sIp
      integ_II[oi] = integ_II[y * w1 + (x + 1)] + sII
    }
  }
  // Compute a and b per pixel.
  let arr_a = Array::make(h * w, 0.0)
  let arr_b = Array::make(h * w, 0.0)
  for y = 0; y < h; y = y + 1 {
    let y0 = clampi(y - radius, 0, h - 1)
    let y1 = clampi(y + radius, 0, h - 1)
    for x = 0; x < w; x = x + 1 {
      let x0 = clampi(x - radius, 0, w - 1)
      let x1 = clampi(x + radius, 0, w - 1)
      let n = ((y1 - y0 + 1) * (x1 - x0 + 1)).to_double()
      let mean_I = query_integral(integ_I, w1, y0, x0, y1, x1) / n
      let mean_p = query_integral(integ_p, w1, y0, x0, y1, x1) / n
      let mean_Ip = query_integral(integ_Ip, w1, y0, x0, y1, x1) / n
      let mean_II = query_integral(integ_II, w1, y0, x0, y1, x1) / n
      let cov_Ip = mean_Ip - mean_I * mean_p
      let var_I = mean_II - mean_I * mean_I
      let a = cov_Ip / (var_I + epsilon)
      let b = mean_p - a * mean_I
      let idx = y * w + x
      arr_a[idx] = a
      arr_b[idx] = b
    }
  }
  // Build integral images for a and b to compute mean_a, mean_b.
  let integ_a = Array::make((h + 1) * w1, 0.0)
  let integ_b = Array::make((h + 1) * w1, 0.0)
  for y = 0; y < h; y = y + 1 {
    let mut sa = 0.0
    let mut sb = 0.0
    for x = 0; x < w; x = x + 1 {
      let idx = y * w + x
      sa = sa + arr_a[idx]
      sb = sb + arr_b[idx]
      let oi = (y + 1) * w1 + (x + 1)
      integ_a[oi] = integ_a[y * w1 + (x + 1)] + sa
      integ_b[oi] = integ_b[y * w1 + (x + 1)] + sb
    }
  }
  // Final output: q = mean_a * I + mean_b.
  let out = Image::new(h, w)
  for y = 0; y < h; y = y + 1 {
    let y0 = clampi(y - radius, 0, h - 1)
    let y1 = clampi(y + radius, 0, h - 1)
    for x = 0; x < w; x = x + 1 {
      let x0 = clampi(x - radius, 0, w - 1)
      let x1 = clampi(x + radius, 0, w - 1)
      let n = ((y1 - y0 + 1) * (x1 - x0 + 1)).to_double()
      let mean_a = query_integral(integ_a, w1, y0, x0, y1, x1) / n
      let mean_b = query_integral(integ_b, w1, y0, x0, y1, x1) / n
      let idx = y * w + x
      let v = clamp_byte((mean_a * luma_g[idx] + mean_b).to_int())
      let o = out.offset(y, x)
      out.data[o] = v
      out.data[o + 1] = v
      out.data[o + 2] = v
      out.data[o + 3] = 255
    }
  }
  out
}

///|
/// Self-guided filter (guide = img). Convenience wrapper for the common case
/// where the guidance image is the input itself.
pub fn guided_filter_self(img : Image, radius : Int, epsilon : Double) -> Image {
  guided_filter(img, img, radius, epsilon)
}
