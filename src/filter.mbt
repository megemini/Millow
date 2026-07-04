///|
/// General 2D convolution over the RGB channels (border: replicate).
/// When `normalize` is true, the kernel is divided by its sum (if non-zero).
///
/// - `img`: input RGBA8 image; alpha is copied unchanged.
/// - `kernel`: 2D weight matrix; the anchor is the centre (`kh/2, kw/2`).
/// - `normalize`: divide the response by the kernel sum when `true`, by `1.0`
///   otherwise.
pub fn convolve(
  img : Image,
  kernel : Array[Array[Double]],
  normalize : Bool,
) -> Image {
  let kh = kernel.length()
  let kw = kernel[0].length()
  let ay = kh / 2
  let ax = kw / 2
  let mut ksum = 0.0
  for ky = 0; ky < kh; ky = ky + 1 {
    for kx = 0; kx < kw; kx = kx + 1 {
      ksum = ksum + kernel[ky][kx]
    }
  }
  let norm = if normalize && ksum != 0.0 { ksum } else { 1.0 }
  let out = Image::new(img.h, img.w)
  let h = img.h
  let w = img.w
  let stride = w * 4
  let hmax = h - 1
  let wmax = w - 1
  // Flatten kernel for better cache locality
  let flat_k = Array::make(kh * kw, 0.0)
  for ky = 0; ky < kh; ky = ky + 1 {
    for kx = 0; kx < kw; kx = kx + 1 {
      flat_k[ky * kw + kx] = kernel[ky][kx]
    }
  }
  for y = 0; y < h; y = y + 1 {
    let row_base = y * stride
    for x = 0; x < w; x = x + 1 {
      let oo = row_base + x * 4
      let mut acc_r = 0.0
      let mut acc_g = 0.0
      let mut acc_b = 0.0
      for ky = 0; ky < kh; ky = ky + 1 {
        let cy = y + ky - ay
        let clamped_y = if cy < 0 { 0 } else if cy > hmax { hmax } else { cy }
        let k_row = clamped_y * stride
        let k_row_base = ky * kw
        for kx = 0; kx < kw; kx = kx + 1 {
          let cx = x + kx - ax
          let clamped_x = if cx < 0 { 0 } else if cx > wmax { wmax } else { cx }
          let idx = k_row + clamped_x * 4
          let ki = flat_k[k_row_base + kx]
          acc_r = acc_r + ki * img.data[idx].to_double()
          acc_g = acc_g + ki * img.data[idx + 1].to_double()
          acc_b = acc_b + ki * img.data[idx + 2].to_double()
        }
      }
      out.data[oo] = round_byte(acc_r / norm)
      out.data[oo + 1] = round_byte(acc_g / norm)
      out.data[oo + 2] = round_byte(acc_b / norm)
      out.data[oo + 3] = img.data[oo + 3]
    }
  }
  out
}

///|
/// Build a normalized 1D Gaussian kernel.
fn gaussian_kernel(ksize : Int, sigma : Double) -> Array[Double] {
  let k = Array::make(ksize, 0.0)
  let c = ksize / 2
  let s2 = 2.0 * sigma * sigma
  let mut sum = 0.0
  for i = 0; i < ksize; i = i + 1 {
    let d = (i - c).to_double()
    let v = @math.exp(-d * d / s2)
    k[i] = v
    sum = sum + v
  }
  for i = 0; i < ksize; i = i + 1 {
    k[i] = k[i] / sum
  }
  k
}

///|
/// Separable 1D convolution along one axis (RGB only, alpha copied).
/// Inlined bounds checking and interleaved channel processing for speed.
fn conv_1d(img : Image, k : Array[Double], horizontal : Bool) -> Image {
  let radius = k.length() / 2
  let klen = k.length()
  let out = Image::new(img.h, img.w)
  let h = img.h
  let w = img.w
  let stride = w * 4
  let hmax = h - 1
  let wmax = w - 1
  if horizontal {
    let interior_start = radius
    let interior_end = w - radius
    let has_interior = interior_start < interior_end
    for y = 0; y < h; y = y + 1 {
      let row_base = y * stride
      // Interior pixels: skip bounds check
      if has_interior {
        for x = interior_start; x < interior_end; x = x + 1 {
          let oo = row_base + x * 4
          let mut acc_r = 0.0
          let mut acc_g = 0.0
          let mut acc_b = 0.0
          let base_idx = oo - radius * 4
          for i = 0; i < klen; i = i + 1 {
            let ki = k[i]
            let idx = base_idx + i * 4
            acc_r = acc_r + ki * img.data[idx].to_double()
            acc_g = acc_g + ki * img.data[idx + 1].to_double()
            acc_b = acc_b + ki * img.data[idx + 2].to_double()
          }
          out.data[oo] = round_byte(acc_r)
          out.data[oo + 1] = round_byte(acc_g)
          out.data[oo + 2] = round_byte(acc_b)
          out.data[oo + 3] = img.data[oo + 3]
        }
        // Left border
        for x = 0; x < interior_start; x = x + 1 {
          let oo = row_base + x * 4
          let mut acc_r = 0.0
          let mut acc_g = 0.0
          let mut acc_b = 0.0
          for i = 0; i < klen; i = i + 1 {
            let ki = k[i]
            let cx = x + i - radius
            let clamped = if cx < 0 { 0 } else { cx }
            let idx = row_base + clamped * 4
            acc_r = acc_r + ki * img.data[idx].to_double()
            acc_g = acc_g + ki * img.data[idx + 1].to_double()
            acc_b = acc_b + ki * img.data[idx + 2].to_double()
          }
          out.data[oo] = round_byte(acc_r)
          out.data[oo + 1] = round_byte(acc_g)
          out.data[oo + 2] = round_byte(acc_b)
          out.data[oo + 3] = img.data[oo + 3]
        }
        // Right border
        for x = interior_end; x < w; x = x + 1 {
          let oo = row_base + x * 4
          let mut acc_r = 0.0
          let mut acc_g = 0.0
          let mut acc_b = 0.0
          for i = 0; i < klen; i = i + 1 {
            let ki = k[i]
            let cx = x + i - radius
            let clamped = if cx > wmax { wmax } else { cx }
            let idx = row_base + clamped * 4
            acc_r = acc_r + ki * img.data[idx].to_double()
            acc_g = acc_g + ki * img.data[idx + 1].to_double()
            acc_b = acc_b + ki * img.data[idx + 2].to_double()
          }
          out.data[oo] = round_byte(acc_r)
          out.data[oo + 1] = round_byte(acc_g)
          out.data[oo + 2] = round_byte(acc_b)
          out.data[oo + 3] = img.data[oo + 3]
        }
      } else {
        // Image smaller than kernel: all border
        for x = 0; x < w; x = x + 1 {
          let oo = row_base + x * 4
          let mut acc_r = 0.0
          let mut acc_g = 0.0
          let mut acc_b = 0.0
          for i = 0; i < klen; i = i + 1 {
            let ki = k[i]
            let cx = x + i - radius
            let clamped = if cx < 0 { 0 } else if cx > wmax { wmax } else { cx }
            let idx = row_base + clamped * 4
            acc_r = acc_r + ki * img.data[idx].to_double()
            acc_g = acc_g + ki * img.data[idx + 1].to_double()
            acc_b = acc_b + ki * img.data[idx + 2].to_double()
          }
          out.data[oo] = round_byte(acc_r)
          out.data[oo + 1] = round_byte(acc_g)
          out.data[oo + 2] = round_byte(acc_b)
          out.data[oo + 3] = img.data[oo + 3]
        }
      }
    }
  } else {
    // Vertical pass
    let interior_start = radius
    let interior_end = h - radius
    let has_interior = interior_start < interior_end
    for y = 0; y < h; y = y + 1 {
      let row_base = y * stride
      let is_interior = has_interior && y >= interior_start && y < interior_end
      for x = 0; x < w; x = x + 1 {
        let oo = row_base + x * 4
        let mut acc_r = 0.0
        let mut acc_g = 0.0
        let mut acc_b = 0.0
        if is_interior {
          let base_idx = oo - radius * stride
          for i = 0; i < klen; i = i + 1 {
            let ki = k[i]
            let idx = base_idx + i * stride
            acc_r = acc_r + ki * img.data[idx].to_double()
            acc_g = acc_g + ki * img.data[idx + 1].to_double()
            acc_b = acc_b + ki * img.data[idx + 2].to_double()
          }
        } else {
          for i = 0; i < klen; i = i + 1 {
            let ki = k[i]
            let cy = y + i - radius
            let clamped = if cy < 0 { 0 } else if cy > hmax { hmax } else { cy }
            let idx = clamped * stride + x * 4
            acc_r = acc_r + ki * img.data[idx].to_double()
            acc_g = acc_g + ki * img.data[idx + 1].to_double()
            acc_b = acc_b + ki * img.data[idx + 2].to_double()
          }
        }
        out.data[oo] = round_byte(acc_r)
        out.data[oo + 1] = round_byte(acc_g)
        out.data[oo + 2] = round_byte(acc_b)
        out.data[oo + 3] = img.data[oo + 3]
      }
    }
  }
  out
}

///|
/// Gaussian blur with an explicit (odd) kernel size.
///
/// - `img`: input RGBA8 image; alpha is copied unchanged.
/// - `ksize`: kernel diameter; an even value is rounded up to the next odd
///   integer.
/// - `sigma`: Gaussian standard deviation; weights are `exp(-d²/(2σ²))`
///   normalised, applied as a separable 1D×1D pass with replicate borders.
pub fn gaussian_blur_kernel(img : Image, ksize : Int, sigma : Double) -> Image {
  let ks = if ksize % 2 == 0 { ksize + 1 } else { ksize }
  let k = gaussian_kernel(ks, sigma)
  conv_1d(conv_1d(img, k, true), k, false)
}

///|
/// Gaussian blur; kernel size is derived from `sigma`.
///
/// - `img`: input RGBA8 image; alpha is copied unchanged.
/// - `sigma`: Gaussian standard deviation. The kernel size is
///   `clampi(ceil(sigma*3)*2+1, 3, 99)`; weights are `exp(-d²/(2σ²))`
///   normalised, applied as a separable 1D×1D pass with replicate borders.
pub fn gaussian_blur(img : Image, sigma : Double) -> Image {
  let radius = @math.ceil(sigma * 3.0).to_int()
  let ks = clampi(radius * 2 + 1, 3, 99)
  gaussian_blur_kernel(img, ks, sigma)
}

///|
/// Gaussian blur for double arrays (used for structure tensor smoothing).
///
/// - `arr`: 2D `h × w` array of doubles to smooth in place of an image.
/// - `sigma`: Gaussian standard deviation; the kernel size is
///   `clampi(ceil(sigma*3)*2+1, 3, 99)`. Out-of-bounds samples are treated
///   as `0.0` (zero-padding), unlike the image variants which replicate.
pub fn gaussian_blur_double(
  arr : Array[Array[Double]],
  sigma : Double,
) -> Array[Array[Double]] {
  let h = arr.length()
  let w = arr[0].length()
  let radius = @math.ceil(sigma * 3.0).to_int()
  let ks = clampi(radius * 2 + 1, 3, 99)
  let k = gaussian_kernel(ks, sigma)
  let c = ks / 2
  // Flatten input for cache-friendly access
  let flat_in = Array::make(h * w, 0.0)
  for y = 0; y < h; y = y + 1 {
    let row_base = y * w
    for x = 0; x < w; x = x + 1 {
      flat_in[row_base + x] = arr[y][x]
    }
  }
  // Horizontal pass (zero-padding border)
  let temp = Array::make(h * w, 0.0)
  let wmax = w - 1
  for y = 0; y < h; y = y + 1 {
    let row_base = y * w
    for x = 0; x < w; x = x + 1 {
      let mut acc = 0.0
      for i = 0; i < ks; i = i + 1 {
        let cx = x + i - c
        let v = if cx < 0 || cx > wmax { 0.0 } else { flat_in[row_base + cx] }
        acc = acc + k[i] * v
      }
      temp[row_base + x] = acc
    }
  }
  // Vertical pass (zero-padding border)
  let flat_out = Array::make(h * w, 0.0)
  let hmax = h - 1
  for y = 0; y < h; y = y + 1 {
    let row_base = y * w
    for x = 0; x < w; x = x + 1 {
      let mut acc = 0.0
      for i = 0; i < ks; i = i + 1 {
        let cy = y + i - c
        let v = if cy < 0 || cy > hmax { 0.0 } else { temp[cy * w + x] }
        acc = acc + k[i] * v
      }
      flat_out[row_base + x] = acc
    }
  }
  // Unflatten
  let out = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  for y = 0; y < h; y = y + 1 {
    let row_base = y * w
    for x = 0; x < w; x = x + 1 {
      out[y][x] = flat_out[row_base + x]
    }
  }
  out
}

///|
/// Fused Gaussian blur of three double arrays simultaneously.
/// Avoids triple flatten/unflatten overhead when smoothing structure tensor
/// components. Returns `(out1, out2, out3)`.
pub fn gaussian_blur_double3(
  arr1 : Array[Array[Double]],
  arr2 : Array[Array[Double]],
  arr3 : Array[Array[Double]],
  sigma : Double,
) -> (Array[Array[Double]], Array[Array[Double]], Array[Array[Double]]) {
  let h = arr1.length()
  let w = arr1[0].length()
  let radius = @math.ceil(sigma * 3.0).to_int()
  let ks = clampi(radius * 2 + 1, 3, 99)
  let k = gaussian_kernel(ks, sigma)
  let c = ks / 2
  let n = h * w
  // Flatten all three inputs
  let in1 = Array::make(n, 0.0)
  let in2 = Array::make(n, 0.0)
  let in3 = Array::make(n, 0.0)
  for y = 0; y < h; y = y + 1 {
    let row_base = y * w
    let r1 = arr1[y]
    let r2 = arr2[y]
    let r3 = arr3[y]
    for x = 0; x < w; x = x + 1 {
      in1[row_base + x] = r1[x]
      in2[row_base + x] = r2[x]
      in3[row_base + x] = r3[x]
    }
  }
  // Horizontal pass (zero-padding border)
  let t1 = Array::make(n, 0.0)
  let t2 = Array::make(n, 0.0)
  let t3 = Array::make(n, 0.0)
  let wmax = w - 1
  for y = 0; y < h; y = y + 1 {
    let row_base = y * w
    for x = 0; x < w; x = x + 1 {
      let mut a1 = 0.0
      let mut a2 = 0.0
      let mut a3 = 0.0
      for i = 0; i < ks; i = i + 1 {
        let cx = x + i - c
        let ki = k[i]
        if cx < 0 || cx > wmax {
          // v = 0, no contribution
          continue
        }
        let idx = row_base + cx
        a1 = a1 + ki * in1[idx]
        a2 = a2 + ki * in2[idx]
        a3 = a3 + ki * in3[idx]
      }
      t1[row_base + x] = a1
      t2[row_base + x] = a2
      t3[row_base + x] = a3
    }
  }
  // Vertical pass (zero-padding border)
  let o1 = Array::make(n, 0.0)
  let o2 = Array::make(n, 0.0)
  let o3 = Array::make(n, 0.0)
  let hmax = h - 1
  for y = 0; y < h; y = y + 1 {
    let row_base = y * w
    for x = 0; x < w; x = x + 1 {
      let mut a1 = 0.0
      let mut a2 = 0.0
      let mut a3 = 0.0
      for i = 0; i < ks; i = i + 1 {
        let cy = y + i - c
        let ki = k[i]
        if cy < 0 || cy > hmax {
          continue
        }
        let idx = cy * w + x
        a1 = a1 + ki * t1[idx]
        a2 = a2 + ki * t2[idx]
        a3 = a3 + ki * t3[idx]
      }
      o1[row_base + x] = a1
      o2[row_base + x] = a2
      o3[row_base + x] = a3
    }
  }
  // Unflatten
  let out1 = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  let out2 = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  let out3 = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  for y = 0; y < h; y = y + 1 {
    let row_base = y * w
    let r1 = out1[y]
    let r2 = out2[y]
    let r3 = out3[y]
    for x = 0; x < w; x = x + 1 {
      r1[x] = o1[row_base + x]
      r2[x] = o2[row_base + x]
      r3[x] = o3[row_base + x]
    }
  }
  (out1, out2, out3)
}
