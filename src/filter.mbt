///|
/// Read RGB channel `c` at clamped coordinates as a double.
fn at_clamped(img : Image, y : Int, x : Int, c : Int) -> Double {
  let cy = clampi(y, 0, img.h - 1)
  let cx = clampi(x, 0, img.w - 1)
  img.data[img.offset(cy, cx) + c].to_double()
}

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
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let oo = out.offset(y, x)
      for c = 0; c < 3; c = c + 1 {
        let mut acc = 0.0
        for ky = 0; ky < kh; ky = ky + 1 {
          for kx = 0; kx < kw; kx = kx + 1 {
            acc = acc +
              kernel[ky][kx] * at_clamped(img, y + ky - ay, x + kx - ax, c)
          }
        }
        out.data[oo + c] = round_byte(acc / norm)
      }
      out.data[oo + 3] = img.data[img.offset(y, x) + 3]
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
fn conv_1d(img : Image, k : Array[Double], horizontal : Bool) -> Image {
  let c = k.length() / 2
  let out = Image::new(img.h, img.w)
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let oo = out.offset(y, x)
      for ch = 0; ch < 3; ch = ch + 1 {
        let mut acc = 0.0
        for i = 0; i < k.length(); i = i + 1 {
          let d = i - c
          let v = if horizontal {
            at_clamped(img, y, x + d, ch)
          } else {
            at_clamped(img, y + d, x, ch)
          }
          acc = acc + k[i] * v
        }
        out.data[oo + ch] = round_byte(acc)
      }
      out.data[oo + 3] = img.data[img.offset(y, x) + 3]
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
  let temp = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      let mut acc = 0.0
      for i = 0; i < ks; i = i + 1 {
        let dx = i - c
        let cx = x + dx
        let v = if cx >= 0 && cx < w { arr[y][cx] } else { 0.0 }
        acc = acc + k[i] * v
      }
      temp[y][x] = acc
    }
  }
  let out = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      let mut acc = 0.0
      for i = 0; i < ks; i = i + 1 {
        let dy = i - c
        let cy = y + dy
        let v = if cy >= 0 && cy < h { temp[cy][x] } else { 0.0 }
        acc = acc + k[i] * v
      }
      out[y][x] = acc
    }
  }
  out
}
