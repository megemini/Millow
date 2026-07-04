///|
/// Apply a per-channel window reducer over the RGB channels (alpha copied).
/// Uses a preallocated buffer to avoid per-pixel allocation.
fn window_map(img : Image, size : Int, reduce : (Array[Byte]) -> Byte) -> Image {
  let r = size / 2
  let h = img.h
  let w = img.w
  let stride = w * 4
  let out = Image::new(h, w)
  let hmax = h - 1
  let wmax = w - 1
  let win_n = (2 * r + 1) * (2 * r + 1)
  let buf = Array::make(win_n, (0 : Byte))
  for y = 0; y < h; y = y + 1 {
    let out_base = y * stride
    for x = 0; x < w; x = x + 1 {
      let oo = out_base + x * 4
      for c = 0; c < 3; c = c + 1 {
        let mut bi = 0
        for dy = -r; dy <= r; dy = dy + 1 {
          let cy = y + dy
          let clamped_y = if cy < 0 { 0 } else if cy > hmax { hmax } else { cy }
          let row_base = clamped_y * stride
          for dx = -r; dx <= r; dx = dx + 1 {
            let cx = x + dx
            let clamped_x = if cx < 0 {
              0
            } else if cx > wmax {
              wmax
            } else {
              cx
            }
            buf[bi] = img.data[row_base + clamped_x * 4 + c]
            bi = bi + 1
          }
        }
        out.data[oo + c] = reduce(buf)
      }
      out.data[oo + 3] = img.data[oo + 3]
    }
  }
  out
}

///|
/// Median filter with an odd window `size`. Reduces salt-and-pepper noise
/// while preserving edges. Uses replicate borders; alpha is copied unchanged.
pub fn median_filter(img : Image, size : Int) -> Image {
  window_map(img, size, fn(vals) {
    vals.sort()
    vals[vals.length() / 2]
  })
}

///|
/// Maximum (dilation-like) filter with an odd window `size`. Each output
/// pixel is the max over a `(2r+1)²` window with replicate borders; alpha
/// is copied unchanged.
pub fn max_filter(img : Image, size : Int) -> Image {
  window_map(img, size, fn(vals) {
    let mut m : Byte = 0
    for i = 0; i < vals.length(); i = i + 1 {
      if vals[i] > m {
        m = vals[i]
      }
    }
    m
  })
}

///|
/// Minimum (erosion-like) filter with an odd window `size`. Each output
/// pixel is the min over a `(2r+1)²` window with replicate borders; alpha
/// is copied unchanged.
pub fn min_filter(img : Image, size : Int) -> Image {
  window_map(img, size, fn(vals) {
    let mut m : Byte = 255
    for i = 0; i < vals.length(); i = i + 1 {
      if vals[i] < m {
        m = vals[i]
      }
    }
    m
  })
}

///|
/// Laplacian-based sharpening. `strength` scales the high-frequency boost;
/// `0.0` returns the input unchanged.
pub fn sharpen(img : Image, strength : Double) -> Image {
  let s = strength
  let kernel = [[0.0, -s, 0.0], [-s, 1.0 + 4.0 * s, -s], [0.0, -s, 0.0]]
  convolve(img, kernel, false)
}

///|
/// Unsharp masking: add a scaled high-pass component above a threshold.
///
/// - `radius`: Gaussian blur radius used to derive the low-pass component.
/// - `amount`: scaling factor applied to the high-pass residual.
/// - `threshold`: only pixels whose `|orig - blurred|` ≥ `threshold` are
///   boosted; smaller differences are passed through unchanged.
pub fn unsharp_mask(
  img : Image,
  radius : Double,
  amount : Double,
  threshold : Byte,
) -> Image {
  let blurred = gaussian_blur(img, radius)
  let t = threshold.to_int()
  let out = Image::new(img.h, img.w)
  let n = img.h * img.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    for c = 0; c < 3; c = c + 1 {
      let orig = img.data[o + c].to_int()
      let diff = orig - blurred.data[o + c].to_int()
      let d = if diff < 0 { -diff } else { diff }
      out.data[o + c] = if d >= t {
        round_byte(orig.to_double() + amount * diff.to_double())
      } else {
        img.data[o + c]
      }
    }
    out.data[o + 3] = img.data[o + 3]
  }
  out
}
