///|
/// Collect the values of channel `c` in a `(2r+1)` window (clamped border).
fn window_vals(img : Image, y : Int, x : Int, r : Int, c : Int) -> Array[Byte] {
  let vals : Array[Byte] = []
  for dy = -r; dy <= r; dy = dy + 1 {
    for dx = -r; dx <= r; dx = dx + 1 {
      let cy = clampi(y + dy, 0, img.h - 1)
      let cx = clampi(x + dx, 0, img.w - 1)
      vals.push(img.data[img.offset(cy, cx) + c])
    }
  }
  vals
}

///|
/// Apply a per-channel window reducer over the RGB channels (alpha copied).
fn window_map(img : Image, size : Int, reduce : (Array[Byte]) -> Byte) -> Image {
  let r = size / 2
  let out = Image::new(img.h, img.w)
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let oo = out.offset(y, x)
      for c = 0; c < 3; c = c + 1 {
        out.data[oo + c] = reduce(window_vals(img, y, x, r, c))
      }
      out.data[oo + 3] = img.data[img.offset(y, x) + 3]
    }
  }
  out
}

///|
/// Median filter with an odd window `size`.
pub fn median_filter(img : Image, size : Int) -> Image {
  window_map(img, size, fn(vals) {
    vals.sort()
    vals[vals.length() / 2]
  })
}

///|
/// Maximum (dilation-like) filter with an odd window `size`.
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
/// Minimum (erosion-like) filter with an odd window `size`.
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
/// Laplacian-based sharpening. `strength` scales the high-frequency boost.
pub fn sharpen(img : Image, strength : Double) -> Image {
  let s = strength
  let kernel = [[0.0, -s, 0.0], [-s, 1.0 + 4.0 * s, -s], [0.0, -s, 0.0]]
  convolve(img, kernel, false)
}

///|
/// Unsharp masking: add a scaled high-pass component above a threshold.
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
