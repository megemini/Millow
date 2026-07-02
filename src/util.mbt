///|
/// Append the ASCII bytes of `s` to `out`.
fn push_ascii(out : Array[Byte], s : String) -> Unit {
  for c in s.iter() {
    out.push(c.to_int().to_byte())
  }
}

///|
/// Encode as a binary PPM (P6) byte stream.
pub fn to_ppm(img : Image) -> Array[Byte] {
  let out : Array[Byte] = []
  push_ascii(out, "P6\n\{img.w} \{img.h}\n255\n")
  let n = img.h * img.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    out.push(img.data[o])
    out.push(img.data[o + 1])
    out.push(img.data[o + 2])
  }
  out
}

///|
/// Encode as a binary PGM (P5) grayscale byte stream.
pub fn to_pgm(img : Image) -> Array[Byte] {
  let out : Array[Byte] = []
  push_ascii(out, "P5\n\{img.w} \{img.h}\n255\n")
  let n = img.h * img.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    out.push(luma(img.data[o], img.data[o + 1], img.data[o + 2]).to_byte())
  }
  out
}

///|
/// RGB-only byte buffer (alpha dropped), length `h * w * 3`.
pub fn to_display_bytes(img : Image) -> Array[Byte] {
  let n = img.h * img.w
  let out = Array::make(n * 3, (0 : Byte))
  for i = 0; i < n; i = i + 1 {
    out[i * 3] = img.data[i * 4]
    out[i * 3 + 1] = img.data[i * 4 + 1]
    out[i * 3 + 2] = img.data[i * 4 + 2]
  }
  out
}

///|
/// Human-readable summary of the image.
pub fn info(img : Image) -> String {
  "Image \{img.w}x\{img.h}, \{img.len()} bytes"
}

///|
/// Pixel-exact equality (same dimensions and bytes).
pub fn equals(a : Image, b : Image) -> Bool {
  if a.h != b.h || a.w != b.w {
    return false
  }
  for i = 0; i < a.data.length(); i = i + 1 {
    if a.data[i] != b.data[i] {
      return false
    }
  }
  true
}

///|
/// Mean squared error over all RGBA channels; `-1.0` on size mismatch.
pub fn mse(a : Image, b : Image) -> Double {
  if a.h != b.h || a.w != b.w || a.is_empty() {
    return -1.0
  }
  let mut acc = 0.0
  let len = a.data.length()
  for i = 0; i < len; i = i + 1 {
    let d = a.data[i].to_double() - b.data[i].to_double()
    acc = acc + d * d
  }
  acc / len.to_double()
}

///|
/// Peak signal-to-noise ratio in dB. Returns a large value for identical input.
pub fn psnr(a : Image, b : Image) -> Double {
  let m = mse(a, b)
  if m <= 0.0 {
    return 99.0
  }
  10.0 * @math.log10(255.0 * 255.0 / m)
}

///|
/// Downscale to fit within `max_h x max_w` (nearest-neighbor).
pub fn thumbnail(img : Image, max_h : Int, max_w : Int) -> Image {
  resize_to_fit(img, max_h, max_w, Nearest)
}
