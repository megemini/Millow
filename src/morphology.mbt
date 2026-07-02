///|
/// Expand a structuring element into a list of `(dy, dx)` offsets.
fn kernel_offsets(k : Kernel) -> Array[(Int, Int)] {
  let offs : Array[(Int, Int)] = []
  match k {
    Cross(size) => {
      let r = size / 2
      for d = -r; d <= r; d = d + 1 {
        offs.push((d, 0))
        if d != 0 {
          offs.push((0, d))
        }
      }
    }
    Square(size) => {
      let r = size / 2
      for dy = -r; dy <= r; dy = dy + 1 {
        for dx = -r; dx <= r; dx = dx + 1 {
          offs.push((dy, dx))
        }
      }
    }
    Custom(mask) => {
      let cr = mask.length() / 2
      for i = 0; i < mask.length(); i = i + 1 {
        let cc = mask[i].length() / 2
        for j = 0; j < mask[i].length(); j = j + 1 {
          if mask[i][j] {
            offs.push((i - cr, j - cc))
          }
        }
      }
    }
  }
  offs
}

///|
/// Grayscale morphology core: max (dilate) or min (erode) over the element.
fn morph(img : Image, k : Kernel, is_dilate : Bool) -> Image {
  let offs = kernel_offsets(k)
  let out = Image::new(img.h, img.w)
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let oo = out.offset(y, x)
      for c = 0; c < 3; c = c + 1 {
        let mut m = if is_dilate { 0 } else { 255 }
        for off in offs {
          let (dy, dx) = off
          let cy = clampi(y + dy, 0, img.h - 1)
          let cx = clampi(x + dx, 0, img.w - 1)
          let v = img.data[img.offset(cy, cx) + c].to_int()
          if is_dilate {
            if v > m {
              m = v
            }
          } else if v < m {
            m = v
          }
        }
        out.data[oo + c] = m.to_byte()
      }
      out.data[oo + 3] = img.data[img.offset(y, x) + 3]
    }
  }
  out
}

///|
/// Morphological dilation.
pub fn dilate(img : Image, kernel : Kernel) -> Image {
  morph(img, kernel, true)
}

///|
/// Morphological erosion.
pub fn erode(img : Image, kernel : Kernel) -> Image {
  morph(img, kernel, false)
}

///|
/// Per-channel subtraction `a - b` (clamped), alpha taken from `a`.
fn sub_img(a : Image, b : Image) -> Image {
  let out = Image::new(a.h, a.w)
  let n = a.h * a.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    for c = 0; c < 3; c = c + 1 {
      out.data[o + c] = clamp_byte(
        a.data[o + c].to_int() - b.data[o + c].to_int(),
      )
    }
    out.data[o + 3] = a.data[o + 3]
  }
  out
}

///|
/// Opening: erosion followed by dilation.
pub fn morph_open(img : Image, kernel : Kernel) -> Image {
  dilate(erode(img, kernel), kernel)
}

///|
/// Closing: dilation followed by erosion.
pub fn morph_close(img : Image, kernel : Kernel) -> Image {
  erode(dilate(img, kernel), kernel)
}

///|
/// Morphological gradient: dilation minus erosion.
pub fn morph_gradient(img : Image, kernel : Kernel) -> Image {
  sub_img(dilate(img, kernel), erode(img, kernel))
}

///|
/// Top-hat: original minus opening.
pub fn top_hat(img : Image, kernel : Kernel) -> Image {
  sub_img(img, morph_open(img, kernel))
}

///|
/// Black-hat: closing minus original.
pub fn black_hat(img : Image, kernel : Kernel) -> Image {
  sub_img(morph_close(img, kernel), img)
}
