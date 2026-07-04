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
  let noff = offs.length()
  let h = img.h
  let w = img.w
  let stride = w * 4
  let out = Image::new(h, w)
  let hmax = h - 1
  let wmax = w - 1
  // Compute max radius for interior check
  let mut max_r = 0
  for i = 0; i < noff; i = i + 1 {
    let (dy, dx) = offs[i]
    let ady = if dy < 0 { -dy } else { dy }
    let adx = if dx < 0 { -dx } else { dx }
    let m = if ady > adx { ady } else { adx }
    if m > max_r {
      max_r = m
    }
  }
  for y = 0; y < h; y = y + 1 {
    let out_base = y * stride
    let is_interior_y = y >= max_r && y < h - max_r
    for x = 0; x < w; x = x + 1 {
      let oo = out_base + x * 4
      let is_interior = is_interior_y && x >= max_r && x < w - max_r
      for c = 0; c < 3; c = c + 1 {
        let mut m = if is_dilate { 0 } else { 255 }
        if is_interior {
          for i = 0; i < noff; i = i + 1 {
            let (dy, dx) = offs[i]
            let v = img.data[(y + dy) * stride + (x + dx) * 4 + c].to_int()
            if is_dilate {
              if v > m {
                m = v
              }
            } else if v < m {
              m = v
            }
          }
        } else {
          for i = 0; i < noff; i = i + 1 {
            let (dy, dx) = offs[i]
            let cy = y + dy
            let cx = x + dx
            let clamped_y = if cy < 0 {
              0
            } else if cy > hmax {
              hmax
            } else {
              cy
            }
            let clamped_x = if cx < 0 {
              0
            } else if cx > wmax {
              wmax
            } else {
              cx
            }
            let v = img.data[clamped_y * stride + clamped_x * 4 + c].to_int()
            if is_dilate {
              if v > m {
                m = v
              }
            } else if v < m {
              m = v
            }
          }
        }
        out.data[oo + c] = m.to_byte()
      }
      out.data[oo + 3] = img.data[oo + 3]
    }
  }
  out
}

///|
/// Morphological dilation.
///
/// Each output pixel is the maximum of the RGB channel values over the
/// structuring element defined by `kernel`, growing bright regions. Border
/// coordinates are clamped (replicate); alpha is preserved.
pub fn dilate(img : Image, kernel : Kernel) -> Image {
  morph(img, kernel, true)
}

///|
/// Morphological erosion.
///
/// Each output pixel is the minimum of the RGB channel values over the
/// structuring element defined by `kernel`, shrinking bright regions. Border
/// coordinates are clamped (replicate); alpha is preserved.
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
///
/// Removes small bright foreground structures smaller than `kernel` while
/// preserving the shape of larger regions.
pub fn morph_open(img : Image, kernel : Kernel) -> Image {
  dilate(erode(img, kernel), kernel)
}

///|
/// Closing: dilation followed by erosion.
///
/// Fills small dark holes and gaps smaller than `kernel` while preserving the
/// shape of larger regions.
pub fn morph_close(img : Image, kernel : Kernel) -> Image {
  erode(dilate(img, kernel), kernel)
}

///|
/// Morphological gradient: dilation minus erosion.
///
/// Highlights region edges by computing the per-channel difference between the
/// dilated and eroded images.
pub fn morph_gradient(img : Image, kernel : Kernel) -> Image {
  sub_img(dilate(img, kernel), erode(img, kernel))
}

///|
/// Top-hat: original minus opening.
///
/// Isolates small bright structures smaller than `kernel` that are removed by
/// the opening.
pub fn top_hat(img : Image, kernel : Kernel) -> Image {
  sub_img(img, morph_open(img, kernel))
}

///|
/// Black-hat: closing minus original.
///
/// Isolates small dark structures smaller than `kernel` that are filled by the
/// closing.
pub fn black_hat(img : Image, kernel : Kernel) -> Image {
  sub_img(morph_close(img, kernel), img)
}

///|
/// Skeletonize via iterative thinning (Zhang-Suen). Works on binary images
/// (non-zero pixels are foreground).
///
/// Iteratively removes boundary pixels over two sub-iterations until no
/// further pixels can be thinned, yielding a one-pixel-wide medial axis.
/// Border pixels are left untouched.
pub fn skeletonize(img : Image) -> Image {
  let mut cur = img.clone()
  let mut changed = true
  while changed {
    changed = false
    for pass = 0; pass < 2; pass = pass + 1 {
      let next = cur.clone()
      for y = 1; y < img.h - 1; y = y + 1 {
        for x = 1; x < img.w - 1; x = x + 1 {
          let o = cur.offset(y, x)
          if cur.data[o] == 0 {
            continue
          }
          // 8-neighbourhood: P2..P9 clockwise from top
          let p2 = if cur.data[cur.offset(y - 1, x)] != 0 { 1 } else { 0 }
          let p3 = if cur.data[cur.offset(y - 1, x + 1)] != 0 { 1 } else { 0 }
          let p4 = if cur.data[cur.offset(y, x + 1)] != 0 { 1 } else { 0 }
          let p5 = if cur.data[cur.offset(y + 1, x + 1)] != 0 { 1 } else { 0 }
          let p6 = if cur.data[cur.offset(y + 1, x)] != 0 { 1 } else { 0 }
          let p7 = if cur.data[cur.offset(y + 1, x - 1)] != 0 { 1 } else { 0 }
          let p8 = if cur.data[cur.offset(y, x - 1)] != 0 { 1 } else { 0 }
          let p9 = if cur.data[cur.offset(y - 1, x - 1)] != 0 { 1 } else { 0 }
          let b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
          if b < 2 || b > 6 {
            continue
          }
          // Transitions 0→1 in ordered sequence
          let seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
          let mut a = 0
          for i = 0; i < 8; i = i + 1 {
            if seq[i] == 0 && seq[i + 1] == 1 {
              a = a + 1
            }
          }
          if a != 1 {
            continue
          }
          let cond = if pass == 0 {
            p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0
          } else {
            p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0
          }
          if cond {
            let oo = next.offset(y, x)
            next.data[oo] = 0
            next.data[oo + 1] = 0
            next.data[oo + 2] = 0
            changed = true
          }
        }
      }
      cur = next
    }
  }
  cur
}

///|
/// Hit-or-miss transform. The kernel values are: 1 = foreground, 0 = background,
/// -1 = don't care. Only pixels matching the pattern are kept.
///
/// Treats non-zero RGB pixels as foreground. A pixel is set to white when its
/// neighbourhood matches `kernel` exactly (foreground/background), and black
/// otherwise. Output alpha is set to 255.
pub fn hit_or_miss(img : Image, kernel : Array[Array[Int]]) -> Image {
  let kh = kernel.length()
  let kw = kernel[0].length()
  let cr = kh / 2
  let cc = kw / 2
  let out = Image::new(img.h, img.w)
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let mut match_ = true
      for ky = 0; ky < kh && match_; ky = ky + 1 {
        for kx = 0; kx < kw && match_; kx = kx + 1 {
          let k = kernel[ky][kx]
          if k == -1 {
            continue
          }
          let cy = y + ky - cr
          let cx = x + kx - cc
          let pixel_fg = if cy < 0 || cx < 0 || cy >= img.h || cx >= img.w {
            false
          } else {
            img.data[img.offset(cy, cx)] != 0
          }
          if k == 1 && !pixel_fg {
            match_ = false
          }
          if k == 0 && pixel_fg {
            match_ = false
          }
        }
      }
      let o = out.offset(y, x)
      let v : Byte = if match_ { 255 } else { 0 }
      out.data[o] = v
      out.data[o + 1] = v
      out.data[o + 2] = v
      out.data[o + 3] = 255
    }
  }
  out
}
