///|
/// Copy the 4 RGBA bytes at `so` in `src` to `do` in `dst`.
fn copy_px(src : Image, so : Int, dst : Image, dof : Int) -> Unit {
  dst.data[dof] = src.data[so]
  dst.data[dof + 1] = src.data[so + 1]
  dst.data[dof + 2] = src.data[so + 2]
  dst.data[dof + 3] = src.data[so + 3]
}

///|
/// Crop a rectangular region starting at `(y, x)` with size `h x w`.
///
/// - `y`, `x`: top-left corner of the region (inclusive).
/// - `h`, `w`: height and width of the region.
///
/// Raises `ImageError` if the region lies outside the image bounds.
pub fn crop(
  img : Image,
  y : Int,
  x : Int,
  h : Int,
  w : Int,
) -> Image raise ImageError {
  if y < 0 || x < 0 || h < 0 || w < 0 || y + h > img.h || x + w > img.w {
    raise ImageError("crop: region out of bounds")
  }
  let out = Image::new(h, w)
  for yy = 0; yy < h; yy = yy + 1 {
    for xx = 0; xx < w; xx = xx + 1 {
      copy_px(img, img.offset(y + yy, x + xx), out, out.offset(yy, xx))
    }
  }
  out
}

///|
/// Alias of `crop` (this library stores contiguous buffers, so no true view).
pub fn Image::sub_image(
  self : Image,
  y : Int,
  x : Int,
  h : Int,
  w : Int,
) -> Image raise ImageError {
  crop(self, y, x, h, w)
}

///|
/// Mirror horizontally (left-right).
pub fn flip_horizontal(img : Image) -> Image {
  let out = Image::new(img.h, img.w)
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      copy_px(img, img.offset(y, x), out, out.offset(y, img.w - 1 - x))
    }
  }
  out
}

///|
/// Mirror vertically (top-bottom).
pub fn flip_vertical(img : Image) -> Image {
  let out = Image::new(img.h, img.w)
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      copy_px(img, img.offset(y, x), out, out.offset(img.h - 1 - y, x))
    }
  }
  out
}

///|
/// Rotate 90 degrees clockwise.
pub fn rotate_90(img : Image) -> Image {
  let out = Image::new(img.w, img.h)
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      copy_px(img, img.offset(y, x), out, out.offset(x, img.h - 1 - y))
    }
  }
  out
}

///|
/// Rotate 180 degrees.
pub fn rotate_180(img : Image) -> Image {
  let out = Image::new(img.h, img.w)
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      copy_px(
        img,
        img.offset(y, x),
        out,
        out.offset(img.h - 1 - y, img.w - 1 - x),
      )
    }
  }
  out
}

///|
/// Rotate 270 degrees clockwise (90 counter-clockwise).
pub fn rotate_270(img : Image) -> Image {
  let out = Image::new(img.w, img.h)
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      copy_px(img, img.offset(y, x), out, out.offset(img.w - 1 - x, y))
    }
  }
  out
}

///|
/// Remove blank borders by finding the content bounding box.
///
/// Scans the image and finds the tightest bounding box containing pixels with
/// luma below `threshold` (i.e. dark/content pixels). Returns the cropped
/// sub-image.
///
/// - `threshold`: luma value at or above which a pixel is considered blank
///   (default `240`, suitable for white-background documents). For binary
///   images (foreground = 255, background = 0), set `threshold` to `128` or
///   lower.
///
/// Raises `ImageError` if no content is found.
pub fn auto_crop(
  img : Image,
  threshold? : Byte = 240,
) -> Image raise ImageError {
  let th = threshold.to_int()
  let h = img.h
  let w = img.w
  let mut y_min = h
  let mut y_max = 0
  let mut x_min = w
  let mut x_max = 0
  let mut found = false
  for y = 0; y < h; y = y + 1 {
    let row = y * w * 4
    for x = 0; x < w; x = x + 1 {
      let o = row + x * 4
      let l = luma(img.data[o], img.data[o + 1], img.data[o + 2])
      if l < th {
        found = true
        if y < y_min {
          y_min = y
        }
        if y > y_max {
          y_max = y
        }
        if x < x_min {
          x_min = x
        }
        if x > x_max {
          x_max = x
        }
      }
    }
  }
  if !found {
    raise ImageError("auto_crop: no content found")
  }
  crop(img, y_min, x_min, y_max - y_min + 1, x_max - x_min + 1)
}
