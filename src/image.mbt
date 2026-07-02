///|
/// Error raised by image operations that receive invalid arguments.
pub suberror ImageError {
  ImageError(String)
}

///|
pub impl Show for ImageError with fn output(self, logger) {
  let ImageError(m) = self
  logger.write_string("ImageError(\{m})")
}

///|
/// Core image type. Internal storage is RGBA8, row-major, `h * w * 4` bytes.
/// The byte at offset `(y * w + x) * 4` is R, then G, B, A.
pub(all) struct Image {
  data : Array[Byte]
  h : Int
  w : Int
}

///|
/// Create an all-zero (transparent black) image of the given size.
pub fn Image::new(h : Int, w : Int) -> Image {
  { data: Array::make(h * w * 4, (0 : Byte)), h, w }
}

///|
/// Create a solid-color image.
pub fn Image::from_pixel(
  h : Int,
  w : Int,
  r : Byte,
  g : Byte,
  b : Byte,
  a : Byte,
) -> Image {
  let data = Array::makei(h * w * 4, fn(i) {
    match i % 4 {
      0 => r
      1 => g
      2 => b
      _ => a
    }
  })
  { data, h, w }
}

///|
/// Build an image from raw RGBA8 data. Fails if the length does not match.
pub fn Image::from_data(
  data : Array[Byte],
  h : Int,
  w : Int,
) -> Image raise ImageError {
  if data.length() != h * w * 4 {
    raise ImageError("from_data: length mismatch")
  }
  { data, h, w }
}

///|
/// Deep copy of the image buffer.
pub fn Image::clone(self : Image) -> Image {
  { data: self.data.copy(), h: self.h, w: self.w }
}

///|
/// Image height in pixels.
pub fn Image::height(self : Image) -> Int {
  self.h
}

///|
/// Image width in pixels.
pub fn Image::width(self : Image) -> Int {
  self.w
}

///|
/// `(height, width)` pair.
pub fn Image::shape(self : Image) -> (Int, Int) {
  (self.h, self.w)
}

///|
/// Length of the underlying byte buffer (`h * w * 4`).
pub fn Image::len(self : Image) -> Int {
  self.data.length()
}

///|
/// Whether the image has no pixels.
pub fn Image::is_empty(self : Image) -> Bool {
  self.h == 0 || self.w == 0
}

///|
/// Flat byte offset of pixel `(y, x)`.
pub fn Image::offset(self : Image, y : Int, x : Int) -> Int {
  (y * self.w + x) * 4
}

///|
/// Read pixel `(y, x)`; returns `None` when out of bounds.
pub fn Image::pixel_at(
  self : Image,
  y : Int,
  x : Int,
) -> (Byte, Byte, Byte, Byte)? {
  if y < 0 || x < 0 || y >= self.h || x >= self.w {
    None
  } else {
    let o = self.offset(y, x)
    Some((self.data[o], self.data[o + 1], self.data[o + 2], self.data[o + 3]))
  }
}

///|
/// Write pixel `(y, x)`; fails when out of bounds.
pub fn Image::pixel_set(
  self : Image,
  y : Int,
  x : Int,
  r : Byte,
  g : Byte,
  b : Byte,
  a : Byte,
) -> Unit raise ImageError {
  if y < 0 || x < 0 || y >= self.h || x >= self.w {
    raise ImageError("pixel_set: out of bounds")
  }
  let o = self.offset(y, x)
  self.data[o] = r
  self.data[o + 1] = g
  self.data[o + 2] = b
  self.data[o + 3] = a
}

///|
/// Iterate over every pixel with its coordinates and RGBA values.
pub fn Image::for_each_pixel(
  self : Image,
  f : (Int, Int, Byte, Byte, Byte, Byte) -> Unit,
) -> Unit {
  for y = 0; y < self.h; y = y + 1 {
    for x = 0; x < self.w; x = x + 1 {
      let o = self.offset(y, x)
      f(
        y,
        x,
        self.data[o],
        self.data[o + 1],
        self.data[o + 2],
        self.data[o + 3],
      )
    }
  }
}
