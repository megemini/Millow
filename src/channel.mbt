///|
/// Extract a single channel (0=R, 1=G, 2=B, 3=A) as an `h * w` byte array.
pub fn Image::channel(self : Image, ch : Int) -> Array[Byte] {
  Array::makei(self.h * self.w, fn(i) { self.data[i * 4 + ch] })
}

///|
/// Red channel.
pub fn Image::channel_r(self : Image) -> Array[Byte] {
  self.channel(0)
}

///|
/// Green channel.
pub fn Image::channel_g(self : Image) -> Array[Byte] {
  self.channel(1)
}

///|
/// Blue channel.
pub fn Image::channel_b(self : Image) -> Array[Byte] {
  self.channel(2)
}

///|
/// Alpha channel.
pub fn Image::channel_a(self : Image) -> Array[Byte] {
  self.channel(3)
}

///|
/// Return a copy of the image with one channel replaced.
pub fn Image::set_channel(self : Image, ch : Int, data : Array[Byte]) -> Image {
  let out = self.data.copy()
  let n = self.h * self.w
  for i = 0; i < n; i = i + 1 {
    out[i * 4 + ch] = data[i]
  }
  { data: out, h: self.h, w: self.w }
}

///|
/// Split into four channel arrays `(r, g, b, a)`.
pub fn Image::split_channels(
  self : Image,
) -> (Array[Byte], Array[Byte], Array[Byte], Array[Byte]) {
  (self.channel(0), self.channel(1), self.channel(2), self.channel(3))
}

///|
/// Merge channel arrays into an image. Alpha defaults to fully opaque.
pub fn merge_channels(
  r : Array[Byte],
  g : Array[Byte],
  b : Array[Byte],
  a : Array[Byte]?,
) -> Image raise ImageError {
  let n = r.length()
  if g.length() != n || b.length() != n {
    raise ImageError("merge_channels: channel length mismatch")
  }
  match a {
    Some(av) =>
      if av.length() != n {
        raise ImageError("merge_channels: alpha length mismatch")
      }
    None => ()
  }
  let data = Array::make(n * 4, (0 : Byte))
  for i = 0; i < n; i = i + 1 {
    data[i * 4] = r[i]
    data[i * 4 + 1] = g[i]
    data[i * 4 + 2] = b[i]
    data[i * 4 + 3] = match a {
      Some(av) => av[i]
      None => 255
    }
  }
  { data, h: 1, w: n }
}
