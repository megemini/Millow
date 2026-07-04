///|
/// Apply per-channel lookup tables (each 256 entries). When `a_lut` is `None`,
/// the alpha channel is copied unchanged.
///
/// - `r_lut`, `g_lut`, `b_lut`: 256-entry tables indexed by the source byte.
/// - `a_lut`: optional 256-entry table for the alpha channel.
pub fn apply_lut(
  img : Image,
  r_lut : Array[Byte],
  g_lut : Array[Byte],
  b_lut : Array[Byte],
  a_lut : Array[Byte]?,
) -> Image {
  map_rgba(img, fn(r, g, b, a) {
    (
      r_lut[r.to_int()],
      g_lut[g.to_int()],
      b_lut[b.to_int()],
      match a_lut {
        Some(lut) => lut[a.to_int()]
        None => a
      },
    )
  })
}

///|
/// Apply a single 256-entry LUT to the RGB channels uniformly; alpha is
/// copied unchanged.
///
/// - `lut`: 256-entry table indexed by the source byte.
pub fn apply_lut_uniform(img : Image, lut : Array[Byte]) -> Image {
  map_rgba(img, fn(r, g, b, a) {
    (lut[r.to_int()], lut[g.to_int()], lut[b.to_int()], a)
  })
}
