///|
/// Interpolation method used by geometric resampling operations.
///
/// - `Nearest`: pick the closest source pixel.
/// - `Bilinear`: weighted average of the 4 nearest pixels.
/// - `Bicubic`: cubic interpolation over a 4x4 neighborhood.
pub(all) enum Interp {
  Nearest
  Bilinear
  Bicubic
} derive(Eq)

///|
/// Border handling strategy used by padding and neighborhood operations.
///
/// - `Constant(r, g, b, a)`: fill outside pixels with the given RGBA color.
/// - `Replicate`: copy the nearest edge pixel.
/// - `Reflect`: mirror with edge duplication (the boundary pixel is repeated).
/// - `Wrap`: tile the image periodically.
pub(all) enum PadMode {
  Constant(Byte, Byte, Byte, Byte)
  Replicate
  Reflect
  Wrap
} derive(Eq)

///|
/// Structuring element for morphological operations.
///
/// - `Cross(n)`: plus-shaped mask with radius `n` (arm length `n`).
/// - `Square(n)`: full `(2n+1) x (2n+1)` box.
/// - `Custom(mask)`: arbitrary boolean mask given as rows of booleans.
pub(all) enum Kernel {
  Cross(Int)
  Square(Int)
  Custom(Array[Array[Bool]])
} derive(Eq)

///|
/// Clamp an integer into the inclusive range `[lo, hi]`.
pub fn clampi(v : Int, lo : Int, hi : Int) -> Int {
  if v < lo {
    lo
  } else if v > hi {
    hi
  } else {
    v
  }
}

///|
/// Convert an integer to a byte, clamping into `[0, 255]`.
pub fn clamp_byte(v : Int) -> Byte {
  clampi(v, 0, 255).to_byte()
}

///|
/// Clamp a double into the inclusive range `[lo, hi]`.
pub fn clampd(v : Double, lo : Double, hi : Double) -> Double {
  if v < lo {
    lo
  } else if v > hi {
    hi
  } else {
    v
  }
}

///|
/// Round a double and clamp into a byte range `[0, 255]`.
pub fn round_byte(v : Double) -> Byte {
  clamp_byte(@math.round(v).to_int())
}
