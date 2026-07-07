///|
/// Detect the skew angle of a document image in degrees.
///
/// Uses the projection-profile method: binarizes the image (Otsu), rotates it
/// at fine increments within `[-max_angle, +max_angle]`, and returns the angle
/// that maximizes the variance of the horizontal projection. A well-aligned
/// text image has high projection variance because text rows produce sharp
/// peaks and gaps.
///
/// - `max_angle`: maximum absolute skew to search (default `15.0` degrees).
///
/// Returns the detected angle in degrees. Positive means the image is skewed
/// clockwise and should be counter-rotated by that amount to deskew.
///
/// Uses downscaling to 512 px width and a two-stage coarse (0.5°) then fine
/// (0.1°) search to keep computation practical.
pub fn skew_angle(img : Image, max_angle? : Double = 15.0) -> Double {
  // 1. Downscale to ≤512 px width for speed.
  let target_w = 512
  let small = if img.w > target_w {
    let scale = target_w.to_double() / img.w.to_double()
    let dst_h = (img.h.to_double() * scale).to_int().max(1)
    resize(img, dst_h, target_w, Nearest)
  } else {
    img
  }

  // 2. Binarize with Otsu (skip if the image looks already binary).
  let binary = if count_nonzero(small) < small.h * small.w / 2 {
    small
  } else {
    let gray = to_grayscale(small)
    let (_, b) = threshold_otsu(gray)
    b
  }

  // 3. Coarse search: ±max_angle, step 0.5°.
  let mut best_angle = 0.0
  let mut best_var = -1.0
  let coarse_step = 0.5
  let mut angle = -max_angle
  while angle <= max_angle {
    let v = projection_variance(binary, angle)
    if v > best_var {
      best_var = v
      best_angle = angle
    }
    angle = angle + coarse_step
  }

  // 4. Fine search: best_angle ± 0.5°, step 0.1°.
  let fine_lo = best_angle - coarse_step
  let fine_hi = best_angle + coarse_step
  let fine_step = 0.1
  angle = fine_lo
  while angle <= fine_hi {
    let v = projection_variance(binary, angle)
    if v > best_var {
      best_var = v
      best_angle = angle
    }
    angle = angle + fine_step
  }

  best_angle
}

///|
/// Automatically detect and correct document skew.
///
/// Calls `skew_angle` to find the skew, then rotates the **original** image
/// (full resolution) by the negative of that angle using bilinear
/// interpolation. Areas uncovered by the rotation become opaque black.
///
/// - `max_angle`: maximum absolute skew to search (default `15.0` degrees).
pub fn deskew(img : Image, max_angle? : Double = 15.0) -> Image {
  let angle = skew_angle(img, max_angle~)
  rotate_any(img, -angle, Bilinear)
}

///|
/// Compute the variance of the binary horizontal projection after rotating
/// by `angle` degrees. Higher variance → better text-row alignment.
fn projection_variance(binary : Image, angle : Double) -> Double {
  let rotated = rotate_any(binary, angle, Nearest)
  let proj = project_horizontal_binary(rotated)
  let n = proj.length().to_double()
  if n == 0.0 {
    return 0.0
  }
  let mut sum = 0.0
  let mut sum2 = 0.0
  for v in proj {
    let d = v.to_double()
    sum = sum + d
    sum2 = sum2 + d * d
  }
  let mean = sum / n
  sum2 / n - mean * mean
}
