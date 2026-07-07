///|
/// Two-pass chamfer distance transform.
///
/// Computes the distance from every foreground pixel (luma > 0) to the nearest
/// background pixel (luma == 0) using a 3-4 chamfer kernel (3 for axial moves,
/// 4 for diagonal moves), then rescales so the result approximates the
/// Euclidean distance (factor `3 / (3 + 4) ≈ 0.8619` would be exact only for
/// the 3-4 weights; here we divide by 3 so axial-unit moves yield 1.0).
///
/// - `foreground_value`: luma threshold; pixels with `luma >= foreground_value`
///   are treated as foreground (object). Defaults to 128.
///
/// Returns a 2D `Array[Array[Double]]` of shape `(h, w)`. Foreground pixels
/// have positive distances; background pixels are 0.
///
/// Two-pass implementation (Felzenszwalb & Huttenlocher style, but with the
/// simpler chamfer kernel): forward pass (top-left to bottom-right) and
/// backward pass (bottom-right to top-left). O(n) complexity.
pub fn distance_transform(
  img : Image,
  foreground_value? : Int = 128,
) -> Array[Array[Double]] {
  let h = img.h
  let w = img.w
  let stride = w * 4
  // Build a boolean mask: true = foreground (large distance), false = background (0).
  let fg = Array::make(h * w, false)
  for y = 0; y < h; y = y + 1 {
    let row_base = y * stride
    for x = 0; x < w; x = x + 1 {
      let o = row_base + x * 4
      let l = (
          img.data[o].to_int() * 77 +
          img.data[o + 1].to_int() * 150 +
          img.data[o + 2].to_int() * 29
        ) >>
        8
      fg[y * w + x] = l >= foreground_value
    }
  }
  // Initialize: foreground = large, background = 0.
  let large = (h + w).to_double() * 3.0
  let dist = Array::make(h * w, 0.0)
  for i = 0; i < h * w; i = i + 1 {
    dist[i] = if fg[i] { large } else { 0.0 }
  }
  // Forward pass: top-left to bottom-right.
  // Neighbors considered: (y-1, x), (y, x-1), (y-1, x-1), (y-1, x+1).
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      let idx = y * w + x
      if !fg[idx] {
        continue
      }
      let mut d = dist[idx]
      if y > 0 {
        let up = dist[(y - 1) * w + x] + 3.0
        if up < d {
          d = up
        }
        if x > 0 {
          let ul = dist[(y - 1) * w + (x - 1)] + 4.0
          if ul < d {
            d = ul
          }
        }
        if x < w - 1 {
          let ur = dist[(y - 1) * w + (x + 1)] + 4.0
          if ur < d {
            d = ur
          }
        }
      }
      if x > 0 {
        let left = dist[y * w + (x - 1)] + 3.0
        if left < d {
          d = left
        }
      }
      dist[idx] = d
    }
  }
  // Backward pass: bottom-right to top-left.
  // Neighbors considered: (y+1, x), (y, x+1), (y+1, x+1), (y+1, x-1).
  for y = h - 1; y >= 0; y = y - 1 {
    for x = w - 1; x >= 0; x = x - 1 {
      let idx = y * w + x
      if !fg[idx] {
        continue
      }
      let mut d = dist[idx]
      if y < h - 1 {
        let down = dist[(y + 1) * w + x] + 3.0
        if down < d {
          d = down
        }
        if x < w - 1 {
          let dr = dist[(y + 1) * w + (x + 1)] + 4.0
          if dr < d {
            d = dr
          }
        }
        if x > 0 {
          let dl = dist[(y + 1) * w + (x - 1)] + 4.0
          if dl < d {
            d = dl
          }
        }
      }
      if x < w - 1 {
        let right = dist[y * w + (x + 1)] + 3.0
        if right < d {
          d = right
        }
      }
      dist[idx] = d
    }
  }
  // Rescale chamfer (3-4) to approximate Euclidean: divide by 3.
  // The 3-4 chamfer is a known approximation; factor 1/3 normalizes axial moves
  // to unit distance.
  let result = Array::makei(h, fn(_y) { Array::make(w, 0.0) })
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      result[y][x] = dist[y * w + x] / 3.0
    }
  }
  result
}
