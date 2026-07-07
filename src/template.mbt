///|
/// Template matching via normalized cross-correlation (NCC).
///
/// Slides `template` over `img` and computes the NCC score at each valid
/// location `(y, x)` where the template fits entirely within the image.
///
/// The score at each location is:
/// ```
/// Σ((I - mean_I) * (T - mean_T)) / sqrt(Σ(I - mean_I)² * Σ(T - mean_T)²)
/// ```
///
/// computed on luma values, where `I` is the image patch under the template
/// and `T` is the template. The result is in `[-1, 1]`; `1` means a perfect
/// match. Returns a 2D `Array[Array[Double]]` of shape
/// `(img.h - tpl.h + 1, img.w - tpl.w + 1)`. If the template is larger than the
/// image in either dimension, returns an empty array.
///
/// Border handling: only valid (fully-overlapping) locations are scored.
pub fn template_match(img : Image, template : Image) -> Array[Array[Double]] {
  let ih = img.h
  let iw = img.w
  let th = template.h
  let tw = template.w
  let istride = iw * 4
  let tstride = tw * 4
  if th > ih || tw > iw || th == 0 || tw == 0 {
    return []
  }
  let out_h = ih - th + 1
  let out_w = iw - tw + 1
  // Precompute template luma and its mean and norm.
  let tgray = Array::make(th * tw, 0.0)
  let mut tsum = 0.0
  for y = 0; y < th; y = y + 1 {
    for x = 0; x < tw; x = x + 1 {
      let o = y * tstride + x * 4
      let l = ((
        template.data[o].to_int() * 77 +
        template.data[o + 1].to_int() * 150 +
        template.data[o + 2].to_int() * 29
      ) >>
      8).to_double()
      tgray[y * tw + x] = l
      tsum = tsum + l
    }
  }
  let tmean = tsum / (th * tw).to_double()
  let mut tnorm_sq = 0.0
  for i = 0; i < th * tw; i = i + 1 {
    let d = tgray[i] - tmean
    tnorm_sq = tnorm_sq + d * d
  }
  let result = Array::makei(out_h, fn(_y) { Array::make(out_w, 0.0) })
  // If template is constant, NCC is undefined; output zeros.
  if tnorm_sq == 0.0 {
    return result
  }
  // Precompute image luma.
  let igray = Array::make(ih * iw, 0.0)
  for y = 0; y < ih; y = y + 1 {
    for x = 0; x < iw; x = x + 1 {
      let o = y * istride + x * 4
      igray[y * iw + x] = ((
        img.data[o].to_int() * 77 +
        img.data[o + 1].to_int() * 150 +
        img.data[o + 2].to_int() * 29
      ) >>
      8).to_double()
    }
  }
  for oy = 0; oy < out_h; oy = oy + 1 {
    for ox = 0; ox < out_w; ox = ox + 1 {
      // Compute patch mean and cross-correlation in one pass.
      let mut isum = 0.0
      for ty = 0; ty < th; ty = ty + 1 {
        let irow = (oy + ty) * iw + ox
        for tx = 0; tx < tw; tx = tx + 1 {
          isum = isum + igray[irow + tx]
        }
      }
      let imean = isum / (th * tw).to_double()
      let mut cross = 0.0
      let mut inorm_sq = 0.0
      for ty = 0; ty < th; ty = ty + 1 {
        let irow = (oy + ty) * iw + ox
        let trow = ty * tw
        for tx = 0; tx < tw; tx = tx + 1 {
          let id = igray[irow + tx] - imean
          let td = tgray[trow + tx] - tmean
          cross = cross + id * td
          inorm_sq = inorm_sq + id * id
        }
      }
      let denom = (inorm_sq * tnorm_sq).sqrt()
      result[oy][ox] = if denom > 0.0 { cross / denom } else { 0.0 }
    }
  }
  result
}

///|
/// Find the best template match location.
///
/// Runs `template_match` and returns `(best_y, best_x, score)` for the location
/// with the highest NCC score. Returns `(-1, -1, 0.0)` if the template is
/// larger than the image.
pub fn template_match_best(img : Image, template : Image) -> (Int, Int, Double) {
  let scores = template_match(img, template)
  if scores.length() == 0 || scores[0].length() == 0 {
    return (-1, -1, 0.0)
  }
  let mut best_y = 0
  let mut best_x = 0
  let mut best_s = scores[0][0]
  for y = 0; y < scores.length(); y = y + 1 {
    for x = 0; x < scores[y].length(); x = x + 1 {
      if scores[y][x] > best_s {
        best_s = scores[y][x]
        best_y = y
        best_x = x
      }
    }
  }
  (best_y, best_x, best_s)
}
