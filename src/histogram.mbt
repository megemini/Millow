///|
/// BT.601 luminance of an RGB triple as an integer in `[0, 255]`.
fn luma(r : Byte, g : Byte, b : Byte) -> Int {
  clampi((r.to_int() * 77 + g.to_int() * 150 + b.to_int() * 29) >> 8, 0, 255)
}

///|
/// 256-bin luminance histogram.
pub fn histogram(img : Image) -> Array[Int] {
  let h = Array::make(256, 0)
  let n = img.h * img.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    let v = luma(img.data[o], img.data[o + 1], img.data[o + 2])
    h[v] = h[v] + 1
  }
  h
}

///|
/// Per-channel histograms for R, G, B (each 256 bins).
pub fn histogram_color(img : Image) -> Array[Array[Int]] {
  let hr = Array::make(256, 0)
  let hg = Array::make(256, 0)
  let hb = Array::make(256, 0)
  let n = img.h * img.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    let r = img.data[o].to_int()
    let g = img.data[o + 1].to_int()
    let b = img.data[o + 2].to_int()
    hr[r] = hr[r] + 1
    hg[g] = hg[g] + 1
    hb[b] = hb[b] + 1
  }
  [hr, hg, hb]
}

///|
/// Histogram equalization applied to each RGB channel independently.
pub fn equalize_histogram(img : Image) -> Image {
  let out = Image::new(img.h, img.w)
  let total = img.h * img.w
  if total == 0 {
    return img.clone()
  }
  for c = 0; c < 3; c = c + 1 {
    let hist = Array::make(256, 0)
    for i = 0; i < total; i = i + 1 {
      let v = img.data[i * 4 + c].to_int()
      hist[v] = hist[v] + 1
    }
    let cdf = Array::make(256, 0)
    let mut acc = 0
    for i = 0; i < 256; i = i + 1 {
      acc = acc + hist[i]
      cdf[i] = acc
    }
    let mut cdf_min_idx = 0
    while cdf_min_idx < 256 && hist[cdf_min_idx] == 0 {
      cdf_min_idx = cdf_min_idx + 1
    }
    let cdf_min = if cdf_min_idx < 256 { cdf[cdf_min_idx] } else { 0 }
    let denom = acc - cdf_min
    let lut = Array::makei(256, fn(i) {
      if denom <= 0 {
        i.to_byte()
      } else {
        round_byte((cdf[i] - cdf_min).to_double() * 255.0 / denom.to_double())
      }
    })
    for i = 0; i < total; i = i + 1 {
      let v = img.data[i * 4 + c].to_int()
      out.data[i * 4 + c] = lut[v]
    }
  }
  for i = 0; i < total; i = i + 1 {
    out.data[i * 4 + 3] = img.data[i * 4 + 3]
  }
  out
}

///|
/// CLAHE (Contrast-Limited Adaptive Histogram Equalization).
/// Divides the image into `grid_size` tiles, clips histograms at `clip_limit`,
/// equalizes each tile, and blends with bilinear interpolation.
pub fn clahe(img : Image, clip_limit : Double, grid_size : (Int, Int)) -> Image {
  let (gh, gw) = grid_size
  let th = (img.h + gh - 1) / gh
  let tw = (img.w + gw - 1) / gw
  // Compute equalization LUTs for each tile
  let luts = Array::makei(gh * gw, fn(_i) { Array::make(256, (0 : Byte)) })
  for ty = 0; ty < gh; ty = ty + 1 {
    for tx = 0; tx < gw; tx = tx + 1 {
      let y0 = ty * th
      let x0 = tx * tw
      let y1 = (y0 + th).min(img.h)
      let x1 = (x0 + tw).min(img.w)
      let tile_hist = Array::make(256, 0)
      for y = y0; y < y1; y = y + 1 {
        for x = x0; x < x1; x = x + 1 {
          let o = img.offset(y, x)
          let v = luma(img.data[o], img.data[o + 1], img.data[o + 2])
          tile_hist[v] = tile_hist[v] + 1
        }
      }
      // Clip histogram
      let tile_n = (y1 - y0) * (x1 - x0)
      let limit = (clip_limit * tile_n.to_double() / 256.0).to_int().max(1)
      let mut excess = 0
      for i = 0; i < 256; i = i + 1 {
        if tile_hist[i] > limit {
          excess = excess + tile_hist[i] - limit
          tile_hist[i] = limit
        }
      }
      let redistrib = excess / 256
      for i = 0; i < 256; i = i + 1 {
        tile_hist[i] = tile_hist[i] + redistrib
      }
      // Build CDF LUT
      let mut cdf_min = 0
      let mut found = false
      let mut acc = 0
      let cdf = Array::make(256, 0)
      for i = 0; i < 256; i = i + 1 {
        acc = acc + tile_hist[i]
        cdf[i] = acc
        if !found && tile_hist[i] > 0 {
          cdf_min = acc
          found = true
        }
      }
      let denom = acc - cdf_min
      let idx = ty * gw + tx
      luts[idx] = Array::makei(256, fn(i) {
        if denom <= 0 {
          i.to_byte()
        } else {
          round_byte((cdf[i] - cdf_min).to_double() * 255.0 / denom.to_double())
        }
      })
    }
  }
  // Apply with bilinear interpolation between tile LUTs
  let out = Image::new(img.h, img.w)
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let o = img.offset(y, x)
      let v = luma(img.data[o], img.data[o + 1], img.data[o + 2])
      // Tile coordinates (fractional)
      let fy = y.to_double() / th.to_double() - 0.5
      let fx = x.to_double() / tw.to_double() - 0.5
      let ty0 = clampi(fy.floor().to_int(), 0, gh - 1)
      let tx0 = clampi(fx.floor().to_int(), 0, gw - 1)
      let ty1 = clampi(ty0 + 1, 0, gh - 1)
      let tx1 = clampi(tx0 + 1, 0, gw - 1)
      let dy = clampd(fy - ty0.to_double(), 0.0, 1.0)
      let dx = clampd(fx - tx0.to_double(), 0.0, 1.0)
      let lut00 = luts[ty0 * gw + tx0]
      let lut01 = luts[ty0 * gw + tx1]
      let lut10 = luts[ty1 * gw + tx0]
      let lut11 = luts[ty1 * gw + tx1]
      let mapped = lut00[v].to_double() * (1.0 - dy) * (1.0 - dx) +
        lut01[v].to_double() * (1.0 - dy) * dx +
        lut10[v].to_double() * dy * (1.0 - dx) +
        lut11[v].to_double() * dy * dx
      let vb = round_byte(mapped)
      let oo = out.offset(y, x)
      out.data[oo] = vb
      out.data[oo + 1] = vb
      out.data[oo + 2] = vb
      out.data[oo + 3] = img.data[o + 3]
    }
  }
  out
}

///|
/// Histogram matching: adjust `img` so its luminance histogram matches
/// `ref_img`.
pub fn match_histogram(img : Image, ref_img : Image) -> Image {
  let hist_src = histogram(img)
  let hist_ref = histogram(ref_img)
  let n_src = img.h * img.w
  let n_ref = ref_img.h * ref_img.w
  if n_src == 0 || n_ref == 0 {
    return img.clone()
  }
  // Build CDFs
  let cdf_src = Array::make(256, 0.0)
  let cdf_ref = Array::make(256, 0.0)
  let mut acc_s = 0
  let mut acc_r = 0
  for i = 0; i < 256; i = i + 1 {
    acc_s = acc_s + hist_src[i]
    cdf_src[i] = acc_s.to_double() / n_src.to_double()
    acc_r = acc_r + hist_ref[i]
    cdf_ref[i] = acc_r.to_double() / n_ref.to_double()
  }
  // Build mapping: for each source value, find the ref value with closest CDF
  let mapping = Array::makei(256, fn(i) {
    let target = cdf_src[i]
    let mut best = 0
    let mut best_diff = (cdf_ref[0] - target).abs()
    for j = 1; j < 256; j = j + 1 {
      let diff = (cdf_ref[j] - target).abs()
      if diff < best_diff {
        best_diff = diff
        best = j
      }
    }
    best.to_byte()
  })
  apply_rgb_lut(img, mapping)
}

///|
/// Histogram correlation coefficient between two images' luminance histograms.
pub fn histogram_correlation(img : Image, ref_img : Image) -> Double {
  let h1 = histogram(img)
  let h2 = histogram(ref_img)
  let n1 = img.h * img.w
  let n2 = ref_img.h * ref_img.w
  if n1 == 0 || n2 == 0 {
    return 0.0
  }
  let mean1 = n1.to_double() / 256.0
  let mean2 = n2.to_double() / 256.0
  let mut num = 0.0
  let mut den1 = 0.0
  let mut den2 = 0.0
  for i = 0; i < 256; i = i + 1 {
    let d1 = h1[i].to_double() - mean1
    let d2 = h2[i].to_double() - mean2
    num = num + d1 * d2
    den1 = den1 + d1 * d1
    den2 = den2 + d2 * d2
  }
  let den = (den1 * den2).sqrt()
  if den <= 0.0 {
    0.0
  } else {
    num / den
  }
}
