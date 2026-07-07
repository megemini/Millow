///|
/// Multiplier for the LCG used by `kmeans_quantize`.
///
/// These are the constants from ANSI C / POSIX.1-2001 `rand()` (see
/// `man 3 rand` on glibc). MoonBit `Int` is 32-bit signed, so the recurrence
/// `state = state * LCG_A + LCG_C` wraps modulo 2³² on overflow, producing a
/// deterministic, reproducible sequence for any given `seed`.
///
/// We use a hand-written LCG rather than `@random.Rand` because `kmeans_quantize`
/// exposes an `Int` seed in its public API and the alignment test suite needs
/// to reproduce the exact same sequence in Python (`generate_fixtures.py`
/// mimics the 32-bit wrapping). `@random.Rand::chacha8` takes a `Bytes` seed
/// and would be impractical to replicate byte-for-byte in the reference.
const LCG_A : Int = 1103515245

///|
/// Additive constant for the LCG (ANSI C / POSIX `rand()`).
const LCG_C : Int = 12345

///|
/// Color quantization via K-means clustering on RGB pixels.
///
/// Clusters all pixels into `k` color groups using Lloyd's algorithm and
/// replaces each pixel with its cluster's centroid color. The result is an
/// image with at most `k` distinct colors.
///
/// - `k`: number of clusters (colors). Must be >= 1.
/// - `max_iter`: maximum number of K-means iterations (default 20).
/// - `seed`: RNG seed for reproducible centroid initialization (default 0).
///   Initialization picks `k` pixels spread across the image with a deterministic
///   jitter drawn from an LCG (ANSI C `rand()` constants); the same `seed`
///   always produces the same result.
///
/// Alpha is preserved. The algorithm runs on RGB only.
pub fn kmeans_quantize(
  img : Image,
  k : Int,
  max_iter? : Int = 20,
  seed? : Int = 0,
) -> Image {
  let h = img.h
  let w = img.w
  let n = h * w
  let out = Image::new(h, w)
  if n == 0 || k <= 0 {
    return out
  }
  if k == 1 {
    // Mean color.
    let mut sr = 0.0
    let mut sg = 0.0
    let mut sb = 0.0
    for i = 0; i < n; i = i + 1 {
      let o = i * 4
      sr = sr + img.data[o].to_double()
      sg = sg + img.data[o + 1].to_double()
      sb = sb + img.data[o + 2].to_double()
    }
    let mr = (sr / n.to_double()).round().to_int()
    let mg = (sg / n.to_double()).round().to_int()
    let mb = (sb / n.to_double()).round().to_int()
    for i = 0; i < n; i = i + 1 {
      let o = i * 4
      out.data[o] = clamp_byte(mr)
      out.data[o + 1] = clamp_byte(mg)
      out.data[o + 2] = clamp_byte(mb)
      out.data[o + 3] = img.data[o + 3]
    }
    return out
  }
  // Initialize centroids by picking k pixels spread across the image.
  let centroids_r = Array::make(k, 0.0)
  let centroids_g = Array::make(k, 0.0)
  let centroids_b = Array::make(k, 0.0)
  // Even spacing plus a deterministic seed-based jitter for diversity.
  let stride_n = if n >= k { (n / k).max(1) } else { 1 }
  let mut state = seed
  for i = 0; i < k; i = i + 1 {
    // Advance the LCG (wraps modulo 2^32 on MoonBit's 32-bit signed Int).
    state = state * LCG_A + LCG_C
    let base = i * stride_n
    let offset_raw = if state < 0 { -state } else { state }
    let offset = offset_raw % stride_n
    let idx = base + offset
    let idx = if idx >= n { n - 1 } else { idx }
    let o = idx * 4
    centroids_r[i] = img.data[o].to_double()
    centroids_g[i] = img.data[o + 1].to_double()
    centroids_b[i] = img.data[o + 2].to_double()
  }
  // Lloyd's algorithm.
  let labels = Array::make(n, 0)
  let mut iter = 0
  let mut changed = true
  while changed && iter < max_iter {
    changed = false
    iter = iter + 1
    // Assignment step.
    for i = 0; i < n; i = i + 1 {
      let o = i * 4
      let pr = img.data[o].to_double()
      let pg = img.data[o + 1].to_double()
      let pb = img.data[o + 2].to_double()
      let mut best = 0
      let mut best_d = -1.0
      for c = 0; c < k; c = c + 1 {
        let dr = pr - centroids_r[c]
        let dg = pg - centroids_g[c]
        let db = pb - centroids_b[c]
        let d = dr * dr + dg * dg + db * db
        if best_d < 0.0 || d < best_d {
          best_d = d
          best = c
        }
      }
      if labels[i] != best {
        labels[i] = best
        changed = true
      }
    }
    // Update step.
    let sum_r = Array::make(k, 0.0)
    let sum_g = Array::make(k, 0.0)
    let sum_b = Array::make(k, 0.0)
    let cnt = Array::make(k, 0)
    for i = 0; i < n; i = i + 1 {
      let c = labels[i]
      let o = i * 4
      sum_r[c] = sum_r[c] + img.data[o].to_double()
      sum_g[c] = sum_g[c] + img.data[o + 1].to_double()
      sum_b[c] = sum_b[c] + img.data[o + 2].to_double()
      cnt[c] = cnt[c] + 1
    }
    for c = 0; c < k; c = c + 1 {
      if cnt[c] > 0 {
        let cn = cnt[c].to_double()
        centroids_r[c] = sum_r[c] / cn
        centroids_g[c] = sum_g[c] / cn
        centroids_b[c] = sum_b[c] / cn
      }
    }
  }
  // Render output.
  for i = 0; i < n; i = i + 1 {
    let c = labels[i]
    let o = i * 4
    out.data[o] = round_byte(centroids_r[c])
    out.data[o + 1] = round_byte(centroids_g[c])
    out.data[o + 2] = round_byte(centroids_b[c])
    out.data[o + 3] = img.data[o + 3]
  }
  out
}
