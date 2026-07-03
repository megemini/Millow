///|
/// Connected components labelling (4- or 8-connectivity).
/// Returns `(labels, num_labels)` where `labels[y][x]` is the label (1-based).
pub fn connected_components(
  img : Image,
  connectivity : Int,
) -> (Array[Array[Int]], Int) {
  let h = img.h
  let w = img.w
  let labels = Array::makei(h, fn(_i) { Array::make(w, 0) })
  let parent : Array[Int] = [0]
  let find = fn(x0 : Int) -> Int {
    let mut x = x0
    while parent[x] != x {
      parent[x] = parent[parent[x]]
      x = parent[x]
    }
    x
  }
  let union = fn(a : Int, b : Int) -> Unit {
    let ra = find(a)
    let rb = find(b)
    if ra != rb {
      parent[ra] = rb
    }
  }
  let mut next_label = 1
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      let o = img.offset(y, x)
      let fg = img.data[o] != 0 || img.data[o + 1] != 0 || img.data[o + 2] != 0
      if !fg {
        continue
      }
      let neighbors : Array[Int] = []
      if y > 0 && labels[y - 1][x] > 0 {
        neighbors.push(labels[y - 1][x])
      }
      if x > 0 && labels[y][x - 1] > 0 {
        neighbors.push(labels[y][x - 1])
      }
      if connectivity >= 8 {
        if y > 0 && x > 0 && labels[y - 1][x - 1] > 0 {
          neighbors.push(labels[y - 1][x - 1])
        }
        if y > 0 && x < w - 1 && labels[y - 1][x + 1] > 0 {
          neighbors.push(labels[y - 1][x + 1])
        }
      }
      if neighbors.length() == 0 {
        labels[y][x] = next_label
        parent.push(next_label)
        next_label = next_label + 1
      } else {
        let mut min_label = neighbors[0]
        for n in neighbors {
          let r = find(n)
          if r < min_label {
            min_label = r
          }
        }
        labels[y][x] = min_label
        for n in neighbors {
          union(n, min_label)
        }
      }
    }
  }
  // Flatten labels
  let remap = Array::makei(next_label, fn(_i) { 0 })
  let mut count = 0
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      if labels[y][x] > 0 {
        let root = find(labels[y][x])
        if remap[root] == 0 {
          count = count + 1
          remap[root] = count
        }
        labels[y][x] = remap[root]
      }
    }
  }
  (labels, count)
}

///|
/// Find contours as arrays of `(y, x)` boundary pixels (simple boundary trace).
pub fn find_contours(img : Image) -> Array[Array[(Int, Int)]] {
  let (labels, n) = connected_components(img, 4)
  let contours : Array[Array[(Int, Int)]] = []
  for label = 1; label <= n; label = label + 1 {
    let contour : Array[(Int, Int)] = []
    for y = 0; y < img.h; y = y + 1 {
      for x = 0; x < img.w; x = x + 1 {
        if labels[y][x] != label {
          continue
        }
        // A boundary pixel has at least one 4-neighbour with a different label
        let is_boundary = (y == 0 || labels[y - 1][x] != label) ||
          (y == img.h - 1 || labels[y + 1][x] != label) ||
          (x == 0 || labels[y][x - 1] != label) ||
          x == img.w - 1 ||
          labels[y][x + 1] != label
        if is_boundary {
          contour.push((y, x))
        }
      }
    }
    if contour.length() > 0 {
      contours.push(contour)
    }
  }
  contours
}

///|
/// Region properties for labelled images.
pub struct RegionProps {
  label : Int
  area : Int
  centroid : (Double, Double)
  bbox : (Int, Int, Int, Int)
}

///|
/// Compute region properties for each label (1..num_labels).
pub fn regionprops(
  labels : Array[Array[Int]],
  num_labels : Int,
) -> Array[RegionProps] {
  let h = labels.length()
  let w = if h == 0 { 0 } else { labels[0].length() }
  let props = Array::makei(num_labels + 1, fn(_i) {
    { label: 0, area: 0, centroid: (0.0, 0.0), bbox: (0, 0, 0, 0) }
  })
  for l = 1; l <= num_labels; l = l + 1 {
    props[l] = { label: l, area: 0, centroid: (0.0, 0.0), bbox: (h, w, 0, 0) }
  }
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      let l = labels[y][x]
      if l <= 0 || l > num_labels {
        continue
      }
      props[l] = {
        label: l,
        area: props[l].area + 1,
        centroid: (
          props[l].centroid.0 + y.to_double(),
          props[l].centroid.1 + x.to_double(),
        ),
        bbox: (
          props[l].bbox.0.min(y),
          props[l].bbox.1.min(x),
          props[l].bbox.2.max(y),
          props[l].bbox.3.max(x),
        ),
      }
    }
  }
  for l = 1; l <= num_labels; l = l + 1 {
    if props[l].area > 0 {
      props[l] = {
        label: l,
        area: props[l].area,
        centroid: (
          props[l].centroid.0 / props[l].area.to_double(),
          props[l].centroid.1 / props[l].area.to_double(),
        ),
        bbox: props[l].bbox,
      }
    }
  }
  Array::makei(num_labels, fn(i) { props[i + 1] })
}

///|
/// Count non-zero (non-black) pixels.
pub fn count_nonzero(img : Image) -> Int {
  let mut count = 0
  let n = img.h * img.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    if img.data[o] != 0 || img.data[o + 1] != 0 || img.data[o + 2] != 0 {
      count = count + 1
    }
  }
  count
}

///|
/// Count pixels whose luma exceeds `threshold`.
pub fn count_pixels(img : Image, threshold : Byte) -> Int {
  let t = threshold.to_int()
  let mut count = 0
  let n = img.h * img.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    if luma(img.data[o], img.data[o + 1], img.data[o + 2]) > t {
      count = count + 1
    }
  }
  count
}

///|
/// Raw image moments (m00, m10, m01, m20, m11, m02, m30, m21, m12, m03).
pub fn moments(img : Image) -> Array[Double] {
  let m = Array::make(10, 0.0)
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let o = img.offset(y, x)
      let v = luma(img.data[o], img.data[o + 1], img.data[o + 2]).to_double()
      m[0] = m[0] + v
      m[1] = m[1] + y.to_double() * v
      m[2] = m[2] + x.to_double() * v
      m[3] = m[3] + y.to_double() * y.to_double() * v
      m[4] = m[4] + y.to_double() * x.to_double() * v
      m[5] = m[5] + x.to_double() * x.to_double() * v
      m[6] = m[6] + y.to_double() * y.to_double() * y.to_double() * v
      m[7] = m[7] + y.to_double() * y.to_double() * x.to_double() * v
      m[8] = m[8] + y.to_double() * x.to_double() * x.to_double() * v
      m[9] = m[9] + x.to_double() * x.to_double() * x.to_double() * v
    }
  }
  m
}

///|
/// Hu moments (7 translation/scale/rotation invariant moments).
pub fn hu_moments(img : Image) -> Array[Double] {
  let m = moments(img)
  let m00 = m[0]
  if m00 == 0.0 {
    return Array::make(7, 0.0)
  }
  let yb = m[1] / m00
  let xb = m[2] / m00
  // Central moments
  let mut mu20 = 0.0
  let mut mu11 = 0.0
  let mut mu02 = 0.0
  let mut mu30 = 0.0
  let mut mu21 = 0.0
  let mut mu12 = 0.0
  let mut mu03 = 0.0
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let o = img.offset(y, x)
      let v = luma(img.data[o], img.data[o + 1], img.data[o + 2]).to_double()
      let dy = y.to_double() - yb
      let dx = x.to_double() - xb
      mu20 = mu20 + dy * dy * v
      mu11 = mu11 + dy * dx * v
      mu02 = mu02 + dx * dx * v
      mu30 = mu30 + dy * dy * dy * v
      mu21 = mu21 + dy * dy * dx * v
      mu12 = mu12 + dy * dx * dx * v
      mu03 = mu03 + dx * dx * dx * v
    }
  }
  // Normalized central moments
  let n20 = mu20 / (m00 * m00)
  let n11 = mu11 / (m00 * m00)
  let n02 = mu02 / (m00 * m00)
  let n30 = mu30 / (m00 * m00 * m00.sqrt())
  let n21 = mu21 / (m00 * m00 * m00.sqrt())
  let n12 = mu12 / (m00 * m00 * m00.sqrt())
  let n03 = mu03 / (m00 * m00 * m00.sqrt())
  // Hu moments
  let h1 = n20 + n02
  let h2 = (n20 - n02) * (n20 - n02) + 4.0 * n11 * n11
  let h3 = (n30 - 3.0 * n12) * (n30 - 3.0 * n12) +
    (3.0 * n21 - n03) * (3.0 * n21 - n03)
  let h4 = (n30 + n12) * (n30 + n12) + (n21 + n03) * (n21 + n03)
  let h5 = (n30 - 3.0 * n12) *
    (n30 + n12) *
    ((n30 + n12) * (n30 + n12) - 3.0 * (n21 + n03) * (n21 + n03)) +
    (3.0 * n21 - n03) *
    (n21 + n03) *
    (3.0 * (n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03))
  let h6 = (n20 - n02) * ((n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03)) +
    4.0 * n11 * (n30 + n12) * (n21 + n03)
  let h7 = (3.0 * n21 - n03) *
    (n30 + n12) *
    ((n30 + n12) * (n30 + n12) - 3.0 * (n21 + n03) * (n21 + n03)) -
    (n30 - 3.0 * n12) *
    (n21 + n03) *
    (3.0 * (n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03))
  [h1, h2, h3, h4, h5, h6, h7]
}
