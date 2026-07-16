///|
/// Scale image based on text pixel density.
///
/// Core algorithm:
///   1. Binarize and count foreground (text) pixels
///   2. Compute average pixels per character from ground-truth char count
///   3. Apply a margin to account for non-text foreground noise
///   4. scale = sqrt(min_pixels_per_char / effective_pixels_per_char)
///   5. Clamp with min_scale and bbox floor
///
/// Parameters:
///   - `img`: input image
///   - `char_count`: optional ground-truth character count. If not provided or <= 0,
///                   estimated from connected components.
///   - `min_pixels_per_char`: target minimum pixels per character (default 50)
///   - `margin_ratio`: margin for non-text foreground noise (default 1.5)
///   - `dark_text`: whether text is dark on light background (default true)
///   - `min_scale`: minimum scaling factor (default 0.4)
///   - `bboxes`: optional bounding boxes for character protection
///   - `min_char_width`: minimum character width for protection (default 3)
///   - `min_char_height`: minimum character height for protection (default 8)
///
/// Returns:
///   The scaled image.
pub fn density_scaling(
  img : Image,
  char_count? : Int? = None,
  min_pixels_per_char? : Int = 50,
  margin_ratio? : Double = 1.5,
  dark_text? : Bool = true,
  min_scale? : Double = 0.4,
  bboxes? : Array[(Int, Int, Int, Int)]? = None,
  min_char_width? : Int = 3,
  min_char_height? : Int = 8,
) -> Image {
  let h = img.h
  let w = img.w
  let total_pixels = h * w

  if total_pixels == 0 {
    return img.clone()
  }

  let (_, binary) = if dark_text {
    let (t, _) = threshold_otsu(img)
    (t, threshold_inv(img, t))
  } else {
    threshold_otsu(img)
  }

  let text_pixels = count_nonzero(binary)
  let text_ratio = text_pixels.to_double() / total_pixels.to_double()

  let estimated_char_count = match char_count {
    Some(c) => if c > 0 { c } else { 0 }
    None => 0
  }

  let final_char_count = if estimated_char_count > 0 {
    estimated_char_count
  } else {
    let kernel = Square(3)
    let cleaned = morph_open(binary, kernel)
    let (labels, num_labels) = connected_components(cleaned, 8)
    let props = regionprops(labels, num_labels)
    let areas : Array[Int] = []
    for p in props {
      if p.area > 20 {
        areas.push(p.area)
      }
    }
    if areas.length() > 0 {
      let mut sum_areas = 0
      for a in areas {
        sum_areas = sum_areas + a
      }
      let avg_area = sum_areas.to_double() / areas.length().to_double()
      if avg_area > 400.0 {
        let s = sum_areas / 200
        if s < 1 {
          1
        } else {
          s
        }
      } else {
        let l = areas.length()
        if l < 1 {
          1
        } else {
          l
        }
      }
    } else {
      1
    }
  }

  let avg_pixels_per_char = if final_char_count > 0 {
    text_pixels.to_double() / final_char_count.to_double()
  } else {
    total_pixels.to_double()
  }

  let dynamic_margin = if text_ratio > 0.3 {
    margin_ratio * 0.8
  } else if text_ratio > 0.1 {
    margin_ratio * 1.0
  } else {
    margin_ratio * 1.4
  }

  let effective_pixels_per_char = avg_pixels_per_char / dynamic_margin
  let scale_density = if effective_pixels_per_char > 0.0 {
    (min_pixels_per_char.to_double() / effective_pixels_per_char).sqrt()
  } else {
    1.0
  }

  let scale_floor = match bboxes {
    Some(bb) =>
      if bb.length() > 0 {
        let mut min_bbox_w = 0x7fffffff
        let mut min_bbox_h = 0x7fffffff
        for b in bb {
          let bw = b.2 - b.0
          let bh = b.3 - b.1
          if bw > 0 && bw < min_bbox_w {
            min_bbox_w = bw
          }
          if bh > 0 && bh < min_bbox_h {
            min_bbox_h = bh
          }
        }
        let need_protect_w = min_bbox_w < min_char_width * 2
        let need_protect_h = min_bbox_h < min_char_height * 2
        let floor_w = if min_bbox_w > 0 && need_protect_w {
          1.0 / min_bbox_w.to_double()
        } else {
          0.0
        }
        let floor_h = if min_bbox_h > 0 && need_protect_h {
          min_char_height.to_double() / 2.0 / min_bbox_h.to_double()
        } else {
          0.0
        }
        if floor_w > floor_h {
          floor_w
        } else {
          floor_h
        }
      } else {
        0.0
      }
    None => 0.0
  }

  let scale1 = if scale_density > scale_floor {
    scale_density
  } else {
    scale_floor
  }
  let scale2 = if scale1 > min_scale { scale1 } else { min_scale }
  let scale = if scale2 < 1.0 { scale2 } else { 1.0 }

  let new_w_unrounded = (w.to_double() * scale).to_int()
  let new_h_unrounded = (h.to_double() * scale).to_int()
  let mut new_w = if new_w_unrounded < 32 { 32 } else { new_w_unrounded }
  let mut new_h = if new_h_unrounded < 32 { 32 } else { new_h_unrounded }
  new_w = new_w / 32 * 32
  new_h = new_h / 32 * 32

  resize(img, new_h, new_w, Bilinear)
}
