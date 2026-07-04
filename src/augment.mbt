///|
/// Data augmentation operations that can be composed into a pipeline.
///
/// Each variant wraps the parameters of one transformation; `augment_pipeline`
/// applies them in order. Variants:
///
/// - `Crop(y, x, h, w)`: extract the sub-image at `(y, x)` of size `h x w`.
/// - `Resize(dst_h, dst_w)`: resize to `dst_h x dst_w` using bilinear sampling.
/// - `FlipHorizontal` / `FlipVertical`: mirror the image.
/// - `Rotate(angle)`: rotate by `angle` degrees (bilinear).
/// - `Brightness(factor)` / `Contrast(factor)`: multiplicative adjustments.
/// - `Gamma(g)`: gamma correction with exponent `g`.
/// - `NoiseGaussian(std)`: add Gaussian noise with standard deviation `std`.
/// - `NoiseSaltPepper(prob)`: corrupt pixels with probability `prob`.
/// - `ColorJitter(b, c, s, h)`: brightness/contrast/saturation/hue jitter.
pub(all) enum Augmentation {
  Crop(Int, Int, Int, Int) // y, x, h, w
  Resize(Int, Int) // dst_h, dst_w
  FlipHorizontal
  FlipVertical
  Rotate(Double) // angle degrees
  Brightness(Double) // factor
  Contrast(Double) // factor
  Gamma(Double) // gamma
  NoiseGaussian(Double) // std
  NoiseSaltPepper(Double) // prob
  ColorJitter(Double, Double, Double, Double) // brightness, contrast, saturation, hue
}

///|
/// Random crop to target size.
///
/// Picks a random top-left corner so the `h x w` window fits inside the image.
/// Returns a clone of the input when it already fits within `h x w`.
pub fn random_crop(img : Image, h : Int, w : Int) -> Image raise {
  if h >= img.h && w >= img.w {
    return img.clone()
  }
  let rng = @random.Rand::new()
  let y0 = @math.floor((img.h - h).to_double() * rng.double()).to_int().max(0)
  let x0 = @math.floor((img.w - w).to_double() * rng.double()).to_int().max(0)
  crop(img, y0, x0, h.min(img.h), w.min(img.w))
}

///|
/// Random horizontal flip (50% probability).
///
/// Returns a mirrored copy with probability `0.5`, otherwise a clone of the
/// input. Each call draws a fresh random sample.
pub fn random_flip_horizontal(img : Image) -> Image {
  let rng = @random.Rand::new()
  if rng.double() < 0.5 {
    flip_horizontal(img)
  } else {
    img.clone()
  }
}

///|
/// Random rotation within `[-max_angle, max_angle]` degrees.
///
/// Samples an angle uniformly from the symmetric range and rotates with
/// bilinear sampling.
pub fn random_rotate(img : Image, max_angle : Double) -> Image {
  let rng = @random.Rand::new()
  let angle = (2.0 * rng.double() - 1.0) * max_angle
  rotate_any(img, angle, Bilinear)
}

///|
/// Random brightness adjustment within `[1-max_factor, 1+max_factor]`.
///
/// Multiplies each RGB channel by a factor sampled uniformly from the
/// symmetric range around `1.0`.
pub fn random_brightness(img : Image, max_factor : Double) -> Image {
  let rng = @random.Rand::new()
  let factor = 1.0 + (2.0 * rng.double() - 1.0) * max_factor
  adjust_brightness(img, factor)
}

///|
/// Random contrast adjustment within `[1-max_factor, 1+max_factor]`.
///
/// Scales pixel deviations from the image mean by a factor sampled uniformly
/// from the symmetric range around `1.0`.
pub fn random_contrast(img : Image, max_factor : Double) -> Image {
  let rng = @random.Rand::new()
  let factor = 1.0 + (2.0 * rng.double() - 1.0) * max_factor
  adjust_contrast(img, factor)
}

///|
/// Random gamma within `[1/max_gamma, max_gamma]`.
///
/// Samples the exponent log-uniformly so low and high gamma are equally
/// likely, then applies `adjust_gamma`.
pub fn random_gamma(img : Image, max_gamma : Double) -> Image {
  let rng = @random.Rand::new()
  let log_g = (2.0 * rng.double() - 1.0) * @math.ln(max_gamma)
  adjust_gamma(img, @math.exp(log_g))
}

///|
/// Add Gaussian noise with the given standard deviation.
///
/// Adds independent zero-mean Gaussian noise (via Box-Muller) of standard
/// deviation `std` to each RGB channel; alpha is preserved.
pub fn random_noise_gaussian(img : Image, std : Double) -> Image {
  let out = img.clone()
  let rng = @random.Rand::new()
  let n = out.h * out.w
  for i = 0; i < n; i = i + 1 {
    // Box-Muller transform
    let u1 = rng.double().max(1.0e-10)
    let u2 = rng.double()
    let z0 = (-2.0 * @math.ln(u1)).sqrt() * @math.cos(2.0 * @math.PI * u2)
    let noise = z0 * std
    let o = i * 4
    for c = 0; c < 3; c = c + 1 {
      out.data[o + c] = round_byte(out.data[o + c].to_double() + noise)
    }
  }
  out
}

///|
/// Add salt-and-pepper noise with the given probability.
///
/// Each pixel is set to black with probability `prob / 2` and to white with
/// probability `prob / 2`; otherwise it is left unchanged. Only RGB channels
/// are affected; alpha is preserved.
pub fn random_noise_salt_pepper(img : Image, prob : Double) -> Image {
  let out = img.clone()
  let rng = @random.Rand::new()
  let n = out.h * out.w
  for i = 0; i < n; i = i + 1 {
    let r = rng.double()
    if r < prob / 2.0 {
      let o = i * 4
      out.data[o] = 0
      out.data[o + 1] = 0
      out.data[o + 2] = 0
    } else if r < prob {
      let o = i * 4
      out.data[o] = 255
      out.data[o + 1] = 255
      out.data[o + 2] = 255
    }
  }
  out
}

///|
/// Random colour jitter (brightness, contrast, saturation, hue).
///
/// Applies brightness and contrast jitter (multiplicative, via
/// `random_brightness` / `random_contrast`) when their magnitudes are
/// positive. Saturation jitter scales the HSV S channel by a factor sampled
/// from `[1-saturation, 1+saturation]` (clamped to `[0, 1]`). Hue jitter
/// shifts the H channel by a uniform sample in `[-hue, hue]` degrees
/// (wrapped to `[0, 360)`). The HSV round-trip is done once when either
/// saturation or hue is requested.
pub fn random_color_jitter(
  img : Image,
  brightness : Double,
  contrast : Double,
  saturation : Double,
  hue : Double,
) -> Image {
  let mut out = img.clone()
  if brightness > 0.0 {
    out = random_brightness(out, brightness)
  }
  if contrast > 0.0 {
    out = random_contrast(out, contrast)
  }
  if saturation > 0.0 || hue > 0.0 {
    let rng = @random.Rand::new()
    let s_factor = if saturation > 0.0 {
      1.0 + (2.0 * rng.double() - 1.0) * saturation
    } else {
      1.0
    }
    let h_shift = if hue > 0.0 { (2.0 * rng.double() - 1.0) * hue } else { 0.0 }
    let (h_arr, s_arr, v_arr) = to_hsv(out)
    let rows = h_arr.length()
    let cols = if rows == 0 { 0 } else { h_arr[0].length() }
    for y = 0; y < rows; y = y + 1 {
      for x = 0; x < cols; x = x + 1 {
        s_arr[y][x] = (s_arr[y][x] * s_factor).max(0.0).min(1.0)
        let mut new_h = h_arr[y][x] + h_shift
        if new_h < 0.0 {
          new_h = new_h + 360.0
        }
        if new_h >= 360.0 {
          new_h = new_h - 360.0
        }
        h_arr[y][x] = new_h
      }
    }
    out = from_hsv(h_arr, s_arr, v_arr)
  }
  out
}

///|
/// Apply a sequence of augmentations in order.
///
/// Folds `pipeline` left-to-right over `img`, dispatching each `Augmentation`
/// variant to its underlying operation. May raise from crop/resize.
pub fn augment_pipeline(
  img : Image,
  pipeline : Array[Augmentation],
) -> Image raise {
  let mut out = img
  for aug in pipeline {
    out = match aug {
      Crop(y, x, h, w) => crop(out, y, x, h, w)
      Resize(h, w) => resize(out, h, w, Bilinear)
      FlipHorizontal => flip_horizontal(out)
      FlipVertical => flip_vertical(out)
      Rotate(angle) => rotate_any(out, angle, Bilinear)
      Brightness(delta) => adjust_brightness(out, delta)
      Contrast(factor) => adjust_contrast(out, factor)
      Gamma(g) => adjust_gamma(out, g)
      NoiseGaussian(std) => random_noise_gaussian(out, std)
      NoiseSaltPepper(prob) => random_noise_salt_pepper(out, prob)
      ColorJitter(b, c, s, h) => random_color_jitter(out, b, c, s, h)
    }
  }
  out
}

///|
/// Randomly choose and apply one augmentation from a weighted list.
///
/// `choices` pairs each `Augmentation` with a non-negative weight; the chosen
/// entry is sampled proportionally and applied via `augment_pipeline`. Returns
/// a clone of the input when `choices` is empty.
pub fn augment_random_choice(
  img : Image,
  choices : Array[(Double, Augmentation)],
) -> Image raise {
  let mut total = 0.0
  for entry in choices {
    total = total + entry.0
  }
  let rng = @random.Rand::new()
  let mut r = rng.double() * total
  for entry in choices {
    r = r - entry.0
    if r <= 0.0 {
      return augment_pipeline(img, [entry.1])
    }
  }
  img.clone()
}

///|
/// Apply a function to a batch of images.
///
/// Returns a new array where `f` has been applied to each element of `imgs`
/// in index order.
pub fn process_batch(imgs : Array[Image], f : (Image) -> Image) -> Array[Image] {
  Array::makei(imgs.length(), fn(i) { f(imgs[i]) })
}
