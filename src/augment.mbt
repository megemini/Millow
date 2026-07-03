///|
/// Data augmentation operations that can be composed into a pipeline.
pub enum Augmentation {
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
pub fn random_rotate(img : Image, max_angle : Double) -> Image {
  let rng = @random.Rand::new()
  let angle = (2.0 * rng.double() - 1.0) * max_angle
  rotate_any(img, angle, Bilinear)
}

///|
/// Random brightness adjustment within `[1-max_factor, 1+max_factor]`.
pub fn random_brightness(img : Image, max_factor : Double) -> Image {
  let rng = @random.Rand::new()
  let factor = 1.0 + (2.0 * rng.double() - 1.0) * max_factor
  adjust_brightness(img, factor)
}

///|
/// Random contrast adjustment within `[1-max_factor, 1+max_factor]`.
pub fn random_contrast(img : Image, max_factor : Double) -> Image {
  let rng = @random.Rand::new()
  let factor = 1.0 + (2.0 * rng.double() - 1.0) * max_factor
  adjust_contrast(img, factor)
}

///|
/// Random gamma within `[1/max_gamma, max_gamma]`.
pub fn random_gamma(img : Image, max_gamma : Double) -> Image {
  let rng = @random.Rand::new()
  let log_g = (2.0 * rng.double() - 1.0) * @math.ln(max_gamma)
  adjust_gamma(img, @math.exp(log_g))
}

///|
/// Add Gaussian noise with the given standard deviation.
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
pub fn random_color_jitter(
  img : Image,
  brightness : Double,
  contrast : Double,
  _saturation : Double,
  _hue : Double,
) -> Image {
  let mut out = img.clone()
  if brightness > 0.0 {
    out = random_brightness(out, brightness)
  }
  if contrast > 0.0 {
    out = random_contrast(out, contrast)
  }
  // Saturation and hue jitter require HSV conversion; placeholder for now
  out
}

///|
/// Apply a sequence of augmentations in order.
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
pub fn process_batch(imgs : Array[Image], f : (Image) -> Image) -> Array[Image] {
  Array::makei(imgs.length(), fn(i) { f(imgs[i]) })
}
