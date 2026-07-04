///|
test "all Augmentation variants are constructible" {
  let _pipeline = [
    Crop(0, 0, 2, 2),
    Resize(4, 4),
    FlipHorizontal,
    FlipVertical,
    Rotate(0.0),
    Brightness(1.0),
    Contrast(1.0),
    Gamma(1.0),
    NoiseGaussian(0.0),
    NoiseSaltPepper(0.0),
    ColorJitter(0.0, 0.0, 0.0, 0.0),
  ]
}

///|
test "all BorderMode variants are constructible" {
  let _modes = [
    BorderMode::Replicate,
    BorderMode::Reflect,
    BorderMode::Wrap,
    BorderMode::Constant(0, 0, 0, 255),
  ]
}
