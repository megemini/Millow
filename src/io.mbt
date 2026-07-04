///|
/// Pluggable decoder: turn an encoded byte buffer into an image.
///
/// Implementations decode `data` into an `Image`, raising `ImageError` on a
/// malformed or unsupported buffer. Register instances with
/// `register_decoder` to make them available via `decode`.
pub trait Decoder {
  fn decode(Self, data : Array[Byte]) -> Image raise ImageError
}

///|
/// Pluggable encoder: turn an image into an encoded byte buffer.
///
/// Implementations serialize `img` to a format-specific byte stream, raising
/// `ImageError` on failure. Register instances with `register_encoder` to
/// make them available via `encode`.
pub trait Encoder {
  fn encode(Self, img : Image) -> Array[Byte] raise ImageError
}

///|
/// Global decoder registry keyed by format extension.
let decoder_registry : Map[String, &Decoder] = {}

///|
/// Global encoder registry keyed by format extension.
let encoder_registry : Map[String, &Encoder] = {}

///|
/// Register a decoder for a format extension (e.g. "ppm").
///
/// Subsequent calls to `decode(data, ext)` dispatch to `decoder`. Re-registering
/// an extension replaces the previous entry.
pub fn register_decoder(ext : String, decoder : &Decoder) -> Unit {
  decoder_registry[ext] = decoder
}

///|
/// Register an encoder for a format extension (e.g. "ppm").
///
/// Subsequent calls to `encode(img, ext)` dispatch to `encoder`. Re-registering
/// an extension replaces the previous entry.
pub fn register_encoder(ext : String, encoder : &Encoder) -> Unit {
  encoder_registry[ext] = encoder
}

///|
/// Decode a buffer using the decoder registered for `ext`.
///
/// Raises `ImageError` when no decoder is registered for `ext` or the decoder
/// rejects the buffer.
pub fn decode(data : Array[Byte], ext : String) -> Image raise ImageError {
  match decoder_registry.get(ext) {
    Some(d) => d.decode(data)
    None => raise ImageError("no decoder registered for \{ext}")
  }
}

///|
/// Encode an image using the encoder registered for `ext`.
///
/// Raises `ImageError` when no encoder is registered for `ext` or the encoder
/// fails.
pub fn encode(img : Image, ext : String) -> Array[Byte] raise ImageError {
  match encoder_registry.get(ext) {
    Some(e) => e.encode(img)
    None => raise ImageError("no encoder registered for \{ext}")
  }
}

///|
/// Built-in encoder that writes binary PPM (P6).
///
/// A unit struct implementing `Encoder` by delegating to `to_ppm`. Register it
/// with `register_encoder("ppm", &PpmEncoder{})` to plug into `encode`.
pub(all) struct PpmEncoder {}

///|
pub impl Encoder for PpmEncoder with fn encode(_self, img) {
  to_ppm(img)
}
