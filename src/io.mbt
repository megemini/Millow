///|
/// Pluggable decoder: turn an encoded byte buffer into an image.
pub trait Decoder {
  fn decode(Self, data : Array[Byte]) -> Image raise ImageError
}

///|
/// Pluggable encoder: turn an image into an encoded byte buffer.
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
pub fn register_decoder(ext : String, decoder : &Decoder) -> Unit {
  decoder_registry[ext] = decoder
}

///|
/// Register an encoder for a format extension (e.g. "ppm").
pub fn register_encoder(ext : String, encoder : &Encoder) -> Unit {
  encoder_registry[ext] = encoder
}

///|
/// Decode a buffer using the decoder registered for `ext`.
pub fn decode(data : Array[Byte], ext : String) -> Image raise ImageError {
  match decoder_registry.get(ext) {
    Some(d) => d.decode(data)
    None => raise ImageError("no decoder registered for \{ext}")
  }
}

///|
/// Encode an image using the encoder registered for `ext`.
pub fn encode(img : Image, ext : String) -> Array[Byte] raise ImageError {
  match encoder_registry.get(ext) {
    Some(e) => e.encode(img)
    None => raise ImageError("no encoder registered for \{ext}")
  }
}

///|
/// Built-in encoder that writes binary PPM (P6).
pub(all) struct PpmEncoder {}

///|
pub impl Encoder for PpmEncoder with fn encode(_self, img) {
  to_ppm(img)
}
