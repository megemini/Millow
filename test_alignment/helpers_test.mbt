// Helpers for the alignment test suite: build images from `Array[Int]`
// fixtures and compare image bytes against expected values with optional
// tolerance for rounding differences.

///|
/// Build an `Image` from an `Array[Int]` of RGBA8 values (length must be
/// `h * w * 4`). Raises on length mismatch.
fn img_from_ints(h : Int, w : Int, data : Array[Int]) -> Image raise {
  let bytes = Array::makei(data.length(), fn(i) { data[i].to_byte() })
  Image::from_data(bytes, h, w)
}

///|
/// Assert every byte of `img` equals `expected` exactly.
fn assert_img_eq(img : Image, expected : Array[Int]) -> Unit raise {
  let bytes = img.data
  assert_eq(bytes.length(), expected.length())
  for i = 0; i < bytes.length(); i = i + 1 {
    assert_eq(bytes[i].to_int(), expected[i])
  }
}

///|
/// Assert every byte of `img` is within `tol` of `expected`.
fn assert_img_close(
  img : Image,
  expected : Array[Int],
  tol : Int,
) -> Unit raise {
  let bytes = img.data
  assert_eq(bytes.length(), expected.length())
  for i = 0; i < bytes.length(); i = i + 1 {
    let d = (bytes[i].to_int() - expected[i]).abs()
    assert_true(d <= tol)
  }
}

///|
/// Compare an `Array[Int]` (e.g. a histogram) exactly.
fn assert_arr_eq(actual : Array[Int], expected : Array[Int]) -> Unit raise {
  assert_eq(actual.length(), expected.length())
  for i = 0; i < actual.length(); i = i + 1 {
    assert_eq(actual[i], expected[i])
  }
}

///|
/// Compare an `Array[Array[Float]]` (e.g. sobel gradient) with tolerance.
fn assert_arr2_close(
  actual : Array[Array[Float]],
  expected : Array[Array[Double]],
  tol : Double,
) -> Unit raise {
  assert_eq(actual.length(), expected.length())
  for y = 0; y < actual.length(); y = y + 1 {
    assert_eq(actual[y].length(), expected[y].length())
    for x = 0; x < actual[y].length(); x = x + 1 {
      let d = (actual[y][x].to_double() - expected[y][x]).abs()
      assert_true(d <= tol)
    }
  }
}

///|
/// Compare an `Array[Array[Double]]` (e.g. corner detection) with tolerance.
fn assert_arr2_close_double(
  actual : Array[Array[Double]],
  expected : Array[Array[Double]],
  tol : Double,
) -> Unit raise {
  assert_eq(actual.length(), expected.length())
  for y = 0; y < actual.length(); y = y + 1 {
    assert_eq(actual[y].length(), expected[y].length())
    for x = 0; x < actual[y].length(); x = x + 1 {
      let d = (actual[y][x] - expected[y][x]).abs()
      assert_true(d <= tol)
    }
  }
}
