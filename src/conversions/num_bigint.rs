#![cfg(feature = "num-bigint")]
//!  Conversions to and from [num-bigint](https://docs.rs/num-bigint)’s [`BigInt`] and [`BigUint`] types.
//!
//! This is useful for converting Python integers when they may not fit in Rust's built-in integer types.
//!
//! # Setup
//!
//! To use this feature, add this to your **`Cargo.toml`**:
//!
//! ```toml
//! [dependencies]
//! num-bigint = "*"
#![doc = concat!("pyo3 = { version = \"", env!("CARGO_PKG_VERSION"),  "\", features = [\"num-bigint\"] }")]
//! ```
//!
//! Note that you must use compatible versions of num-bigint and PyO3.
//! The required num-bigint version may vary based on the version of PyO3.
//!
//! ## Examples
//!
//! Using [`BigInt`] to correctly increment an arbitrary precision integer.
//! This is not possible with Rust's native integers if the Python integer is too large,
//! in which case it will fail its conversion and raise `OverflowError`.
//! ```rust,no_run
//! use num_bigint::BigInt;
//! use pyo3::prelude::*;
//!
//! #[pyfunction]
//! fn add_one(n: BigInt) -> BigInt {
//!     n + 1
//! }
//!
//! #[pymodule]
//! fn my_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
//!     m.add_function(wrap_pyfunction!(add_one, m)?)?;
//!     Ok(())
//! }
//! ```
//!
//! Python code:
//! ```python
//! from my_module import add_one
//!
//! n = 1 << 1337
//! value = add_one(n)
//!
//! assert n + 1 == value
//! ```

#[cfg(Py_LIMITED_API)]
use crate::types::{bytes::PyBytesMethods, PyBytes};
use crate::{
    conversion::IntoPyObject, std::num::nb_index, types::PyInt, Borrowed, Bound, FromPyObject,
    PyAny, PyErr, PyResult, Python,
};

use num_bigint::{BigInt, BigUint};

#[cfg(not(Py_LIMITED_API))]
use num_bigint::Sign;

// for identical functionality between BigInt and BigUint
macro_rules! bigint_conversion {
    ($rust_ty: ty, $is_signed: literal) => {
        #[cfg_attr(docsrs, doc(cfg(feature = "num-bigint")))]
        impl<'py> IntoPyObject<'py> for $rust_ty {
            type Target = PyInt;
            type Output = Bound<'py, Self::Target>;
            type Error = PyErr;

            #[inline]
            fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
                (&self).into_pyobject(py)
            }
        }

        #[cfg_attr(docsrs, doc(cfg(feature = "num-bigint")))]
        impl<'py> IntoPyObject<'py> for &$rust_ty {
            type Target = PyInt;
            type Output = Bound<'py, Self::Target>;
            type Error = PyErr;

            fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
                use num_traits::ToBytes;

                #[cfg(all(not(Py_LIMITED_API), Py_3_13))]
                {
                    use crate::conversions::std::num::int_from_ne_bytes;
                    let bytes = self.to_ne_bytes();
                    Ok(int_from_ne_bytes::<{ $is_signed }>(py, &bytes))
                }

                #[cfg(all(not(Py_LIMITED_API), not(Py_3_13)))]
                {
                    use crate::conversions::std::num::int_from_le_bytes;
                    let bytes = self.to_le_bytes();
                    Ok(int_from_le_bytes::<{ $is_signed }>(py, &bytes))
                }

                #[cfg(Py_LIMITED_API)]
                {
                    use $crate::py_result_ext::PyResultExt;
                    use $crate::types::any::PyAnyMethods;
                    let bytes = self.to_le_bytes();
                    let bytes_obj = PyBytes::new(py, &bytes);
                    let kwargs = if $is_signed {
                        let kwargs = crate::types::PyDict::new(py);
                        kwargs.set_item(crate::intern!(py, "signed"), true)?;
                        Some(kwargs)
                    } else {
                        None
                    };
                    unsafe {
                        py.get_type::<PyInt>()
                            .call_method("from_bytes", (bytes_obj, "little"), kwargs.as_ref())
                            .cast_into_unchecked()
                    }
                }
            }
        }
    };
}

bigint_conversion!(BigUint, false);
bigint_conversion!(BigInt, true);

#[cfg_attr(docsrs, doc(cfg(feature = "num-bigint")))]
impl<'py> FromPyObject<'_, 'py> for BigInt {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<BigInt, Self::Error> {
        // fast path - checking for subclass of `int` just checks a bit in the type object
        let num_owned: Bound<'_, PyInt>;
        let num = if let Ok(long) = ob.cast::<PyInt>() {
            long
        } else {
            num_owned = nb_index(&ob)?;
            num_owned.as_borrowed()
        };
        // fast path (CPython 3.11 and older): read the 30-bit digits directly,
        // skipping the `_PyLong_NumBits` / `_PyLong_AsByteArray` FFI calls and
        // the two's-complement byte round-trip
        #[cfg(all(not(Py_LIMITED_API), not(Py_3_12), not(PyPy), not(GraalPy)))]
        {
            if cpython_uses_30bit_digits(num.py()) {
                let (negative, digits) = unsafe { pylong_digit_parts(&num) };
                let sign = if negative { Sign::Minus } else { Sign::Plus };
                return Ok(BigInt::new(sign, u30_digits_to_limbs(digits)));
            }
        }
        #[cfg(not(Py_LIMITED_API))]
        {
            bigint_from_long_bytes(&num)
        }
        #[cfg(Py_LIMITED_API)]
        {
            let n_bits = int_n_bits(&num)?;
            if n_bits == 0 {
                return Ok(BigInt::from(0isize));
            }
            let bytes = int_to_py_bytes(&num, (n_bits + 8) / 8, true)?;
            Ok(BigInt::from_signed_bytes_le(bytes.as_bytes()))
        }
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "num-bigint")))]
impl<'py> FromPyObject<'_, 'py> for BigUint {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<BigUint, Self::Error> {
        // fast path - checking for subclass of `int` just checks a bit in the type object
        let num_owned: Bound<'_, PyInt>;
        let num = if let Ok(long) = ob.cast::<PyInt>() {
            long
        } else {
            num_owned = nb_index(&ob)?;
            num_owned.as_borrowed()
        };
        // fast path (CPython 3.11 and older): read the 30-bit digits directly
        #[cfg(all(not(Py_LIMITED_API), not(Py_3_12), not(PyPy), not(GraalPy)))]
        {
            if cpython_uses_30bit_digits(num.py()) {
                let (negative, digits) = unsafe { pylong_digit_parts(&num) };
                if !negative {
                    // negative input falls through to the FFI path to preserve
                    // its exact error/magnitude semantics for unsigned conversion
                    return Ok(BigUint::new(u30_digits_to_limbs(digits)));
                }
            }
        }
        #[cfg(not(Py_LIMITED_API))]
        {
            let buffer = int_to_u32_vec::<false>(&num)?;
            Ok(BigUint::new(buffer))
        }
        #[cfg(Py_LIMITED_API)]
        {
            let n_bits = int_n_bits(&num)?;
            if n_bits == 0 {
                return Ok(BigUint::from(0usize));
            }
            let bytes = int_to_py_bytes(&num, n_bits.div_ceil(8), false)?;
            Ok(BigUint::from_bytes_le(bytes.as_bytes()))
        }
    }
}

#[cfg(any(test, all(not(Py_LIMITED_API), not(Py_3_12), not(PyPy), not(GraalPy))))]
/// Repacks CPython's little-endian 30-bit digits into `BigUint`'s little-endian
/// 32-bit limbs, trimmed of trailing zero limbs.
fn u30_digits_to_limbs(digits: &[u32]) -> Vec<u32> {
    if digits.is_empty() {
        return Vec::new();
    }
    // `len - len / 16` is always >= `ceil(len * 30 / 32)` (exact for most
    // lengths) and, unlike `len * 30`, cannot overflow on 32-bit targets.
    let mut limbs = Vec::with_capacity(digits.len() - digits.len() / 16);
    // Stream the digits through a 64-bit accumulator, emitting one limb each
    // time 32 bits have accumulated.
    let mut acc: u64 = 0;
    let mut acc_bits: u32 = 0;
    for &digit in digits {
        acc |= u64::from(digit) << acc_bits;
        acc_bits += 30;
        while acc_bits >= 32 {
            limbs.push(acc as u32);
            acc >>= 32;
            acc_bits -= 32;
        }
    }
    if acc_bits > 0 {
        limbs.push(acc as u32);
    }
    // A nonzero topmost digit still leaves at most one trailing zero limb
    // (e.g. `[1, 1]` emits two limbs where one suffices); trimming also
    // normalizes degenerate all-zero input.
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
    limbs
}

#[cfg(all(not(Py_LIMITED_API), not(Py_3_12), not(PyPy), not(GraalPy)))]
/// CPython's `struct _longobject` for Python 3.11 and older:
/// `PyObject_VAR_HEAD` followed by 30-bit `ob_digit[]`.
#[repr(C)]
struct PyLongObject311 {
    ob_base: crate::ffi::PyVarObject,
    ob_digit: [u32; 1],
}

#[cfg(all(not(Py_LIMITED_API), not(Py_3_12), not(PyPy), not(GraalPy)))]
const _: () = assert!(
    core::mem::offset_of!(PyLongObject311, ob_digit)
        == core::mem::size_of::<crate::ffi::PyVarObject>()
);

#[cfg(all(not(Py_LIMITED_API), not(Py_3_12), not(PyPy), not(GraalPy)))]
/// Reads the sign and the 30-bit digits directly out of a CPython
/// `PyLongObject` (Python 3.11 and older layout).
///
/// This skips the `_PyLong_NumBits` and `_PyLong_AsByteArray` FFI calls, which
/// repack the same digits byte-by-byte on the C side and round-trip negative
/// values through a two's-complement byte array.
///
/// # Safety
///
/// `long` must be a valid `PyLongObject` (or subclass instance) allocated by
/// CPython with the Python 3.11-and-older layout, and
/// [`cpython_uses_30bit_digits`] must have returned `true` for the running
/// interpreter. The returned slice borrows from `long`.
#[inline]
unsafe fn pylong_digit_parts<'a>(long: &'a Bound<'_, PyInt>) -> (bool, &'a [u32]) {
    let obj: &PyLongObject311 = unsafe { &*long.as_ptr().cast_const().cast() };
    let ob_size = obj.ob_base.ob_size;
    let ndigits = ob_size.unsigned_abs();
    if ndigits == 0 {
        // `ob_size == 0` means the value is zero and CPython leaves
        // `ob_digit[0]` undefined; read nothing.
        return (false, &[]);
    }
    // `ob_size` reflects the actual allocation, so `ndigits` 30-bit digits
    // starting at `ob_digit` are all readable.
    let digits = unsafe { core::slice::from_raw_parts(obj.ob_digit.as_ptr(), ndigits) };
    (ob_size < 0, digits)
}

#[cfg(all(not(Py_LIMITED_API), not(Py_3_12), not(PyPy), not(GraalPy)))]
/// One-time check that the running CPython uses the documented 30-bit /
/// 4-byte digit layout (`sys.int_info`), guarding against nonstandard
/// 15-digit builds. Returns `false` on any failure so that callers fall back
/// to the FFI path.
fn cpython_uses_30bit_digits(py: Python<'_>) -> bool {
    use crate::types::any::PyAnyMethods;
    use crate::types::PyModule;

    static LAYOUT_OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LAYOUT_OK.get_or_init(|| {
        let check = || -> crate::PyResult<bool> {
            let int_info = PyModule::import(py, "sys")?.getattr("int_info")?;
            let bits_per_digit: i64 = int_info.getattr("bits_per_digit")?.extract()?;
            let sizeof_digit: i64 = int_info.getattr("sizeof_digit")?.extract()?;
            Ok(bits_per_digit == 30 && sizeof_digit == 4)
        };
        check().unwrap_or(false)
    })
}

#[cfg(not(any(Py_LIMITED_API, Py_3_13)))]
#[inline]
fn int_to_u32_vec<const SIGNED: bool>(long: &Bound<'_, PyInt>) -> PyResult<Vec<u32>> {
    use crate::ffi;

    let mut buffer = Vec::new();
    let n_bits = int_n_bits(long)?;
    if n_bits == 0 {
        return Ok(buffer);
    }
    let n_digits = if SIGNED {
        (n_bits + 32) / 32
    } else {
        n_bits.div_ceil(32)
    };
    buffer.reserve_exact(n_digits);
    unsafe {
        crate::err::error_on_minusone(
            long.py(),
            ffi::_PyLong_AsByteArray(
                long.as_ptr().cast(),
                buffer.as_mut_ptr() as *mut u8,
                n_digits * 4,
                1,
                SIGNED.into(),
            ),
        )?;
        buffer.set_len(n_digits)
    };
    buffer
        .iter_mut()
        .for_each(|chunk| *chunk = u32::from_le(*chunk));

    Ok(buffer)
}

#[cfg(all(not(Py_LIMITED_API), Py_3_13))]
#[inline]
fn int_to_u32_vec<const SIGNED: bool>(long: &Bound<'_, PyInt>) -> PyResult<Vec<u32>> {
    use crate::ffi;

    let mut buffer = Vec::new();
    let mut flags = ffi::Py_ASNATIVEBYTES_LITTLE_ENDIAN;
    if !SIGNED {
        flags |= ffi::Py_ASNATIVEBYTES_UNSIGNED_BUFFER | ffi::Py_ASNATIVEBYTES_REJECT_NEGATIVE;
    }
    let n_bytes =
        unsafe { ffi::PyLong_AsNativeBytes(long.as_ptr().cast(), std::ptr::null_mut(), 0, flags) };
    let n_bytes_unsigned: usize = n_bytes
        .try_into()
        .map_err(|_| crate::PyErr::fetch(long.py()))?;
    if n_bytes == 0 {
        return Ok(buffer);
    }
    let n_digits = n_bytes_unsigned.div_ceil(4);
    buffer.reserve_exact(n_digits);
    unsafe {
        ffi::PyLong_AsNativeBytes(
            long.as_ptr().cast(),
            buffer.as_mut_ptr().cast(),
            (n_digits * 4).try_into().unwrap(),
            flags,
        );
        buffer.set_len(n_digits);
    };
    buffer
        .iter_mut()
        .for_each(|chunk| *chunk = u32::from_le(*chunk));

    Ok(buffer)
}

#[cfg(not(Py_LIMITED_API))]
/// Extracts a `BigInt` via the portable `_PyLong_AsByteArray` FFI path.
fn bigint_from_long_bytes(long: &Bound<'_, PyInt>) -> PyResult<BigInt> {
    let mut buffer = int_to_u32_vec::<true>(long)?;
    let sign = if buffer.last().copied().is_some_and(|last| last >> 31 != 0) {
        // BigInt::new takes an unsigned array, so need to convert from two's complement
        // flip all bits, 'subtract' 1 (by adding one to the unsigned array)
        let mut elements = buffer.iter_mut();
        for element in elements.by_ref() {
            *element = (!*element).wrapping_add(1);
            if *element != 0 {
                // if the element didn't wrap over, no need to keep adding further ...
                break;
            }
        }
        // ... so just two's complement the rest
        for element in elements {
            *element = !*element;
        }
        Sign::Minus
    } else {
        Sign::Plus
    };
    Ok(BigInt::new(sign, buffer))
}

#[cfg(Py_LIMITED_API)]
fn int_to_py_bytes<'py>(
    long: &Bound<'py, PyInt>,
    n_bytes: usize,
    is_signed: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    use crate::intern;
    use crate::types::any::PyAnyMethods;
    let py = long.py();
    let kwargs = if is_signed {
        let kwargs = crate::types::PyDict::new(py);
        kwargs.set_item(intern!(py, "signed"), true)?;
        Some(kwargs)
    } else {
        None
    };
    let bytes = long.call_method(
        intern!(py, "to_bytes"),
        (n_bytes, intern!(py, "little")),
        kwargs.as_ref(),
    )?;
    Ok(bytes.cast_into()?)
}

#[inline]
#[cfg(any(not(Py_3_13), Py_LIMITED_API))]
fn int_n_bits(long: &Bound<'_, PyInt>) -> PyResult<usize> {
    let py = long.py();
    #[cfg(not(Py_LIMITED_API))]
    {
        // fast path
        let n_bits = unsafe { crate::ffi::_PyLong_NumBits(long.as_ptr()) };
        if n_bits == (-1isize as usize) {
            return Err(crate::PyErr::fetch(py));
        }
        Ok(n_bits)
    }

    #[cfg(Py_LIMITED_API)]
    {
        // slow path
        use crate::types::PyAnyMethods;
        long.call_method0(crate::intern!(py, "bit_length"))
            .and_then(|any| any.extract())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exceptions::PyTypeError;
    use crate::test_utils::generate_unique_module_name;
    use crate::types::{PyAnyMethods as _, PyDict, PyModule};
    use pyo3_ffi::c_str;

    fn rust_fib<T>() -> impl Iterator<Item = T>
    where
        T: From<u16>,
        for<'a> &'a T: std::ops::Add<Output = T>,
    {
        let mut f0: T = T::from(1);
        let mut f1: T = T::from(1);
        std::iter::from_fn(move || {
            let f2 = &f0 + &f1;
            Some(std::mem::replace(&mut f0, std::mem::replace(&mut f1, f2)))
        })
    }

    fn python_fib(py: Python<'_>) -> impl Iterator<Item = Bound<'_, PyAny>> + '_ {
        let mut f0 = 1i32.into_pyobject(py).unwrap().into_any();
        let mut f1 = 1i32.into_pyobject(py).unwrap().into_any();
        std::iter::from_fn(move || {
            let f2 = f0.call_method1("__add__", (&f1,)).unwrap();
            Some(std::mem::replace(&mut f0, std::mem::replace(&mut f1, f2)))
        })
    }

    #[test]
    fn u30_digits_to_limbs_edge_cases() {
        // digits are CPython 30-bit digits (little-endian); expected values are
        // num-bigint's 32-bit limbs (little-endian, trailing zeros trimmed).
        let cases: &[(&[u32], &[u32])] = &[
            (&[], &[]),  // zero
            (&[0], &[]), // degenerate single zero digit
            (&[1], &[1]),
            (&[(1 << 30) - 1], &[(1 << 30) - 1]), // full single digit
            (&[0, 1], &[1 << 30]),                // 2^30
            (&[1, 1], &[(1 << 30) + 1]),          // 2^30 + 1
            (&[(1 << 30) - 1, 3], &[u32::MAX]),   // 2^32 - 1
            (&[0, 0, 1], &[0, 1 << 28]),          // 2^60
            (&[1, 0, 1], &[1, 1 << 28]),          // 2^60 + 1
            (&[0, 0, 8], &[0, 1 << 31]),          // 2^63 (= |i64::MIN|)
        ];
        for (digits, expected) in cases {
            assert_eq!(&u30_digits_to_limbs(digits), expected, "digits: {digits:?}");
        }
    }

    #[test]
    fn u30_digits_to_limbs_matches_biguint_reference() {
        // deterministic PRNG so failures are reproducible without external deps
        struct XorShift(u64);
        impl XorShift {
            fn next_u64(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                x
            }
        }
        let mut rng = XorShift(0x5EED_1234_ABCD_0001);

        // lengths up to 120 digits plus a few 10^3000-scale magnitudes
        let lengths: Vec<usize> = (0..150)
            .map(|_| 1 + (rng.next_u64() % 120) as usize)
            .chain([333, 334, 340, 341, 350])
            .collect();

        for (case, &n_digits) in lengths.iter().enumerate() {
            let mut digits: Vec<u32> = (0..n_digits)
                .map(|_| (rng.next_u64() as u32) & ((1 << 30) - 1))
                .collect();
            // CPython invariant: the topmost digit of a nonzero int is nonzero
            *digits.last_mut().unwrap() = 1 + (rng.next_u64() as u32) % ((1 << 30) - 1);

            // reference: rebuild the value as a BigUint and take its limbs
            let mut reference = BigUint::from(0u32);
            for (i, &digit) in digits.iter().enumerate() {
                reference += BigUint::from(digit) << (30 * i);
            }
            let expected = reference.to_u32_digits();

            assert_eq!(
                u30_digits_to_limbs(&digits),
                expected,
                "case {case} with {n_digits} digits"
            );
        }
    }

    /// Cross-validates the direct digit read against the FFI byte path on
    /// CPython 3.11 and older (the only builds where the fast path exists).
    #[cfg(all(not(Py_LIMITED_API), not(Py_3_12), not(PyPy), not(GraalPy)))]
    #[test]
    fn digit_fast_path_agrees_with_ffi_path() {
        Python::attach(|py| {
            if !cpython_uses_30bit_digits(py) {
                // nonstandard digit layout (15-digit build): the fast path is
                // disabled at runtime and there is nothing to cross-check here
                return;
            }
            let mut values: Vec<BigInt> = vec![
                BigInt::from(0),
                BigInt::from(i64::MIN),
                BigInt::from(i64::MAX),
                BigInt::from(u64::MAX),
                BigInt::from(i128::MIN),
                BigInt::from(i128::MAX),
                BigInt::from(u128::MAX),
                BigInt::from(1) << 9966, // ~10^3000, huge-benchmark scale
            ];
            for i in 0..=1024usize {
                values.push(BigInt::from(1) << i);
                values.push(-BigInt::from(1) << i);
                values.push((BigInt::from(1) << i) + 1u32);
                values.push((BigInt::from(1) << i) - 1u32);
                values.push((-BigInt::from(1) << i) - 1u32);
            }
            for value in values {
                let obj = value.clone().into_pyobject(py).unwrap();

                let (negative, digits) = unsafe { pylong_digit_parts(&obj) };
                let fast = BigInt::new(
                    if negative { Sign::Minus } else { Sign::Plus },
                    u30_digits_to_limbs(digits),
                );
                assert_eq!(fast, value, "digit fast path disagrees for {value}");

                let via_ffi = bigint_from_long_bytes(&obj).unwrap();
                assert_eq!(via_ffi, value, "FFI path disagrees for {value}");
            }

            // int subclasses (bool and user subclasses) reach the fast path
            // through the same `cast` above and must read identically
            let mut subclass_cases: Vec<(BigInt, Bound<'_, PyInt>)> = vec![
                (
                    BigInt::from(1),
                    py.eval(c"True", None, None)
                        .unwrap()
                        .cast_into::<PyInt>()
                        .unwrap(),
                ),
                (
                    BigInt::from(0),
                    py.eval(c"False", None, None)
                        .unwrap()
                        .cast_into::<PyInt>()
                        .unwrap(),
                ),
            ];
            let subclass_code = c_str!("class MyInt(int):\n    pass\n");
            let module = PyModule::from_code(
                py,
                subclass_code,
                c"subclass_int.py",
                &generate_unique_module_name("subclass_int"),
            )
            .unwrap();
            for value in [BigInt::from(7), -BigInt::from(7), BigInt::from(1) << 200] {
                let int_obj = value.clone().into_pyobject(py).unwrap();
                let obj: Bound<'_, PyInt> = module
                    .getattr("MyInt")
                    .unwrap()
                    .call1((int_obj,))
                    .unwrap()
                    .cast_into::<PyInt>()
                    .unwrap();
                subclass_cases.push((value, obj));
            }
            for (value, obj) in subclass_cases {
                let (negative, digits) = unsafe { pylong_digit_parts(&obj) };
                let fast = BigInt::new(
                    if negative { Sign::Minus } else { Sign::Plus },
                    u30_digits_to_limbs(digits),
                );
                assert_eq!(fast, value, "digit fast path disagrees for {value}");
                assert_eq!(
                    bigint_from_long_bytes(&obj).unwrap(),
                    value,
                    "FFI path disagrees for {value}"
                );
            }
        });
    }

    #[test]
    fn convert_biguint() {
        Python::attach(|py| {
            // check the first 2000 numbers in the fibonacci sequence
            for (py_result, rs_result) in python_fib(py).zip(rust_fib::<BigUint>()).take(2000) {
                // Python -> Rust
                assert_eq!(py_result.extract::<BigUint>().unwrap(), rs_result);
                // Rust -> Python
                assert!(py_result.eq(rs_result).unwrap());
            }
        });
    }

    #[test]
    fn convert_bigint() {
        Python::attach(|py| {
            // check the first 2000 numbers in the fibonacci sequence
            for (py_result, rs_result) in python_fib(py).zip(rust_fib::<BigInt>()).take(2000) {
                // Python -> Rust
                assert_eq!(py_result.extract::<BigInt>().unwrap(), rs_result);
                // Rust -> Python
                assert!(py_result.eq(&rs_result).unwrap());

                // negate

                let rs_result = rs_result * -1;
                let py_result = py_result.call_method0("__neg__").unwrap();

                // Python -> Rust
                assert_eq!(py_result.extract::<BigInt>().unwrap(), rs_result);
                // Rust -> Python
                assert!(py_result.eq(rs_result).unwrap());
            }
        });
    }

    fn python_index_class(py: Python<'_>) -> Bound<'_, PyModule> {
        let index_code = c_str!(
            r#"
class C:
    def __init__(self, x):
        self.x = x
    def __index__(self):
        return self.x
"#
        );
        PyModule::from_code(
            py,
            index_code,
            c"index.py",
            &generate_unique_module_name("index"),
        )
        .unwrap()
    }

    #[test]
    fn convert_index_class() {
        Python::attach(|py| {
            let index = python_index_class(py);
            let locals = PyDict::new(py);
            locals.set_item("index", index).unwrap();
            let ob = py.eval(c"index.C(10)", None, Some(&locals)).unwrap();
            let _: BigInt = ob.extract().unwrap();
            let _: BigUint = ob.extract().unwrap();
        });
    }

    #[test]
    fn handle_zero() {
        Python::attach(|py| {
            let zero: BigInt = 0i32.into_pyobject(py).unwrap().extract().unwrap();
            assert_eq!(zero, BigInt::from(0));
        })
    }

    /// `OverflowError` on converting Python int to BigInt, see issue #629
    #[test]
    fn check_overflow() {
        Python::attach(|py| {
            macro_rules! test {
                ($T:ty, $value:expr, $py:expr) => {
                    let value = $value;
                    println!("{}: {}", stringify!($T), value);
                    let python_value = value.clone().into_pyobject(py).unwrap();
                    let roundtrip_value = python_value.extract::<$T>().unwrap();
                    assert_eq!(value, roundtrip_value);
                };
            }

            for i in 0..=256usize {
                // test a lot of values to help catch other bugs too
                test!(BigInt, BigInt::from(i), py);
                test!(BigUint, BigUint::from(i), py);
                test!(BigInt, -BigInt::from(i), py);
                test!(BigInt, BigInt::from(1) << i, py);
                test!(BigUint, BigUint::from(1u32) << i, py);
                test!(BigInt, -BigInt::from(1) << i, py);
                test!(BigInt, (BigInt::from(1) << i) + 1u32, py);
                test!(BigUint, (BigUint::from(1u32) << i) + 1u32, py);
                test!(BigInt, (-BigInt::from(1) << i) + 1u32, py);
                test!(BigInt, (BigInt::from(1) << i) - 1u32, py);
                test!(BigUint, (BigUint::from(1u32) << i) - 1u32, py);
                test!(BigInt, (-BigInt::from(1) << i) - 1u32, py);
            }
        });
    }

    #[test]
    fn from_py_float_type_error() {
        Python::attach(|py| {
            let obj = (12.3f64).into_pyobject(py).unwrap();
            let err = obj.extract::<BigInt>().unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));

            let err = obj.extract::<BigUint>().unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));
        });
    }
}
