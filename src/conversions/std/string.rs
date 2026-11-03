#[cfg(feature = "experimental-inspect")]
use crate::inspect::types::TypeInfo;
#[cfg(feature = "experimental-inspect")]
use crate::inspect::PyStaticExpr;
#[cfg(feature = "experimental-inspect")]
use crate::type_object::PyTypeInfo;
use crate::{
    conversion::IntoPyObject, instance::Bound, types::PyString, Borrowed, FromPyObject, PyAny,
    PyErr, Python,
};
use std::{borrow::Cow, convert::Infallible, marker::PhantomData};

impl<'py> IntoPyObject<'py> for &str {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    #[cfg(feature = "experimental-inspect")]
    const OUTPUT_TYPE: PyStaticExpr = String::OUTPUT_TYPE;

    #[inline]
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyString::new(py, self))
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_output() -> TypeInfo {
        <String>::type_output()
    }
}

impl<'py> IntoPyObject<'py> for &&str {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    #[cfg(feature = "experimental-inspect")]
    const OUTPUT_TYPE: PyStaticExpr = String::OUTPUT_TYPE;

    #[inline]
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        (*self).into_pyobject(py)
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_output() -> TypeInfo {
        <String>::type_output()
    }
}

impl<'py> IntoPyObject<'py> for Cow<'_, str> {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    #[cfg(feature = "experimental-inspect")]
    const OUTPUT_TYPE: PyStaticExpr = String::OUTPUT_TYPE;

    #[inline]
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        (*self).into_pyobject(py)
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_output() -> TypeInfo {
        <String>::type_output()
    }
}

impl<'py> IntoPyObject<'py> for &Cow<'_, str> {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    #[cfg(feature = "experimental-inspect")]
    const OUTPUT_TYPE: PyStaticExpr = String::OUTPUT_TYPE;

    #[inline]
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        (&**self).into_pyobject(py)
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_output() -> TypeInfo {
        <String>::type_output()
    }
}

impl<'py> IntoPyObject<'py> for char {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    #[cfg(feature = "experimental-inspect")]
    const OUTPUT_TYPE: PyStaticExpr = String::OUTPUT_TYPE;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let mut bytes = [0u8; 4];
        Ok(PyString::new(py, self.encode_utf8(&mut bytes)))
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_output() -> TypeInfo {
        <String>::type_output()
    }
}

impl<'py> IntoPyObject<'py> for &char {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    #[cfg(feature = "experimental-inspect")]
    const OUTPUT_TYPE: PyStaticExpr = String::OUTPUT_TYPE;

    #[inline]
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        (*self).into_pyobject(py)
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_output() -> TypeInfo {
        <String>::type_output()
    }
}

impl<'py> IntoPyObject<'py> for String {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    #[cfg(feature = "experimental-inspect")]
    const OUTPUT_TYPE: PyStaticExpr = PyString::TYPE_HINT;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyString::new(py, &self))
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_output() -> TypeInfo {
        TypeInfo::builtin("str")
    }
}

impl<'py> IntoPyObject<'py> for &String {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    #[cfg(feature = "experimental-inspect")]
    const OUTPUT_TYPE: PyStaticExpr = String::OUTPUT_TYPE;

    #[inline]
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyString::new(py, self))
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_output() -> TypeInfo {
        <String>::type_output()
    }
}

#[cfg(any(Py_3_10, not(Py_LIMITED_API)))]
impl<'a> crate::conversion::FromPyObject<'a, '_> for &'a str {
    type Error = PyErr;

    #[cfg(feature = "experimental-inspect")]
    const INPUT_TYPE: PyStaticExpr = PyString::TYPE_HINT;

    fn extract(ob: crate::Borrowed<'a, '_, PyAny>) -> Result<Self, Self::Error> {
        ob.cast::<PyString>()?.to_str()
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_input() -> TypeInfo {
        <String as crate::FromPyObject>::type_input()
    }
}

impl<'a> crate::conversion::FromPyObject<'a, '_> for Cow<'a, str> {
    type Error = PyErr;

    #[cfg(feature = "experimental-inspect")]
    const INPUT_TYPE: PyStaticExpr = PyString::TYPE_HINT;

    fn extract(ob: crate::Borrowed<'a, '_, PyAny>) -> Result<Self, Self::Error> {
        ob.cast::<PyString>()?.to_cow()
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_input() -> TypeInfo {
        <String as crate::FromPyObject>::type_input()
    }
}

/// Allows extracting strings from Python objects.
/// Accepts Python `str` and `unicode` objects.
impl FromPyObject<'_, '_> for String {
    type Error = PyErr;

    #[cfg(feature = "experimental-inspect")]
    const INPUT_TYPE: PyStaticExpr = PyString::TYPE_HINT;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        obj.cast::<PyString>()?.to_cow().map(Cow::into_owned)
    }

    #[cfg(all(
        not(Py_LIMITED_API),
        not(PyPy),
        not(GraalPy),
        not(Py_GIL_DISABLED)
    ))]
    #[doc(hidden)]
    #[inline(always)]
    fn borrowed_item_extractor(
        _: crate::conversion::private::Token,
    ) -> Option<impl crate::conversion::PureItemExtract<Target = String>> {
        Some(ExactStrExtractor(PhantomData))
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_input() -> TypeInfo {
        Self::type_output()
    }
}

/// Whether `obj` is exactly a Python `str` (not a subclass).
///
/// Used by the borrowed sequence-extraction fast path: extracting a `String`
/// from an exact `str` reads the character data directly and cannot execute
/// arbitrary Python code, so the borrowed item pointer cannot be invalidated
/// mid-extraction.
#[cfg(all(
    not(Py_LIMITED_API),
    not(PyPy),
    not(GraalPy),
    not(Py_GIL_DISABLED)
))]
#[inline(always)]
unsafe fn is_exact_str(obj: *mut crate::ffi::PyObject) -> bool {
    unsafe { (*obj).ob_type == std::ptr::addr_of_mut!(crate::ffi::PyUnicode_Type) }
}

/// Borrowed-item extractor for `String` (see
/// `FromPyObject::borrowed_item_extractor`). Returns `Ok(None)` for anything
/// which is not an exact `str`, so that the caller falls back to the owned
/// (reference-counting) path for that element.
#[cfg(all(
    not(Py_LIMITED_API),
    not(PyPy),
    not(GraalPy),
    not(Py_GIL_DISABLED)
))]
pub(crate) struct ExactStrExtractor(PhantomData<String>);

#[cfg(all(
    not(Py_LIMITED_API),
    not(PyPy),
    not(GraalPy),
    not(Py_GIL_DISABLED)
))]
impl crate::conversion::PureItemExtract for ExactStrExtractor {
    type Target = String;

    #[inline(always)]
    fn extract_pure(item: Borrowed<'_, '_, PyAny>) -> Result<Option<String>, PyErr> {
        // SAFETY (per the `PureItemExtract` contract): for exact `str`
        // inputs the extraction below cannot run arbitrary Python code (the
        // `cast` is a plain type check, and `to_cow` reads the character
        // data).
        if unsafe { is_exact_str(item.as_ptr()) } {
            String::extract(item).map(Some)
        } else {
            Ok(None)
        }
    }
}

impl FromPyObject<'_, '_> for char {
    type Error = PyErr;

    #[cfg(feature = "experimental-inspect")]
    const INPUT_TYPE: PyStaticExpr = PyString::TYPE_HINT;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let s = obj.cast::<PyString>()?.to_cow()?;
        let mut iter = s.chars();
        if let (Some(ch), None) = (iter.next(), iter.next()) {
            Ok(ch)
        } else {
            Err(crate::exceptions::PyValueError::new_err(
                "expected a string of length 1",
            ))
        }
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_input() -> TypeInfo {
        <String>::type_input()
    }
}

#[cfg(test)]
mod tests {
    use crate::types::any::PyAnyMethods;
    use crate::{IntoPyObject, Python};
    use std::borrow::Cow;

    #[test]
    fn test_cow_into_pyobject() {
        Python::attach(|py| {
            let s = "Hello Python";
            let py_string = Cow::Borrowed(s).into_pyobject(py).unwrap();
            assert_eq!(s, py_string.extract::<Cow<'_, str>>().unwrap());
            let py_string = Cow::<str>::Owned(s.into()).into_pyobject(py).unwrap();
            assert_eq!(s, py_string.extract::<Cow<'_, str>>().unwrap());
        })
    }

    #[test]
    fn test_non_bmp() {
        Python::attach(|py| {
            let s = "\u{1F30F}";
            let py_string = s.into_pyobject(py).unwrap();
            assert_eq!(s, py_string.extract::<String>().unwrap());
        })
    }

    #[test]
    fn test_extract_str() {
        Python::attach(|py| {
            let s = "Hello Python";
            let py_string = s.into_pyobject(py).unwrap();

            let s2: Cow<'_, str> = py_string.extract().unwrap();
            assert_eq!(s, s2);
        })
    }

    #[test]
    fn test_extract_char() {
        Python::attach(|py| {
            let ch = '😃';
            let py_string = ch.into_pyobject(py).unwrap();
            let ch2: char = py_string.extract().unwrap();
            assert_eq!(ch, ch2);
        })
    }

    #[test]
    fn test_extract_char_err() {
        Python::attach(|py| {
            let s = "Hello Python";
            let py_string = s.into_pyobject(py).unwrap();
            let err: crate::PyResult<char> = py_string.extract();
            assert!(err
                .unwrap_err()
                .to_string()
                .contains("expected a string of length 1"));
        })
    }

    #[test]
    fn test_string_into_pyobject() {
        Python::attach(|py| {
            let s = "Hello Python";
            let s2 = s.to_owned();
            let s3 = &s2;
            assert_eq!(
                s,
                s3.into_pyobject(py)
                    .unwrap()
                    .extract::<Cow<'_, str>>()
                    .unwrap()
            );
            assert_eq!(
                s,
                s2.into_pyobject(py)
                    .unwrap()
                    .extract::<Cow<'_, str>>()
                    .unwrap()
            );
            assert_eq!(
                s,
                s.into_pyobject(py)
                    .unwrap()
                    .extract::<Cow<'_, str>>()
                    .unwrap()
            );
        })
    }
}
