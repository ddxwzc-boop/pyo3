use std::borrow::Cow;

use crate::{
    exceptions,
    type_object::ClassInfo,
    types::{
        PyAnyMethods, PyNone, PyStringMethods, PyTuple, PyTupleMethods, PyType, PyTypeMethods,
    },
    Borrowed, Bound, IntoPyObjectExt, Py, PyAny, PyErr, PyErrArguments, PyTypeInfo, Python,
};

/// Error that indicates an object was not an instance of a given target type.
#[derive(Debug)]
pub struct CastError<'a, 'py> {
    /// The original object that failed the `isinstance` check.
    from: Borrowed<'a, 'py, PyAny>,
    /// The type we tried (and failed) to convert to.
    /// (see `PyTypeCheck::classinfo_object`)
    classinfo: ClassInfo<'py>,
}

impl<'a, 'py> CastError<'a, 'py> {
    /// Create a new `CastError` representing a failure to interpret a smart pointer to
    /// `from` as a type from the given `classinfo`.
    ///
    /// As with [`PyTypeCheck::classinfo_object`][crate::PyTypeCheck::classinfo_object],
    /// valid `classinfo` values are those which can be used with `isinstance`, such as `type`
    /// objects, tuples of `type` objects, or `typing.Union` instances.
    #[inline]
    pub fn new(from: Borrowed<'a, 'py, PyAny>, classinfo: Bound<'py, PyAny>) -> Self {
        Self {
            from,
            classinfo: ClassInfo::Owned(classinfo),
        }
    }

    /// Like [`CastError::new`], but with a pre-resolved classinfo; the
    /// `ClassInfo::Static` form avoids the reference-count pair on
    /// process-global static type objects.
    #[inline]
    pub(crate) fn new_ref(from: Borrowed<'a, 'py, PyAny>, classinfo: ClassInfo<'py>) -> Self {
        Self { from, classinfo }
    }
}

/// Equivalent to [`CastError`] for operations where the smart pointer cast transfers ownership
/// of the original object.
///
/// The original object can be retrieved using [`into_inner`][CastIntoError::into_inner].
#[derive(Debug)]
pub struct CastIntoError<'py> {
    from: Bound<'py, PyAny>,
    classinfo: ClassInfo<'py>,
}

impl<'py> CastIntoError<'py> {
    /// Create a new `CastError` representing a failure to interpret a smart pointer to
    /// `from` as a type from the given `classinfo`.
    ///
    /// Equivalent to [`CastError::new`] for owned objects.
    #[inline]
    pub fn new(from: Bound<'py, PyAny>, classinfo: Bound<'py, PyAny>) -> Self {
        Self {
            from,
            classinfo: ClassInfo::Owned(classinfo),
        }
    }

    /// Like [`CastIntoError::new`], but with a pre-resolved classinfo; the
    /// `ClassInfo::Static` form avoids the reference-count pair on
    /// process-global static type objects.
    #[inline]
    pub(crate) fn new_ref(from: Bound<'py, PyAny>, classinfo: ClassInfo<'py>) -> Self {
        Self { from, classinfo }
    }

    /// Consumes this `CastIntoError` and returns the original object, allowing continued
    /// use of it after a failed conversion.
    ///
    /// See [`cast_into`][Bound::cast_into] for an example.
    pub fn into_inner(self) -> Bound<'py, PyAny> {
        self.from
    }
}

struct CastErrorArguments {
    from: Py<PyAny>,
    classinfo: Py<PyAny>,
}

impl PyErrArguments for CastErrorArguments {
    fn arguments(self, py: Python<'_>) -> Py<PyAny> {
        format!(
            "{}",
            DisplayCastError {
                from: &self.from.into_bound(py),
                classinfo: &self.classinfo.into_bound(py),
            }
        )
        .into_py_any(py)
        .expect("failed to create Python string")
    }
}

/// Convert `CastError` to Python `TypeError`.
impl std::convert::From<CastError<'_, '_>> for PyErr {
    fn from(err: CastError<'_, '_>) -> PyErr {
        let args = CastErrorArguments {
            from: err.from.to_owned().unbind(),
            classinfo: err.classinfo.into_py(),
        };

        exceptions::PyTypeError::new_err(args)
    }
}

impl std::error::Error for CastError<'_, '_> {}

impl std::fmt::Display for CastError<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        DisplayCastError {
            from: &self.from,
            classinfo: self.classinfo.as_bound(),
        }
        .fmt(f)
    }
}

/// Convert `CastIntoError` to Python `TypeError`.
impl std::convert::From<CastIntoError<'_>> for PyErr {
    fn from(err: CastIntoError<'_>) -> PyErr {
        let args = CastErrorArguments {
            from: err.from.to_owned().unbind(),
            classinfo: err.classinfo.into_py(),
        };

        exceptions::PyTypeError::new_err(args)
    }
}

impl std::error::Error for CastIntoError<'_> {}

impl std::fmt::Display for CastIntoError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        DisplayCastError {
            from: &self.from.to_owned(),
            classinfo: self.classinfo.as_bound(),
        }
        .fmt(f)
    }
}

struct DisplayCastError<'a, 'py> {
    from: &'a Bound<'py, PyAny>,
    classinfo: &'a Bound<'py, PyAny>,
}

impl std::fmt::Display for DisplayCastError<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let to = DisplayClassInfo(self.classinfo);
        if self.from.is_none() {
            write!(f, "'None' is not an instance of '{to}'")
        } else {
            let from = self.from.get_type().qualname();
            let from = from
                .as_ref()
                .map(|name| name.to_string_lossy())
                .unwrap_or(Cow::Borrowed("<failed to extract type name>"));
            write!(f, "'{from}' object is not an instance of '{to}'")
        }
    }
}

struct DisplayClassInfo<'a, 'py>(&'a Bound<'py, PyAny>);

impl std::fmt::Display for DisplayClassInfo<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Ok(t) = self.0.cast::<PyType>() {
            if t.is(PyNone::type_object(t.py())) {
                f.write_str("None")
            } else {
                t.qualname()
                    .map_err(|_| std::fmt::Error)?
                    .to_string_lossy()
                    .fmt(f)
            }
        } else if let Ok(t) = self.0.cast::<PyTuple>() {
            for (i, t) in t.iter().enumerate() {
                if i > 0 {
                    f.write_str(" | ")?;
                }
                write!(f, "{}", DisplayClassInfo(&t))?;
            }
            Ok(())
        } else {
            self.0.fmt(f)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
        PyTypeInfo,
    };

    use super::*;

    #[test]
    fn test_display_cast_error() {
        Python::attach(|py| {
            let obj = PyBool::new(py, true).to_any();
            let classinfo = py.get_type::<PyString>().into_any();
            let err = CastError::new(obj, classinfo);
            assert_eq!(err.to_string(), "'bool' object is not an instance of 'str'");
        })
    }

    #[test]
    fn test_display_cast_error_with_none() {
        Python::attach(|py| {
            let obj = py.None().into_bound(py);
            let classinfo = py.get_type::<PyString>().into_any();
            let err = CastError::new(obj.as_borrowed(), classinfo);
            assert_eq!(err.to_string(), "'None' is not an instance of 'str'");
        })
    }

    #[test]
    fn test_display_cast_error_with_tuple() {
        Python::attach(|py| {
            let obj = PyBool::new(py, true).to_any();
            let classinfo = PyTuple::new(
                py,
                &[
                    py.get_type::<PyString>().into_any(),
                    crate::types::PyNone::type_object(py).into_any(),
                ],
            )
            .unwrap()
            .into_any();
            let err = CastError::new(obj, classinfo);
            assert_eq!(
                err.to_string(),
                "'bool' object is not an instance of 'str | None'"
            );
        })
    }

    #[test]
    fn test_display_cast_into_error() {
        Python::attach(|py| {
            let obj = PyBool::new(py, true).to_any();
            let classinfo = py.get_type::<PyString>().into_any();
            let err = CastIntoError::new(obj.to_owned(), classinfo);
            assert_eq!(err.to_string(), "'bool' object is not an instance of 'str'");
        })
    }

    #[test]
    fn test_pyerr_from_cast_error() {
        Python::attach(|py| {
            let obj = PyBool::new(py, true).to_any();
            let classinfo = py.get_type::<PyString>().into_any();
            let err = CastError::new(obj, classinfo);
            let py_err: PyErr = err.into();
            assert_eq!(
                py_err.to_string(),
                "TypeError: 'bool' object is not an instance of 'str'"
            );
        })
    }

    #[test]
    fn test_pyerr_from_cast_into_error() {
        Python::attach(|py| {
            let obj = PyBool::new(py, true).to_any();
            let classinfo = py.get_type::<PyString>().into_any();
            let err = CastIntoError::new(obj.to_owned(), classinfo);
            let py_err: PyErr = err.into();
            assert_eq!(
                py_err.to_string(),
                "TypeError: 'bool' object is not an instance of 'str'"
            );
        })
    }

    // D3: cast failures on native types hold the static type object's
    // classinfo without taking any reference to it.
    #[cfg(all(not(any(PyPy, GraalPy)), not(all(Py_3_14, Py_LIMITED_API))))]
    #[test]
    fn test_cast_error_static_classinfo_takes_no_reference() {
        Python::attach(|py| {
            use crate::types::PyString;

            let d = PyDict::new(py).into_any();
            let ty_ptr =
                <PyString as crate::PyTypeInfo>::type_object_raw(py).cast::<crate::ffi::PyObject>();
            let rc_before = unsafe { crate::ffi::Py_REFCNT(ty_ptr) };

            let err = d.cast::<PyString>().unwrap_err();
            let rc_alive = unsafe { crate::ffi::Py_REFCNT(ty_ptr) };
            assert_eq!(rc_before, rc_alive);

            drop(err);
            let rc_after = unsafe { crate::ffi::Py_REFCNT(ty_ptr) };
            assert_eq!(rc_before, rc_after);
        })
    }

    // D3 negative control: the owned form must still hold a real reference
    // for non-static classinfo values (a runtime-created heap type).
    #[cfg(all(not(any(PyPy, GraalPy)), not(all(Py_3_14, Py_LIMITED_API))))]
    #[test]
    fn test_cast_error_owned_classinfo_holds_reference() {
        Python::attach(|py| {
            let heap_ty: Bound<'_, PyAny> =
                py.eval(c"type('HeapType', (), {})", None, None).unwrap();
            let d = PyDict::new(py).into_any();
            let rc_before = unsafe { crate::ffi::Py_REFCNT(heap_ty.as_ptr()) };

            let err = CastError::new(d.as_borrowed(), heap_ty.clone());
            let rc_alive = unsafe { crate::ffi::Py_REFCNT(heap_ty.as_ptr()) };
            assert_eq!(rc_before + 1, rc_alive);

            drop(err);
            let rc_after = unsafe { crate::ffi::Py_REFCNT(heap_ty.as_ptr()) };
            assert_eq!(rc_before, rc_after);
        })
    }

    #[test]
    fn test_display_cast_error_borrowed_paths() {
        Python::attach(|py| {
            let d = PyDict::new(py).into_any();
            let s = PyString::new(py, "foobar").into_any();
            let b = PyBool::new(py, true).to_owned().into_any();

            assert_eq!(
                d.cast::<PyString>().unwrap_err().to_string(),
                "'dict' object is not an instance of 'str'"
            );
            assert_eq!(
                s.cast::<PyList>().unwrap_err().to_string(),
                "'str' object is not an instance of 'list'"
            );
            assert_eq!(
                d.cast::<PyFloat>().unwrap_err().to_string(),
                "'dict' object is not an instance of 'float'"
            );
            assert_eq!(
                d.cast::<PyInt>().unwrap_err().to_string(),
                "'dict' object is not an instance of 'int'"
            );
            assert_eq!(
                d.clone().cast_into::<PyString>().unwrap_err().to_string(),
                "'dict' object is not an instance of 'str'"
            );
            // bool is a subclass of int, so exact casts must fail
            assert_eq!(
                b.cast_exact::<PyInt>().unwrap_err().to_string(),
                "'bool' object is not an instance of 'int'"
            );
            assert_eq!(
                b.clone()
                    .cast_into_exact::<PyInt>()
                    .unwrap_err()
                    .to_string(),
                "'bool' object is not an instance of 'int'"
            );
            assert_eq!(
                d.as_borrowed().cast::<PyString>().unwrap_err().to_string(),
                "'dict' object is not an instance of 'str'"
            );
        })
    }

    #[test]
    fn test_pyerr_from_cast_error_static_classinfo_parity() {
        Python::attach(|py| {
            let d = PyDict::new(py).into_any();
            let borrowed: PyErr = d.cast::<PyString>().unwrap_err().into();
            let owned: PyErr =
                CastError::new(d.as_borrowed(), py.get_type::<PyString>().into_any()).into();
            assert_eq!(borrowed.to_string(), owned.to_string());
            assert_eq!(
                borrowed.to_string(),
                "TypeError: 'dict' object is not an instance of 'str'"
            );
        })
    }

    // D3: manual `PyTypeCheck` implementations (weakref's tuple classinfo)
    // keep the default owned behavior.
    #[test]
    fn test_weakref_manual_impl_classinfo_owned_path() {
        Python::attach(|py| {
            use crate::type_object::ClassInfo;
            use crate::types::PyWeakrefProxy;

            let d = PyDict::new(py).into_any();
            let err = d.cast::<PyWeakrefProxy>().unwrap_err();
            assert!(matches!(err.classinfo, ClassInfo::Owned(_)));
            let via_cast = err.to_string();
            let manual = CastError::new(
                d.as_borrowed(),
                <PyWeakrefProxy as crate::PyTypeCheck>::classinfo_object(py),
            )
            .to_string();
            assert_eq!(via_cast, manual);
        })
    }
}
