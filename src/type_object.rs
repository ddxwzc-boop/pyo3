//! Python type object information

use crate::ffi_ptr_ext::FfiPtrExt;
#[cfg(feature = "experimental-inspect")]
use crate::inspect::{type_hint_identifier, PyStaticExpr};
use crate::types::{PyAny, PyType};
use crate::{ffi, Borrowed, Bound, Py, Python};
use std::ptr;

/// `T: PyLayout<U>` represents that `T` is a concrete representation of `U` in the Python heap.
/// E.g., `PyClassObject` is a concrete representation of all `pyclass`es, and `ffi::PyObject`
/// is of `PyAny`.
///
/// This trait is intended to be used internally.
///
/// # Safety
///
/// This trait must only be implemented for types which represent valid layouts of Python objects.
pub unsafe trait PyLayout<T> {}

/// `T: PySizedLayout<U>` represents that `T` is not a instance of
/// [`PyVarObject`](https://docs.python.org/3/c-api/structures.html#c.PyVarObject).
///
/// In addition, that `T` is a concrete representation of `U`.
pub trait PySizedLayout<T>: PyLayout<T> + Sized {}

/// Python type information.
/// All Python native types (e.g., `PyDict`) and `#[pyclass]` structs implement this trait.
///
/// This trait is marked unsafe because:
///  - specifying the incorrect layout can lead to memory errors
///  - the return value of type_object must always point to the same PyTypeObject instance
///
/// It is safely implemented by the `pyclass` macro.
///
/// # Safety
///
/// Implementations must provide an implementation for `type_object_raw` which infallibly produces a
/// non-null pointer to the corresponding Python type object.
///
/// `is_type_of` must only return true for objects which can safely be treated as instances of `Self`.
///
/// `is_exact_type_of` must only return true for objects whose type is exactly `Self`.
pub unsafe trait PyTypeInfo: Sized {
    /// Class name.
    #[deprecated(
        since = "0.28.0",
        note = "prefer using `::type_object(py).name()` to get the correct runtime value"
    )]
    const NAME: &'static str;

    /// Module name, if any.
    #[deprecated(
        since = "0.28.0",
        note = "prefer using `::type_object(py).module()` to get the correct runtime value"
    )]
    const MODULE: Option<&'static str>;

    /// Provides the full python type as a type hint.
    #[cfg(feature = "experimental-inspect")]
    const TYPE_HINT: PyStaticExpr = type_hint_identifier!("_typeshed", "Incomplete");

    /// Returns the PyTypeObject instance for this type.
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject;

    /// Returns the safe abstraction over the type object.
    #[inline]
    fn type_object(py: Python<'_>) -> Bound<'_, PyType> {
        // Making the borrowed object `Bound` is necessary for soundness reasons. It's an extreme
        // edge case, but arbitrary Python code _could_ change the __class__ of an object and cause
        // the type object to be freed.
        //
        // By making `Bound` we assume ownership which is then safe against races.
        unsafe {
            Self::type_object_raw(py)
                .cast::<ffi::PyObject>()
                .assume_borrowed_unchecked(py)
                .to_owned()
                .cast_into_unchecked()
        }
    }

    /// Checks if `object` is an instance of this type or a subclass of this type.
    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        unsafe { ffi::PyObject_TypeCheck(object.as_ptr(), Self::type_object_raw(object.py())) != 0 }
    }

    /// Checks if `object` is an instance of this type.
    #[inline]
    fn is_exact_type_of(object: &Bound<'_, PyAny>) -> bool {
        unsafe {
            ptr::eq(
                ffi::Py_TYPE(object.as_ptr()),
                Self::type_object_raw(object.py()),
            )
        }
    }

    /// Whether `type_object_raw` produces a process-global static type object
    /// which is never deallocated for the process lifetime (e.g. the builtin
    /// types such as `PyUnicode_Type`). This allows cast errors to hold the
    /// type object as classinfo without taking a reference to it.
    ///
    /// # Safety
    ///
    /// This may only be overridden to `true` if the pointer returned by
    /// `type_object_raw` is never deallocated for the process lifetime;
    /// otherwise the borrowed classinfo held by cast errors would dangle.
    ///
    /// In-tree this is only set to `true` by the
    /// `pyobject_native_type_info!` macro for the
    /// `pyobject_native_static_type_object!` form, and is forced to `false`
    /// on PyPy and GraalPy.
    #[doc(hidden)]
    const TYPE_OBJECT_IS_STATIC: bool = false;
}

/// The classinfo payload of a cast error, in either a zero-refcount borrowed
/// form or an owned form.
///
/// `Static` is used for classinfo values which are process-global static type
/// objects (e.g. `PyUnicode_Type`); those are never deallocated, so holding
/// them requires no reference count. `Owned` covers everything else (heap
/// types, tuples of types, ...).
///
/// This type is intended for internal use by the cast machinery; it is public
/// only because it appears in a hidden trait method signature.
#[doc(hidden)]
#[derive(Debug)]
pub enum ClassInfo<'py> {
    /// Classinfo is a process-global static type object; held as a borrow.
    Static(Borrowed<'py, 'py, PyAny>),
    /// Classinfo is a dynamically-allocated object; held as an owned
    /// reference.
    Owned(Bound<'py, PyAny>),
}

impl<'py> ClassInfo<'py> {
    /// Zero-cost view of the classinfo as `&Bound`.
    #[inline]
    pub(crate) fn as_bound(&self) -> &Bound<'py, PyAny> {
        match self {
            ClassInfo::Static(borrowed) => borrowed,
            ClassInfo::Owned(bound) => bound,
        }
    }

    /// Detach the classinfo from the `'py` lifetime; the `Static` form pays
    /// its (single) reference count only at this point.
    #[inline]
    pub(crate) fn into_py(self) -> Py<PyAny> {
        match self {
            ClassInfo::Static(borrowed) => borrowed.to_owned().unbind(),
            ClassInfo::Owned(bound) => bound.unbind(),
        }
    }
}

/// Implemented by types which can be used as a concrete Python type inside `Py<T>` smart pointers.
///
/// # Safety
///
/// This trait is used to determine whether [`Bound::cast`] and similar functions can safely cast
/// to a concrete type. The implementor is responsible for ensuring that `type_check` only returns
/// true for objects which can safely be treated as Python instances of `Self`.
pub unsafe trait PyTypeCheck {
    /// Name of self. This is used in error messages, for example.
    #[deprecated(
        since = "0.27.0",
        note = "Use ::classinfo_object() instead and format the type name at runtime. Note that using built-in cast features is often better than manual PyTypeCheck usage."
    )]
    const NAME: &'static str;

    /// Provides the full python type of the allowed values as a Python type hint.
    #[cfg(feature = "experimental-inspect")]
    const TYPE_HINT: PyStaticExpr;

    /// Checks if `object` is an instance of `Self`, which may include a subtype.
    ///
    /// This should be equivalent to the Python expression `isinstance(object, Self)`.
    fn type_check(object: &Bound<'_, PyAny>) -> bool;

    /// Returns the expected type as a possible argument for the `isinstance` and `issubclass` function.
    ///
    /// It may be a single type or a tuple of types.
    fn classinfo_object(py: Python<'_>) -> Bound<'_, PyAny>;

    /// Returns the classinfo used by cast errors, in a form which avoids the
    /// reference-count pair for process-global static type objects.
    ///
    /// # Safety
    ///
    /// Overriders may only return [`ClassInfo::Static`] for classinfo values
    /// which will not be deallocated during the current Python session.
    #[doc(hidden)]
    fn classinfo_object_ref(py: Python<'_>) -> ClassInfo<'_> {
        ClassInfo::Owned(Self::classinfo_object(py))
    }
}

unsafe impl<T> PyTypeCheck for T
where
    T: PyTypeInfo,
{
    #[allow(deprecated)]
    const NAME: &'static str = T::NAME;

    #[cfg(feature = "experimental-inspect")]
    const TYPE_HINT: PyStaticExpr = <T as PyTypeInfo>::TYPE_HINT;

    #[inline]
    fn type_check(object: &Bound<'_, PyAny>) -> bool {
        T::is_type_of(object)
    }

    #[inline]
    fn classinfo_object(py: Python<'_>) -> Bound<'_, PyAny> {
        T::type_object(py).into_any()
    }

    #[inline]
    fn classinfo_object_ref(py: Python<'_>) -> ClassInfo<'_> {
        if T::TYPE_OBJECT_IS_STATIC {
            // SAFETY: `TYPE_OBJECT_IS_STATIC` is only `true` when
            // `type_object_raw` produces a process-global static type object
            // which is never deallocated, so the borrow cannot dangle for any
            // `'py` session.
            ClassInfo::Static(unsafe {
                T::type_object_raw(py)
                    .cast::<ffi::PyObject>()
                    .assume_borrowed_unchecked(py)
            })
        } else {
            ClassInfo::Owned(T::type_object(py).into_any())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::type_object::ClassInfo;
    use crate::types::{PyFloat, PyIterator, PySequence, PyString, PyWeakrefReference};
    use crate::{PyTypeCheck, PyTypeInfo, Python};

    #[test]
    fn test_type_object_is_static_flags() {
        // static marker form: true on CPython, false on PyPy / GraalPy
        assert_eq!(
            <PyString as PyTypeInfo>::TYPE_OBJECT_IS_STATIC,
            cfg!(not(any(PyPy, GraalPy)))
        );
        // `pyobject_native_type!` static arm (layout variant)
        assert_eq!(
            <PyFloat as PyTypeInfo>::TYPE_OBJECT_IS_STATIC,
            cfg!(not(any(PyPy, GraalPy)))
        );
        // marker + `#module=` combination path
        assert_eq!(
            <PyWeakrefReference as PyTypeInfo>::TYPE_OBJECT_IS_STATIC,
            cfg!(not(any(PyPy, GraalPy)))
        );
        // import-based typeobjects and manual impls stay false
        const {
            assert!(!<PyIterator as PyTypeInfo>::TYPE_OBJECT_IS_STATIC);
            assert!(!<PySequence as PyTypeInfo>::TYPE_OBJECT_IS_STATIC);
        }
    }

    #[test]
    fn test_classinfo_object_ref_variants() {
        Python::attach(|py| {
            match <PyString as PyTypeCheck>::classinfo_object_ref(py) {
                ClassInfo::Static(borrowed) => assert_eq!(
                    borrowed.as_ptr(),
                    <PyString as PyTypeInfo>::type_object_raw(py).cast()
                ),
                ClassInfo::Owned(_) => panic!("PyString classinfo should be the static form"),
            }
            assert!(matches!(
                <PyIterator as PyTypeCheck>::classinfo_object_ref(py),
                ClassInfo::Owned(_)
            ));
            assert!(matches!(
                <PySequence as PyTypeCheck>::classinfo_object_ref(py),
                ClassInfo::Owned(_)
            ));
        })
    }
}
