//! Reentrance-free extraction for the safe extracting iterators.
//!
//! [`FastExtract`] designates Rust numeric types whose conversion from an
//! *exact* native Python object (`int` / `float`) is a pure field read: it
//! cannot call into the interpreter, allocate, or raise. This is what lets
//! [`crate::types::PyListMethods::iter_extract`] and
//! [`crate::types::PyDictMethods::iter_extract`] iterate without taking a
//! reference count per item while remaining a **safe** API:
//!
//! - the item pointer is only held unpinned across the pure reads inside one
//!   iterator step, ending before the first operation which could run Python
//!   code (so nothing can free it: on the GIL build no other mutator can run;
//!   on the free-threaded build the container's critical section is held);
//! - every object which cannot take that path (subclasses, custom
//!   `__index__` / `__float__` implementors, anything else) is *pinned* — its
//!   reference count incremented — **before** the ordinary extraction
//!   machinery runs, so Python code which reenters during the conversion
//!   cannot free it.
//!
//! The trait is sealed: pyo3 implements it only for the native integer and
//! float types, where the "pure read" property holds by construction. On the
//! limited API the fast path is unavailable and everything is pinned.

use crate::ffi;
use crate::instance::Bound;
use crate::Python;

use crate::ffi_ptr_ext::FfiPtrExt;

/// Marker for types which can be extracted from an *exact* native Python
/// object by pure field reads: no interpreter calls, no allocation, no
/// exceptions.
///
/// This is a sealed trait implemented by pyo3 for the native integer and
/// float types only. It enables the safe zero-refcount iteration of
/// [`PyListMethods::iter_extract`][crate::types::PyListMethods::iter_extract]
/// and [`PyDictMethods::iter_extract`][crate::types::PyDictMethods::iter_extract]:
/// the returned value needs no lifetime tie to the container because no
/// pointer escapes the iterator step.
///
/// [`iter_extract` (list)]: crate::types::PyListMethods::iter_extract
pub trait FastExtract: crate::sealed::Sealed + Sized {
    /// Attempts to extract `Self` from `obj` without a reference-count change
    /// and without any operation which could run Python code.
    ///
    /// Returns `None` unless `obj` is an exact native instance and the value
    /// is available by pure reads; the caller must then pin `obj` (increment
    /// its reference count) before extracting it through
    /// [`FromPyObject`](crate::FromPyObject) ordinarily.
    ///
    /// # Safety
    ///
    /// - `obj` must point to a live Python object.
    /// - Implementations must uphold the reentrance-freedom contract above;
    ///   the caller relies on it to hold `obj` unpinned across this call.
    #[cfg(not(Py_LIMITED_API))]
    unsafe fn fast_extract(obj: *mut ffi::PyObject) -> Option<Self>;

    /// Limited API: the pure-read path is unavailable; everything is pinned.
    #[cfg(Py_LIMITED_API)]
    #[inline]
    unsafe fn fast_extract(_obj: *mut ffi::PyObject) -> Option<Self> {
        None
    }
}

/// The outcome of one iterator step's phase 1: either the value was read by
/// pure field reads (`Fast`), or the object was pinned (`Pinned`) and must be
/// extracted through the ordinary machinery.
pub(crate) enum FastOrPinned<'py, T> {
    Fast(T),
    Pinned(Bound<'py, crate::PyAny>),
}

/// Phase 1 of one iterator step: read the value if the fast path applies,
/// otherwise pin the object, with no interpreter call in between.
///
/// # Safety
///
/// - `obj` must point to a live Python object.
/// - The caller must not use `obj` again after this function returns.
#[inline]
pub(crate) unsafe fn fast_or_pin<'py, T: FastExtract>(
    py: Python<'py>,
    obj: *mut ffi::PyObject,
) -> FastOrPinned<'py, T> {
    // Safety: the caller guarantees `obj` is live, and `fast_extract` cannot
    // call into the interpreter or allocate, so the unpinned window below
    // cannot observe the object being freed.
    if let Some(value) = unsafe { T::fast_extract(obj) } {
        FastOrPinned::Fast(value)
    } else {
        // Safety: `obj` is a live object; pinning it (Py_INCREF) before the
        // first operation which can run Python code keeps it alive however
        // the container is mutated afterwards.
        FastOrPinned::Pinned(unsafe { obj.assume_borrowed_unchecked(py) }.to_owned())
    }
}

#[cfg(not(Py_LIMITED_API))]
macro_rules! int_fast_extract {
    ($($t:ty)*) => {$(
        impl FastExtract for $t {
            #[inline]
            unsafe fn fast_extract(obj: *mut ffi::PyObject) -> Option<Self> {
                // Two gated pure operations: the exact-type check, then the
                // digit conversion. On an exact `int`,
                // `PyLong_AsLongAndOverflow` reports out-of-`c_long`-range
                // through the flag instead of raising, so this path cannot
                // allocate or call into the interpreter. `c_long` is 32-bit
                // on some platforms; values outside its range fall back to
                // the pinned path, which is correct everywhere.
                if unsafe { ffi::PyLong_CheckExact(obj) } == 0 {
                    return None;
                }
                let mut overflow: std::ffi::c_int = 0;
                let value =
                    unsafe { ffi::PyLong_AsLongAndOverflow(obj, &mut overflow) };
                if overflow != 0 {
                    return None;
                }
                <$t>::try_from(value).ok()
            }
        }
    )*};
}

#[cfg(not(Py_LIMITED_API))]
int_fast_extract!(i8 i16 i32 i64 isize u8 u16 u32 u64 usize);

/// Limited API: the fast path is unavailable, so the types implement the
/// trait with the default (always `None`) method and everything is pinned.
#[cfg(Py_LIMITED_API)]
macro_rules! fallback_fast_extract {
    ($($t:ty)*) => {$(
        impl FastExtract for $t {}
    )*};
}
#[cfg(Py_LIMITED_API)]
fallback_fast_extract!(i8 i16 i32 i64 isize u8 u16 u32 u64 usize f32 f64);

#[cfg(not(Py_LIMITED_API))]
impl FastExtract for f64 {
    #[inline]
    unsafe fn fast_extract(obj: *mut ffi::PyObject) -> Option<Self> {
        if unsafe { ffi::PyFloat_CheckExact(obj) } == 0 {
            return None;
        }
        // On an exact `float` this is a field read (the same one the
        // ordinary `f64` extraction takes).
        Some(unsafe { ffi::PyFloat_AsDouble(obj) })
    }
}

#[cfg(not(Py_LIMITED_API))]
impl FastExtract for f32 {
    #[inline]
    unsafe fn fast_extract(obj: *mut ffi::PyObject) -> Option<Self> {
        if unsafe { ffi::PyFloat_CheckExact(obj) } == 0 {
            return None;
        }
        // On an exact `float` this is a field read; the `as f32` narrowing
        // matches the ordinary `f32` extraction semantics.
        Some(unsafe { ffi::PyFloat_AsDouble(obj) } as f32)
    }
}
