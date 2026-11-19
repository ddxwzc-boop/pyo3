//! Defines how Python calls are dispatched, see [`PyCallArgs`].for more information.

use crate::ffi_ptr_ext::FfiPtrExt as _;
#[cfg(all(Py_3_9, not(any(PyPy, GraalPy, Py_LIMITED_API)), not(Py_3_13)))]
use crate::types::PyTupleMethods as _;
use crate::types::{PyAnyMethods as _, PyDict, PyString, PyTuple};
use crate::{ffi, Borrowed, Bound, IntoPyObjectExt as _, Py, PyAny, PyResult};

pub(crate) mod private {
    use super::*;

    pub trait Sealed {}

    impl Sealed for () {}
    impl Sealed for Bound<'_, PyTuple> {}
    impl Sealed for &'_ Bound<'_, PyTuple> {}
    impl Sealed for Py<PyTuple> {}
    impl Sealed for &'_ Py<PyTuple> {}
    impl Sealed for Borrowed<'_, '_, PyTuple> {}
    pub struct Token;
}

/// This trait marks types that can be used as arguments to Python function
/// calls.
///
/// This trait is currently implemented for Rust tuple (up to a size of 12),
/// [`Bound<'py, PyTuple>`] and [`Py<PyTuple>`]. Custom types that are
/// convertible to `PyTuple` via `IntoPyObject` need to do so before passing it
/// to `call`.
///
/// This trait is not intended to used by downstream crates directly. As such it
/// has no publicly available methods and cannot be implemented outside of
/// `pyo3`. The corresponding public API is available through [`call`]
/// ([`call0`], [`call1`] and friends) on [`PyAnyMethods`].
///
/// # What is `PyCallArgs` used for?
/// `PyCallArgs` is used internally in `pyo3` to dispatch the Python calls in
/// the most optimal way for the current build configuration. Certain types,
/// such as Rust tuples, do allow the usage of a faster calling convention of
/// the Python interpreter (if available). More types that may take advantage
/// from this may be added in the future.
///
/// [`call0`]: crate::types::PyAnyMethods::call0
/// [`call1`]: crate::types::PyAnyMethods::call1
/// [`call`]: crate::types::PyAnyMethods::call
/// [`PyAnyMethods`]: crate::types::PyAnyMethods
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot used as a Python `call` argument",
    note = "`PyCallArgs` is implemented for Rust tuples, `Bound<'py, PyTuple>` and `Py<PyTuple>`",
    note = "if your type is convertible to `PyTuple` via `IntoPyObject`, call `<arg>.into_pyobject(py)` manually",
    note = "if you meant to pass the type as a single argument, wrap it in a 1-tuple, `(<arg>,)`"
)]
pub trait PyCallArgs<'py>: Sized + private::Sealed {
    #[doc(hidden)]
    fn call(
        self,
        function: Borrowed<'_, 'py, PyAny>,
        kwargs: Borrowed<'_, 'py, PyDict>,
        token: private::Token,
    ) -> PyResult<Bound<'py, PyAny>>;

    #[doc(hidden)]
    fn call_positional(
        self,
        function: Borrowed<'_, 'py, PyAny>,
        token: private::Token,
    ) -> PyResult<Bound<'py, PyAny>>;

    #[doc(hidden)]
    fn call_method_positional(
        self,
        object: Borrowed<'_, 'py, PyAny>,
        method_name: Borrowed<'_, 'py, PyString>,
        _: private::Token,
    ) -> PyResult<Bound<'py, PyAny>> {
        object
            .getattr(method_name)
            .and_then(|method| method.call1(self))
    }
}

/// Fast dispatch for calls with keyword arguments on CPython 3.9-3.12.
///
/// When the callable supports vectorcall, the kwargs dict is unpacked on the
/// Rust side into the vectorcall (stack, kwnames) form, and exact `str` keys
/// are interned so that the callee's parameter-name matching hits CPython's
/// interned-pointer comparison fast path instead of falling back to
/// `PyObject_RichCompareBool` per parameter name.
///
/// Returns `None` when the callable has no vectorcall support (or the kwargs
/// dict is empty): the caller must then use the regular dict-based call,
/// which preserves the identity contract for `tp_call`-based callees such
/// as `METH_VARARGS` functions.
#[cfg(all(Py_3_9, not(any(PyPy, GraalPy, Py_LIMITED_API)), not(Py_3_13)))]
pub(crate) fn call_with_interned_kwargs<'py>(
    function: Borrowed<'_, 'py, PyAny>,
    positional: &[Bound<'py, PyAny>],
    kwargs: Borrowed<'_, 'py, PyDict>,
) -> Option<PyResult<Bound<'py, PyAny>>> {
    // SAFETY: `kwargs` is a live dict object; reading the size field only.
    let nkwargs = unsafe { (*kwargs.as_ptr().cast::<ffi::PyDictObject>()).ma_used };
    if nkwargs == 0 {
        // An empty kwargs dict has nothing to intern; CPython's own call path
        // already short-circuits this case, so leave it to the regular path.
        return None;
    }
    // SAFETY: `function` is a live object for the whole call; the probe only
    // reads type metadata and the vectorcall slot, following CPython's own
    // `_PyVectorcall_Function` rules.
    let vectorcall = unsafe { ffi::PyVectorcall_Function(function.as_ptr())? };
    // SAFETY: see `call_vectorcall_kwargs`.
    Some(unsafe { call_vectorcall_kwargs(function, vectorcall, positional, kwargs) })
}

/// Unpack `kwargs` into a vectorcall stack and an interned-kwnames tuple and
/// invoke `vectorcall` directly, mirroring CPython's `_PyStack_UnpackDict`
/// (same reference-counting contract: the stack and kwnames are only
/// borrowed for the duration of the call; the offset slot at `stack[0]` may
/// be written by the callee and is never read here).
///
/// # Safety
///
/// `vectorcall` must be the vectorcall function of `function` (as probed by
/// `PyVectorcall_Function`), and no exception may be pending (the same
/// precondition as `PyObject_Call`).
#[cfg(all(Py_3_9, not(any(PyPy, GraalPy, Py_LIMITED_API)), not(Py_3_13)))]
unsafe fn call_vectorcall_kwargs<'py>(
    function: Borrowed<'_, 'py, PyAny>,
    vectorcall: ffi::vectorcallfunc,
    positional: &[Bound<'py, PyAny>],
    kwargs: Borrowed<'_, 'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = function.py();
    unsafe {
        let kwargs_ptr = kwargs.as_ptr();
        let nkwargs = (*kwargs_ptr.cast::<ffi::PyDictObject>()).ma_used as usize;
        let nargs = positional.len();

        // Build the kwnames tuple first, so that a failed allocation has
        // nothing to clean up yet.
        let kwnames: Bound<'py, PyTuple> = ffi::PyTuple_New(nkwargs as ffi::Py_ssize_t)
            .assume_owned_or_err(py)?
            .cast_into_unchecked();
        let kwnames_ptr = kwnames.as_ptr();

        // Combined stack: [PY_VECTORCALL_ARGUMENTS_OFFSET slot, positional
        // arguments..., keyword values...].
        let mut stack: Vec<*mut ffi::PyObject> = Vec::with_capacity(1 + nargs + nkwargs);
        stack.push(std::ptr::null_mut());
        for arg in positional {
            ffi::Py_INCREF(arg.as_ptr());
            stack.push(arg.as_ptr());
        }

        // The dictionary cannot change size during this loop: no step runs
        // user code. In particular exact `str` keys are interned (their hash
        // is cached because they are dict keys, and interned-dict equality
        // on exact strings is a pure C comparison), while `str` subclass
        // keys are deliberately passed through un-interned, because interning
        // could run user code via a custom `__hash__` / `__eq__` — something
        // CPython's own call path never does.
        let mut pos: ffi::Py_ssize_t = 0;
        let mut key: *mut ffi::PyObject = std::ptr::null_mut();
        let mut value: *mut ffi::PyObject = std::ptr::null_mut();
        let mut i = 0usize;
        while ffi::PyDict_Next(kwargs_ptr, &mut pos, &mut key, &mut value) != 0 {
            let key = if ffi::PyUnicode_CheckExact(key) != 0 {
                // Take a reference of our own and intern it;
                // `PyUnicode_InternInPlace` transfers that reference to the
                // interned twin (which may be this very object).
                let mut owned = key;
                ffi::Py_INCREF(owned);
                ffi::PyUnicode_InternInPlace(&mut owned);
                owned
            } else if ffi::PyUnicode_Check(key) != 0 {
                ffi::Py_INCREF(key);
                key
            } else {
                // CPython's `_PyStack_UnpackDict` rejects these with the
                // same TypeError; release everything borrowed so far.
                for &ptr in &stack[1..] {
                    ffi::Py_DECREF(ptr);
                }
                drop(stack);
                drop(kwnames);
                return Err(crate::exceptions::PyTypeError::new_err(
                    "keywords must be strings",
                ));
            };
            // steals the reference held in `key`
            ffi::PyTuple_SET_ITEM(kwnames_ptr, i as ffi::Py_ssize_t, key);
            ffi::Py_INCREF(value);
            stack.push(value);
            i += 1;
        }

        let result = vectorcall(
            function.as_ptr(),
            stack.as_ptr().add(1),
            nargs + ffi::PY_VECTORCALL_ARGUMENTS_OFFSET,
            kwnames_ptr,
        );

        // The vectorcall convention borrows the stack and kwnames only for
        // the duration of the call; release everything the same way
        // CPython's `_PyStack_UnpackDict_Free` does.
        for &ptr in &stack[1..] {
            ffi::Py_DECREF(ptr);
        }
        drop(stack);
        drop(kwnames);

        // Mirror CPython's `_Py_CheckFunctionResult` for callees that violate
        // the vectorcall result contract (both directions). The call paths
        // this fast path replaces all routed through that guard.
        let exception_set = unsafe { !ffi::PyErr_Occurred().is_null() };
        if result.is_null() {
            if !exception_set {
                // NULL without an error set: synthesize the `SystemError`
                // CPython's wrappers raise, so the fetch below cannot fail.
                unsafe {
                    ffi::PyErr_SetString(
                        ffi::PyExc_SystemError,
                        c"callable returned NULL without setting an exception".as_ptr(),
                    );
                }
            }
        } else if exception_set {
            // A non-NULL result with an error set: drop the result and
            // raise a `SystemError` with the pending error as its cause,
            // exactly as the replaced CPython wrappers did.
            unsafe { ffi::Py_DECREF(result) };
            let err = crate::exceptions::PySystemError::new_err(
                "callable returned a result with an exception set",
            );
            if let Some(cause) = crate::PyErr::take(py) {
                err.set_cause(py, Some(cause));
            }
            return Err(err);
        }

        result.assume_owned_or_err(py)
    }
}

impl<'py> PyCallArgs<'py> for () {
    fn call(
        self,
        function: Borrowed<'_, 'py, PyAny>,
        kwargs: Borrowed<'_, 'py, PyDict>,
        token: private::Token,
    ) -> PyResult<Bound<'py, PyAny>> {
        let args = self.into_pyobject_or_pyerr(function.py())?;
        args.call(function, kwargs, token)
    }

    fn call_positional(
        self,
        function: Borrowed<'_, 'py, PyAny>,
        token: private::Token,
    ) -> PyResult<Bound<'py, PyAny>> {
        let args = self.into_pyobject_or_pyerr(function.py())?;
        args.call_positional(function, token)
    }
}

impl<'py> PyCallArgs<'py> for Bound<'py, PyTuple> {
    #[inline]
    fn call(
        self,
        function: Borrowed<'_, 'py, PyAny>,
        kwargs: Borrowed<'_, 'py, PyDict>,
        token: private::Token,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.as_borrowed().call(function, kwargs, token)
    }

    #[inline]
    fn call_positional(
        self,
        function: Borrowed<'_, 'py, PyAny>,
        token: private::Token,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.as_borrowed().call_positional(function, token)
    }
}

impl<'py> PyCallArgs<'py> for &'_ Bound<'py, PyTuple> {
    #[inline]
    fn call(
        self,
        function: Borrowed<'_, 'py, PyAny>,
        kwargs: Borrowed<'_, 'py, PyDict>,
        token: private::Token,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.as_borrowed().call(function, kwargs, token)
    }

    #[inline]
    fn call_positional(
        self,
        function: Borrowed<'_, 'py, PyAny>,
        token: private::Token,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.as_borrowed().call_positional(function, token)
    }
}

impl<'py> PyCallArgs<'py> for Py<PyTuple> {
    #[inline]
    fn call(
        self,
        function: Borrowed<'_, 'py, PyAny>,
        kwargs: Borrowed<'_, 'py, PyDict>,
        token: private::Token,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.bind_borrowed(function.py())
            .call(function, kwargs, token)
    }

    #[inline]
    fn call_positional(
        self,
        function: Borrowed<'_, 'py, PyAny>,
        token: private::Token,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.bind_borrowed(function.py())
            .call_positional(function, token)
    }
}

impl<'py> PyCallArgs<'py> for &'_ Py<PyTuple> {
    #[inline]
    fn call(
        self,
        function: Borrowed<'_, 'py, PyAny>,
        kwargs: Borrowed<'_, 'py, PyDict>,
        token: private::Token,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.bind_borrowed(function.py())
            .call(function, kwargs, token)
    }

    #[inline]
    fn call_positional(
        self,
        function: Borrowed<'_, 'py, PyAny>,
        token: private::Token,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.bind_borrowed(function.py())
            .call_positional(function, token)
    }
}

impl<'py> PyCallArgs<'py> for Borrowed<'_, 'py, PyTuple> {
    #[inline]
    fn call(
        self,
        function: Borrowed<'_, 'py, PyAny>,
        kwargs: Borrowed<'_, 'py, PyDict>,
        _: private::Token,
    ) -> PyResult<Bound<'py, PyAny>> {
        #[cfg(all(Py_3_9, not(any(PyPy, GraalPy, Py_LIMITED_API)), not(Py_3_13)))]
        {
            // Fast path with interned kwnames; falls back to `PyObject_Call`
            // when the callable has no vectorcall, preserving the identity
            // contract for `tp_call`-based callees.
            if let Some(result) = call_with_interned_kwargs(function, self.as_slice(), kwargs) {
                return result;
            }
        }
        unsafe {
            ffi::PyObject_Call(function.as_ptr(), self.as_ptr(), kwargs.as_ptr())
                .assume_owned_or_err(function.py())
        }
    }

    #[inline]
    fn call_positional(
        self,
        function: Borrowed<'_, 'py, PyAny>,
        _: private::Token,
    ) -> PyResult<Bound<'py, PyAny>> {
        unsafe {
            ffi::PyObject_Call(function.as_ptr(), self.as_ptr(), std::ptr::null_mut())
                .assume_owned_or_err(function.py())
        }
    }
}

#[cfg(test)]
#[cfg(feature = "macros")]
mod tests {
    use crate::{
        pyfunction,
        types::{PyDict, PyTuple},
        Py,
    };

    #[pyfunction(signature = (*args, **kwargs), crate = "crate")]
    fn args_kwargs(
        args: Py<PyTuple>,
        kwargs: Option<Py<PyDict>>,
    ) -> (Py<PyTuple>, Option<Py<PyDict>>) {
        (args, kwargs)
    }

    #[pyfunction(crate = "crate")]
    fn kwargs_named(a: i64, b: i64) -> i64 {
        a * 10 + b
    }

    #[test]
    fn test_call_kwargs_interned_dispatch() {
        use crate::types::{any::PyAnyMethods, dict::PyDictMethods, IntoPyDict, PyDict};
        use crate::Python;

        Python::attach(|py| {
            // A pure-Python callee: parameter-name matching is exactly the
            // machinery the interned-kwnames fast path targets.
            let f = py
                .eval(c"lambda a, b, c, d, e: (a, b, c, d, e)", None, None)
                .unwrap();
            let kwargs = [("d", 1), ("e", 2)].into_py_dict(py).unwrap();

            // Rust tuple arguments (vectorcall fast path).
            let bound: (i32, i32, i32, i32, i32) =
                f.call((3, 4, 5), Some(&kwargs)).unwrap().extract().unwrap();
            assert_eq!(bound, (3, 4, 5, 1, 2));

            // Tuple arguments (the `PyObject_Call` site with the same probe).
            let args = PyTuple::new(py, [3, 4, 5]).unwrap();
            let bound: (i32, i32, i32, i32, i32) = f
                .call(args.as_borrowed(), Some(&kwargs))
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(bound, (3, 4, 5, 1, 2));

            // A non-string key is rejected exactly like CPython's own
            // `_PyStack_UnpackDict` does.
            let bad = PyDict::new(py);
            bad.set_item(1, 2).unwrap();
            let err = f.call((3, 4, 5), Some(&bad)).unwrap_err();
            assert_eq!(err.to_string(), "TypeError: keywords must be strings");

            // Duplicate and unexpected keyword arguments keep being reported
            // by the callee with CPython's own messages.
            let dup = [("a", 9)].into_py_dict(py).unwrap();
            let err = f.call((3, 4, 5), Some(&dup)).unwrap_err();
            assert!(err.to_string().contains("multiple values"));

            let unexpected = [("zz", 1)].into_py_dict(py).unwrap();
            let err = f.call((3, 4, 5), Some(&unexpected)).unwrap_err();
            assert!(err.to_string().contains("unexpected keyword argument"));

            // An explicitly-passed empty kwargs dict is left to the regular
            // path (nothing to intern); it must keep binding fine.
            let g = py
                .eval(c"lambda a, b, c, d=9, e=9: (a, b, c, d, e)", None, None)
                .unwrap();
            let empty = PyDict::new(py);
            let bound: (i32, i32, i32, i32, i32) =
                g.call((3, 4, 5), Some(&empty)).unwrap().extract().unwrap();
            assert_eq!(bound, (3, 4, 5, 9, 9));

            // A `str` subclass key is passed through un-interned and still
            // binds by value.
            let locals = PyDict::new(py);
            py.run(
                c"class S(str):
    pass
d = S('d')
e = S('e')",
                None,
                Some(&locals),
            )
            .unwrap();
            let sub = PyDict::new(py);
            sub.set_item(locals.get_item("d").unwrap().unwrap(), 1)
                .unwrap();
            sub.set_item(locals.get_item("e").unwrap().unwrap(), 2)
                .unwrap();
            let bound: (i32, i32, i32, i32, i32) =
                f.call((3, 4, 5), Some(&sub)).unwrap().extract().unwrap();
            assert_eq!(bound, (3, 4, 5, 1, 2));

            // `call_method` with kwargs shares the same dispatch after the
            // attribute lookup.
            let locals = PyDict::new(py);
            py.run(
                c"class K:\n    def m(self, a, b, c, d, e):\n        return (a, b, c, d, e)\nobj = K()",
                None,
                Some(&locals),
            )
            .unwrap();
            let obj = locals.get_item("obj").unwrap().unwrap();
            let bound: (i32, i32, i32, i32, i32) = obj
                .call_method("m", (3, 4, 5), Some(&kwargs))
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(bound, (3, 4, 5, 1, 2));
        })
    }

    #[test]
    fn test_call_kwargs_pyo3_fastcall_function() {
        use crate::types::any::PyAnyMethods;
        use crate::types::IntoPyDict;
        use crate::{wrap_pyfunction, Python};

        Python::attach(|py| {
            // A pyo3 `#[pyfunction]` is METH_FASTCALL | METH_KEYWORDS: the
            // fast path hands it the interned kwnames tuple directly and its
            // own `&str` matching must keep binding correctly.
            let f = wrap_pyfunction!(kwargs_named, py).unwrap();

            let kwargs = [("a", 1), ("b", 2)].into_py_dict(py).unwrap();
            let r: i64 = f.call((), Some(&kwargs)).unwrap().extract().unwrap();
            assert_eq!(r, 12);

            // Positional / keyword split binding.
            let kwargs = [("b", 5)].into_py_dict(py).unwrap();
            let r: i64 = f.call((7,), Some(&kwargs)).unwrap().extract().unwrap();
            assert_eq!(r, 75);
        })
    }

    #[test]
    fn test_call() {
        use crate::{
            types::{IntoPyDict, PyAnyMethods, PyDict, PyTuple},
            wrap_pyfunction, Py, Python,
        };

        Python::attach(|py| {
            let f = wrap_pyfunction!(args_kwargs, py).unwrap();

            let args = PyTuple::new(py, [1, 2, 3]).unwrap();
            let kwargs = &[("foo", 1), ("bar", 2)].into_py_dict(py).unwrap();

            macro_rules! check_call {
                ($args:expr, $kwargs:expr) => {
                    let (a, k): (Py<PyTuple>, Py<PyDict>) = f
                        .call(args.clone(), Some(kwargs))
                        .unwrap()
                        .extract()
                        .unwrap();
                    assert!(a.is(&args));
                    assert!(k.is(kwargs));
                };
            }

            // Bound<'py, PyTuple>
            check_call!(args.clone(), kwargs);

            // &Bound<'py, PyTuple>
            check_call!(&args, kwargs);

            // Py<PyTuple>
            check_call!(args.clone().unbind(), kwargs);

            // &Py<PyTuple>
            check_call!(&args.as_unbound(), kwargs);

            // Borrowed<'_, '_, PyTuple>
            check_call!(args.as_borrowed(), kwargs);
        })
    }

    #[test]
    fn test_call_positional() {
        use crate::{
            types::{PyAnyMethods, PyNone, PyTuple},
            wrap_pyfunction, Py, Python,
        };

        Python::attach(|py| {
            let f = wrap_pyfunction!(args_kwargs, py).unwrap();

            let args = PyTuple::new(py, [1, 2, 3]).unwrap();

            macro_rules! check_call {
                ($args:expr, $kwargs:expr) => {
                    let (a, k): (Py<PyTuple>, Py<PyNone>) =
                        f.call1(args.clone()).unwrap().extract().unwrap();
                    assert!(a.is(&args));
                    assert!(k.is_none(py));
                };
            }

            // Bound<'py, PyTuple>
            check_call!(args.clone(), kwargs);

            // &Bound<'py, PyTuple>
            check_call!(&args, kwargs);

            // Py<PyTuple>
            check_call!(args.clone().unbind(), kwargs);

            // &Py<PyTuple>
            check_call!(args.as_unbound(), kwargs);

            // Borrowed<'_, '_, PyTuple>
            check_call!(args.as_borrowed(), kwargs);
        })
    }
}
