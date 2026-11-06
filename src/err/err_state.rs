use std::{
    any::TypeId,
    cell::UnsafeCell,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
    exceptions::{PyBaseException, PyTypeError},
    ffi,
    ffi_ptr_ext::FfiPtrExt,
    types::{PyAnyMethods, PyTraceback, PyType},
    Bound, Py, PyAny, PyErrArguments, PyTypeInfo, Python,
};

/// `PyErrState::state` value for an exception which has not yet been
/// materialized into the Python interpreter.
const LAZY: usize = 0;
/// `PyErrState::state` value while a thread is normalizing the state; only
/// that thread may access the inner value.
const NORMALIZING: usize = 1;
/// `PyErrState::state` value once the state holds the normalized (immutable)
/// exception.
const NORMALIZED: usize = 2;
/// `PyErrState::state` value if normalizing the state panicked; any further
/// access panics instead of using the torn state.
const POISONED: usize = 3;

/// Unset value for `PyErrState::normalizing_thread`.
const NO_THREAD: usize = 0;

pub(crate) struct PyErrState {
    /// Synchronization state, progressing monotonically
    /// `LAZY → NORMALIZING → NORMALIZED | POISONED`.
    ///
    /// The store of `NORMALIZED` is a release which publishes the write of
    /// the inner value; readers acquire-load the state before reading the
    /// inner value.
    state: AtomicUsize,
    /// [`thread_key`] of the thread which most recently attempted to
    /// normalize this state; set while `state` is `NORMALIZING` and left in
    /// place if it panics (so that the poisoned state still identifies the
    /// thread). Used to detect re-entrant normalization from the normalizing
    /// thread itself, which must panic rather than deadlock waiting for
    /// itself.
    normalizing_thread: AtomicUsize,
    // Safety: can only hand out references when in the "normalized" state. Will never change
    // after normalization.
    inner: UnsafeCell<Option<PyErrStateInner>>,
}

// Safety: The inner value is protected by the atomic `state` word: the single
// thread which transitions the state to `NORMALIZING` has exclusive access to
// the inner value until it publishes `NORMALIZED`, after which the inner
// value is immutable and only read.
unsafe impl Send for PyErrState {}
unsafe impl Sync for PyErrState {}
#[cfg(feature = "nightly")]
unsafe impl crate::marker::Ungil for PyErrState {}

impl PyErrState {
    pub(crate) fn lazy(f: Box<PyErrStateLazyFn>) -> Self {
        Self::from_inner(PyErrStateInner::Lazy(LazyState::Boxed(f)))
    }

    pub(crate) fn lazy_typed(ptype: PyErrTypeFn, args: PyErrLazyArguments) -> Self {
        Self::from_inner(PyErrStateInner::Lazy(LazyState::Typed {
            ptype: PyErrTypeSource::Fn(ptype),
            args,
        }))
    }

    pub(crate) fn lazy_arguments(ptype: Py<PyAny>, args: impl PyErrArguments + 'static) -> Self {
        // Fast path: for message-like arguments, store the arguments inline
        // in the lazy state to avoid the heap allocation (and later
        // deallocation) of the boxed closure, mirroring `PyErr::new`.
        match PyErrLazyArguments::try_from_args(args) {
            Ok(args) => Self::from_inner(PyErrStateInner::Lazy(LazyState::Typed {
                ptype: PyErrTypeSource::Py(ptype),
                args,
            })),
            Err(args) => Self::from_inner(PyErrStateInner::Lazy(LazyState::Boxed(Box::new(
                move |py| PyErrStateLazyFnOutput {
                    ptype,
                    pvalue: args.arguments(py),
                },
            )))),
        }
    }

    pub(crate) fn normalized(normalized: PyErrStateNormalized) -> Self {
        // Initialize the state as already `NORMALIZED`, so that
        // `as_normalized` always takes its fast path; as the state machine
        // progresses monotonically, a state born normalized can never reach
        // the (much slower) normalization path.
        Self::from_inner_init(PyErrStateInner::Normalized(normalized), NORMALIZED)
    }

    pub(crate) fn restore(self, py: Python<'_>) {
        self.inner
            .into_inner()
            .expect("PyErr state should never be invalid outside of normalization")
            .restore(py)
    }

    fn from_inner(inner: PyErrStateInner) -> Self {
        Self::from_inner_init(inner, LAZY)
    }

    fn from_inner_init(inner: PyErrStateInner, state: usize) -> Self {
        Self {
            state: AtomicUsize::new(state),
            normalizing_thread: AtomicUsize::new(NO_THREAD),
            inner: UnsafeCell::new(Some(inner)),
        }
    }

    #[inline]
    pub(crate) fn as_normalized(&self, py: Python<'_>) -> &PyErrStateNormalized {
        if self.state.load(Ordering::Acquire) == NORMALIZED {
            return self.normalized_ref();
        }

        self.make_normalized(py)
    }

    /// Returns the normalized state.
    ///
    /// # Safety
    ///
    /// The state must be `NORMALIZED`, so that the inner value will never be
    /// written again.
    #[inline]
    fn normalized_ref(&self) -> &PyErrStateNormalized {
        match unsafe {
            // Safety: self.inner will never be written again once normalized.
            &*self.inner.get()
        } {
            Some(PyErrStateInner::Normalized(n)) => n,
            _ => unreachable!(),
        }
    }

    #[cold]
    fn make_normalized(&self, py: Python<'_>) -> &PyErrStateNormalized {
        // Attempt to become the thread which normalizes this state. As the
        // state progresses monotonically, exactly one thread ever wins this
        // exchange, and normalization happens exactly once.
        match self
            .state
            .compare_exchange(LAZY, NORMALIZING, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => self.normalize_exclusive(py),
            // The state completed between the load in `as_normalized` and
            // this exchange; the failure ordering was an acquire, so the
            // inner value is already visible.
            Err(NORMALIZED) => self.normalized_ref(),
            Err(_) => {
                self.wait_until_normalized(py);
                self.normalized_ref()
            }
        }
    }

    /// Normalizes the state; the caller must have just won the transition of
    /// `state` from `LAZY` to `NORMALIZING`, which grants it exclusive access
    /// to the inner value for the duration of this function.
    #[cold]
    fn normalize_exclusive(&self, py: Python<'_>) -> &PyErrStateNormalized {
        // Record which thread is normalizing, so that re-entrant calls from
        // this thread can be detected in `wait_until_normalized` and panic
        // instead of waiting for this thread to finish (which would
        // self-deadlock).
        //
        // Relaxed ordering suffices: the check is only meaningful to the
        // normalizing thread itself, and a thread always observes its own
        // stores.
        let key = thread_key();
        self.normalizing_thread.store(key, Ordering::Relaxed);

        // If normalization panics, poison the state so that other threads
        // panic instead of waiting forever.
        let _poison_guard = PoisonOnPanic(self);

        // The normalizing thread never blocks, so it does not need to detach
        // from the interpreter; only the waiting threads detach (see
        // `wait_until_normalized`).

        // Safety: we hold the exclusive normalization rights, so no other
        // thread can access the inner value while we are normalizing it.
        let state = unsafe {
            (*self.inner.get())
                .take()
                .expect("Cannot normalize a PyErr while already normalizing it.")
        };

        let normalized_state = state.normalize(py);

        // Safety: we still hold the exclusive normalization rights.
        unsafe {
            *self.inner.get() = Some(PyErrStateInner::Normalized(normalized_state));
        }

        // Publish the normalized state; the release pairs with the acquire
        // loads in `as_normalized`, `make_normalized` and
        // `spin_until_normalized`.
        self.state.store(NORMALIZED, Ordering::Release);

        self.normalized_ref()
    }

    /// Waits for the thread which is currently normalizing this state to
    /// complete, and then returns with the state `NORMALIZED`.
    fn wait_until_normalized(&self, py: Python<'_>) {
        // Waiting for the normalizing thread from the normalizing thread
        // itself would deadlock.
        assert!(
            self.normalizing_thread.load(Ordering::Relaxed) != thread_key(),
            "Re-entrant normalization of PyErrState detected"
        );

        // Do not block while attached to the interpreter: the normalizing
        // thread may need to re-acquire it to execute the lazy exception
        // arguments (see https://github.com/PyO3/pyo3/issues/4764).
        py.detach(|| self.spin_until_normalized());
    }

    #[cold]
    fn spin_until_normalized(&self) {
        let mut spins = 0u32;
        loop {
            match self.state.load(Ordering::Acquire) {
                NORMALIZED => return,
                POISONED => panic!("PyErrState poisoned by a panicking normalization"),
                // Still normalizing.
                _ => {
                    spins += 1;
                    if spins < 32 {
                        std::hint::spin_loop();
                    } else if spins < 1024 {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(std::time::Duration::from_micros(100));
                    }
                }
            }
        }
    }
}

/// Guard which marks a `PyErrState` as poisoned if normalization unwinds
/// before completing.
struct PoisonOnPanic<'a>(&'a PyErrState);

impl Drop for PoisonOnPanic<'_> {
    fn drop(&mut self) {
        // Only poison if normalization did not complete successfully.
        let _ = self.0.state.compare_exchange(
            NORMALIZING,
            POISONED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

/// Process-wide unique nonzero key for the current thread, used for
/// re-entrancy detection during normalization.
///
/// `ThreadId` is opaque on stable Rust, so maintain our own counter. This is
/// only read on the (cold) normalization path, so the thread-local access
/// cost does not matter.
fn thread_key() -> usize {
    thread_local! {
        static THREAD_KEY: usize = {
            static NEXT_THREAD_KEY: AtomicUsize = AtomicUsize::new(1);
            NEXT_THREAD_KEY.fetch_add(1, Ordering::Relaxed)
        };
    }
    THREAD_KEY.with(|key| *key)
}

pub(crate) struct PyErrStateNormalized {
    #[cfg(not(Py_3_12))]
    ptype: Py<PyType>,
    pub pvalue: Py<PyBaseException>,
    #[cfg(not(Py_3_12))]
    ptraceback: Option<Py<PyTraceback>>,
}

impl PyErrStateNormalized {
    pub(crate) fn new(pvalue: Bound<'_, PyBaseException>) -> Self {
        Self {
            #[cfg(not(Py_3_12))]
            ptype: pvalue.get_type().into(),
            #[cfg(not(Py_3_12))]
            ptraceback: unsafe {
                ffi::PyException_GetTraceback(pvalue.as_ptr())
                    .assume_owned_or_opt(pvalue.py())
                    .map(|b| b.cast_into_unchecked().unbind())
            },
            pvalue: pvalue.into(),
        }
    }

    #[cfg(not(Py_3_12))]
    pub(crate) fn ptype<'py>(&self, py: Python<'py>) -> Bound<'py, PyType> {
        self.ptype.bind(py).clone()
    }

    #[cfg(Py_3_12)]
    pub(crate) fn ptype<'py>(&self, py: Python<'py>) -> Bound<'py, PyType> {
        self.pvalue.bind(py).get_type()
    }

    #[cfg(not(Py_3_12))]
    pub(crate) fn ptraceback<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyTraceback>> {
        self.ptraceback
            .as_ref()
            .map(|traceback| traceback.bind(py).clone())
    }

    #[cfg(Py_3_12)]
    pub(crate) fn ptraceback<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyTraceback>> {
        unsafe {
            ffi::PyException_GetTraceback(self.pvalue.as_ptr())
                .assume_owned_or_opt(py)
                .map(|b| b.cast_into_unchecked())
        }
    }

    pub(crate) fn take(py: Python<'_>) -> Option<PyErrStateNormalized> {
        #[cfg(Py_3_12)]
        {
            // Safety: PyErr_GetRaisedException can be called when attached to Python and
            // returns either NULL or an owned reference.
            unsafe { ffi::PyErr_GetRaisedException().assume_owned_or_opt(py) }.map(|pvalue| {
                PyErrStateNormalized {
                    // Safety: PyErr_GetRaisedException returns a valid exception type.
                    pvalue: unsafe { pvalue.cast_into_unchecked() }.unbind(),
                }
            })
        }

        #[cfg(not(Py_3_12))]
        {
            let (ptype, pvalue, ptraceback) = unsafe {
                let mut ptype: *mut ffi::PyObject = std::ptr::null_mut();
                let mut pvalue: *mut ffi::PyObject = std::ptr::null_mut();
                let mut ptraceback: *mut ffi::PyObject = std::ptr::null_mut();

                ffi::PyErr_Fetch(&mut ptype, &mut pvalue, &mut ptraceback);

                // Ensure that the exception coming from the interpreter is normalized.
                if !ptype.is_null() {
                    ffi::PyErr_NormalizeException(&mut ptype, &mut pvalue, &mut ptraceback);
                }

                // Safety: PyErr_NormalizeException will have produced up to three owned
                // references of the correct types.
                (
                    ptype
                        .assume_owned_or_opt(py)
                        .map(|b| b.cast_into_unchecked()),
                    pvalue
                        .assume_owned_or_opt(py)
                        .map(|b| b.cast_into_unchecked()),
                    ptraceback
                        .assume_owned_or_opt(py)
                        .map(|b| b.cast_into_unchecked()),
                )
            };

            ptype.map(|ptype| PyErrStateNormalized {
                ptype: ptype.unbind(),
                pvalue: pvalue.expect("normalized exception value missing").unbind(),
                ptraceback: ptraceback.map(Bound::unbind),
            })
        }
    }

    #[cfg(not(Py_3_12))]
    unsafe fn from_normalized_ffi_tuple(
        py: Python<'_>,
        ptype: *mut ffi::PyObject,
        pvalue: *mut ffi::PyObject,
        ptraceback: *mut ffi::PyObject,
    ) -> Self {
        PyErrStateNormalized {
            ptype: unsafe {
                ptype
                    .assume_owned_or_opt(py)
                    .expect("Exception type missing")
                    .cast_into_unchecked()
            }
            .unbind(),
            pvalue: unsafe {
                pvalue
                    .assume_owned_or_opt(py)
                    .expect("Exception value missing")
                    .cast_into_unchecked()
            }
            .unbind(),
            ptraceback: unsafe { ptraceback.assume_owned_or_opt(py) }
                .map(|b| unsafe { b.cast_into_unchecked() }.unbind()),
        }
    }

    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            #[cfg(not(Py_3_12))]
            ptype: self.ptype.clone_ref(py),
            pvalue: self.pvalue.clone_ref(py),
            #[cfg(not(Py_3_12))]
            ptraceback: self
                .ptraceback
                .as_ref()
                .map(|ptraceback| ptraceback.clone_ref(py)),
        }
    }
}

pub(crate) struct PyErrStateLazyFnOutput {
    pub(crate) ptype: Py<PyAny>,
    pub(crate) pvalue: Py<PyAny>,
}

pub(crate) type PyErrStateLazyFn =
    dyn for<'py> FnOnce(Python<'py>) -> PyErrStateLazyFnOutput + Send + Sync;

/// Function pointer which produces the exception type object of a
/// lazily-constructed exception.
///
/// This is monomorphized over the exception type `T` of `PyErr::new::<T, _>`;
/// as a plain function pointer it can be stored inline in the lazy state
/// without a heap allocation.
pub(crate) type PyErrTypeFn = for<'py> fn(Python<'py>) -> Py<PyAny>;

/// Arguments of a lazily-constructed exception, for the common case where they
/// can be stored inline instead of being captured by a boxed closure.
///
/// This avoids the heap allocation (and later deallocation) of the closure for
/// the overwhelmingly common `new_err("message")` and `new_err(format!(..))`
/// constructions.
pub(crate) enum PyErrLazyArguments {
    /// `&'static str` message arguments.
    StaticStr(&'static str),
    /// `String` message arguments.
    String(String),
    /// A pre-materialized arguments object, e.g. the `None` which
    /// `PyErr::from_value` passes for a type object.
    Py(Py<PyAny>),
}

impl PyErrLazyArguments {
    /// Attempts to store `args` inline, avoiding a heap allocation.
    ///
    /// Returns `Err(args)` for argument types which are not supported inline,
    /// so that the caller can fall back to the boxed-closure representation.
    pub(crate) fn try_from_args<A: 'static>(args: A) -> Result<Self, A> {
        if TypeId::of::<A>() == TypeId::of::<&'static str>() {
            // SAFETY: `TypeId` equality proves that `A` is `&'static str`
            // exactly. Although `TypeId` erases lifetimes, the `A: 'static`
            // bound means that any lifetime parameter of `A` outlives
            // `'static`, leaving `&'static str` as the only possibility.
            Ok(Self::StaticStr(unsafe {
                assume_type::<A, &'static str>(args)
            }))
        } else if TypeId::of::<A>() == TypeId::of::<String>() {
            // SAFETY: `TypeId` equality proves that `A` is `String` exactly
            // (it has no lifetime parameters).
            Ok(Self::String(unsafe { assume_type::<A, String>(args) }))
        } else if TypeId::of::<A>() == TypeId::of::<Py<PyAny>>() {
            // SAFETY: `TypeId` equality proves that `A` is `Py<PyAny>`
            // exactly (it has no lifetime parameters).
            Ok(Self::Py(unsafe { assume_type::<A, Py<PyAny>>(args) }))
        } else {
            Err(args)
        }
    }

    fn arguments(self, py: Python<'_>) -> Py<PyAny> {
        // Route through the same `PyErrArguments` implementations which the
        // boxed-closure path uses, so that the produced arguments are
        // identical.
        match self {
            Self::StaticStr(message) => PyErrArguments::arguments(message, py),
            Self::String(message) => PyErrArguments::arguments(message, py),
            Self::Py(obj) => PyErrArguments::arguments(obj, py),
        }
    }
}

/// Reinterprets `value` as type `T`.
///
/// # Safety
///
/// `A` and `T` must be the same type, e.g. as verified by comparing
/// `TypeId::of::<A>()` with `TypeId::of::<T>()`.
unsafe fn assume_type<A: 'static, T: 'static>(value: A) -> T {
    debug_assert_eq!(TypeId::of::<A>(), TypeId::of::<T>());
    // SAFETY: `A` and `T` are the same type, so this bitwise move reads a
    // valid `T` from `value`'s memory.
    let out = unsafe { std::ptr::read(&value as *const A as *const T) };
    // The contents of `value` have been moved to `out`; it must not be dropped
    // again as an `A`.
    std::mem::forget(value);
    out
}

/// Source of the exception type object of a lazily-constructed exception.
///
/// `Fn` is a monomorphized function pointer, used by `PyErr::new` where the
/// exception type is statically known; `Py` is a runtime type object, used by
/// `PyErr::from_type` / `PyErr::from_value` where the type is only known at
/// runtime.
enum PyErrTypeSource {
    Fn(PyErrTypeFn),
    Py(Py<PyAny>),
}

/// Lazy state of an exception which has not yet been materialized.
///
/// `Typed` (exception type source + inline message arguments) is used by
/// `PyErr::new`, `PyErr::from_type` and `PyErr::from_value` for
/// message-like arguments; `Boxed` is the fallback for arbitrary arguments.
enum LazyState {
    Typed {
        ptype: PyErrTypeSource,
        args: PyErrLazyArguments,
    },
    Boxed(Box<PyErrStateLazyFn>),
}

impl LazyState {
    fn into_output(self, py: Python<'_>) -> PyErrStateLazyFnOutput {
        match self {
            LazyState::Typed { ptype, args } => PyErrStateLazyFnOutput {
                ptype: match ptype {
                    PyErrTypeSource::Fn(f) => f(py),
                    PyErrTypeSource::Py(ty) => ty,
                },
                pvalue: args.arguments(py),
            },
            LazyState::Boxed(f) => f(py),
        }
    }
}

enum PyErrStateInner {
    Lazy(LazyState),
    Normalized(PyErrStateNormalized),
}

impl PyErrStateInner {
    fn normalize(self, py: Python<'_>) -> PyErrStateNormalized {
        match self {
            #[cfg(not(Py_3_12))]
            PyErrStateInner::Lazy(lazy) => {
                let (ptype, pvalue, ptraceback) = lazy_into_normalized_ffi_tuple(py, lazy);
                unsafe {
                    PyErrStateNormalized::from_normalized_ffi_tuple(py, ptype, pvalue, ptraceback)
                }
            }
            #[cfg(Py_3_12)]
            PyErrStateInner::Lazy(lazy) => {
                // To keep the implementation simple, just write the exception into the interpreter,
                // which will cause it to be normalized
                raise_lazy(py, lazy);
                PyErrStateNormalized::take(py)
                    .expect("exception missing after writing to the interpreter")
            }
            PyErrStateInner::Normalized(normalized) => normalized,
        }
    }

    #[cfg(not(Py_3_12))]
    fn restore(self, py: Python<'_>) {
        let (ptype, pvalue, ptraceback) = match self {
            PyErrStateInner::Lazy(lazy) => lazy_into_normalized_ffi_tuple(py, lazy),
            PyErrStateInner::Normalized(PyErrStateNormalized {
                ptype,
                pvalue,
                ptraceback,
            }) => (
                ptype.into_ptr(),
                pvalue.into_ptr(),
                ptraceback.map_or(std::ptr::null_mut(), Py::into_ptr),
            ),
        };
        unsafe { ffi::PyErr_Restore(ptype, pvalue, ptraceback) }
    }

    #[cfg(Py_3_12)]
    fn restore(self, py: Python<'_>) {
        match self {
            PyErrStateInner::Lazy(lazy) => raise_lazy(py, lazy),
            PyErrStateInner::Normalized(PyErrStateNormalized { pvalue }) => unsafe {
                ffi::PyErr_SetRaisedException(pvalue.into_ptr())
            },
        }
    }
}

#[cfg(not(Py_3_12))]
fn lazy_into_normalized_ffi_tuple(
    py: Python<'_>,
    lazy: LazyState,
) -> (*mut ffi::PyObject, *mut ffi::PyObject, *mut ffi::PyObject) {
    // To be consistent with 3.12 logic, go via raise_lazy, but also then normalize
    // the resulting exception
    raise_lazy(py, lazy);
    let mut ptype = std::ptr::null_mut();
    let mut pvalue = std::ptr::null_mut();
    let mut ptraceback = std::ptr::null_mut();
    unsafe {
        ffi::PyErr_Fetch(&mut ptype, &mut pvalue, &mut ptraceback);
        ffi::PyErr_NormalizeException(&mut ptype, &mut pvalue, &mut ptraceback);
    }
    (ptype, pvalue, ptraceback)
}

/// Raises a "lazy" exception state into the Python interpreter.
///
/// In principle this could be split in two; first a function to create an exception
/// in a normalized state, and then a call to `PyErr_SetRaisedException` to raise it.
///
/// This would require either moving some logic from C to Rust, or requesting a new
/// API in CPython.
fn raise_lazy(py: Python<'_>, lazy: LazyState) {
    let PyErrStateLazyFnOutput { ptype, pvalue } = lazy.into_output(py);
    unsafe {
        if ffi::PyExceptionClass_Check(ptype.as_ptr()) == 0 {
            ffi::PyErr_SetString(
                PyTypeError::type_object_raw(py).cast(),
                c"exceptions must derive from BaseException".as_ptr(),
            )
        } else {
            ffi::PyErr_SetObject(ptype.as_ptr(), pvalue.as_ptr())
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        exceptions::PyValueError, sync::PyOnceLock, types::any::PyAnyMethods,
        types::string::PyStringMethods, Py, PyAny, PyErr, PyErrArguments, PyTypeInfo, Python,
    };

    #[test]
    #[should_panic(expected = "Re-entrant normalization of PyErrState detected")]
    fn test_reentrant_normalization() {
        static ERR: PyOnceLock<PyErr> = PyOnceLock::new();

        struct RecursiveArgs;

        impl PyErrArguments for RecursiveArgs {
            fn arguments(self, py: Python<'_>) -> Py<PyAny> {
                // .value(py) triggers normalization
                ERR.get(py)
                    .expect("is set just below")
                    .value(py)
                    .clone()
                    .into()
            }
        }

        Python::attach(|py| {
            ERR.set(py, PyValueError::new_err(RecursiveArgs)).unwrap();
            ERR.get(py).expect("is set just above").value(py);
        })
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))] // We are building wasm Python with pthreads disabled
    fn test_no_deadlock_thread_switch() {
        static ERR: PyOnceLock<PyErr> = PyOnceLock::new();

        struct GILSwitchArgs;

        impl PyErrArguments for GILSwitchArgs {
            fn arguments(self, py: Python<'_>) -> Py<PyAny> {
                // releasing the GIL potentially allows for other threads to deadlock
                // with the normalization going on here
                py.detach(|| {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                });
                py.None()
            }
        }

        Python::attach(|py| ERR.set(py, PyValueError::new_err(GILSwitchArgs)).unwrap());

        // Let many threads attempt to read the normalized value at the same time
        let handles = (0..10)
            .map(|_| {
                std::thread::spawn(|| {
                    Python::attach(|py| {
                        ERR.get(py).expect("is set just above").value(py);
                    });
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle.join().unwrap();
        }

        // We should never have deadlocked, and should be able to run
        // this assertion
        Python::attach(|py| {
            assert!(ERR
                .get(py)
                .expect("is set above")
                .is_instance_of::<PyValueError>(py))
        });
    }

    #[test]
    fn test_lazy_typed_arguments_normalize_correctly() {
        Python::attach(|py| {
            // Inline `&'static str` fast path
            let err = PyValueError::new_err("static message");
            assert!(err.is_instance_of::<PyValueError>(py));
            assert_eq!(
                err.value(py).str().unwrap().to_cow().unwrap(),
                "static message"
            );

            // Inline `String` fast path
            let err = PyValueError::new_err(String::from("owned message"));
            assert!(err.is_instance_of::<PyValueError>(py));
            assert_eq!(
                err.value(py).str().unwrap().to_cow().unwrap(),
                "owned message"
            );

            // Fallback to the boxed-closure representation (tuple arguments)
            let err = PyValueError::new_err(("tuple message", 42));
            assert!(err.is_instance_of::<PyValueError>(py));
            assert_eq!(
                err.value(py).str().unwrap().to_cow().unwrap(),
                "('tuple message', 42)"
            );
        });
    }

    #[test]
    fn test_lazy_typed_arguments_restore() {
        Python::attach(|py| {
            // restore without a prior inspect exercises the restore path for
            // the typed representation
            let err = PyValueError::new_err("restore me");
            err.restore(py);
            let fetched = PyErr::fetch(py);
            assert!(fetched.is_instance_of::<PyValueError>(py));
            assert_eq!(
                fetched.value(py).str().unwrap().to_cow().unwrap(),
                "restore me"
            );

            let err = PyValueError::new_err(String::from("restore me too"));
            err.restore(py);
            let fetched = PyErr::fetch(py);
            assert_eq!(
                fetched.value(py).str().unwrap().to_cow().unwrap(),
                "restore me too"
            );
        });
    }

    #[test]
    fn test_lazy_typed_arguments_drop_without_normalize() {
        // Constructing and dropping lazy errors without ever materializing
        // them must be sound for both inline representations (in particular
        // the `String` must be freed exactly once, without the GIL held).
        drop(PyValueError::new_err(String::from(
            "dropped unmaterialized",
        )));
        drop(PyValueError::new_err("dropped unmaterialized"));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))] // We are building wasm Python with pthreads disabled
    fn test_lazy_typed_arguments_multi_thread_normalize() {
        static ERR: PyOnceLock<PyErr> = PyOnceLock::new();

        Python::attach(|py| {
            ERR.set(py, PyValueError::new_err("race")).unwrap();
        });

        // Many threads racing to normalize the same typed lazy state
        let handles = (0..10)
            .map(|_| {
                std::thread::spawn(|| {
                    Python::attach(|py| {
                        ERR.get(py).expect("is set just above").value(py);
                    });
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle.join().unwrap();
        }

        Python::attach(|py| {
            assert!(ERR
                .get(py)
                .expect("is set above")
                .is_instance_of::<PyValueError>(py))
        });
    }

    #[test]
    fn test_lazy_typed_arguments_display() {
        // The inline representations must produce the same user-visible
        // formatting as the boxed-closure path, both before and after
        // normalization.
        assert_eq!(
            PyValueError::new_err("static display").to_string(),
            "ValueError: static display"
        );
        assert_eq!(
            PyValueError::new_err(String::from("owned display")).to_string(),
            "ValueError: owned display"
        );
        let debug = format!("{:?}", PyValueError::new_err(String::from("debug me")));
        assert!(
            debug.contains("ValueError") && debug.contains("debug me"),
            "unexpected Debug output: {debug}"
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))] // We are building wasm Python with pthreads disabled
    fn test_lazy_typed_string_multi_thread_normalize() {
        // Variant of the above for the `String` representation, whose move /
        // single-drop semantics under the `call_once` race are the easiest to
        // silently break in future refactors (e.g. by adding `Clone`).
        static ERR: PyOnceLock<PyErr> = PyOnceLock::new();

        Python::attach(|py| {
            ERR.set(py, PyValueError::new_err(String::from("race")))
                .unwrap();
        });

        let handles = (0..10)
            .map(|_| {
                std::thread::spawn(|| {
                    Python::attach(|py| {
                        ERR.get(py).expect("is set just above").value(py);
                    });
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle.join().unwrap();
        }

        Python::attach(|py| {
            let err = ERR.get(py).expect("is set above");
            assert!(err.is_instance_of::<PyValueError>(py));
            assert_eq!(err.value(py).str().unwrap().to_cow().unwrap(), "race");
        });
    }

    #[test]
    #[should_panic(expected = "Re-entrant normalization of PyErrState detected")]
    fn test_poisoned_state_panics_on_reaccess() {
        // If normalization panics, later access to the state must panic
        // rather than hang or hand out invalid data.
        static ERR: PyOnceLock<PyErr> = PyOnceLock::new();

        struct PanicArgs;

        impl PyErrArguments for PanicArgs {
            fn arguments(self, _py: Python<'_>) -> Py<PyAny> {
                panic!("panic inside normalization");
            }
        }

        Python::attach(|py| {
            ERR.set(py, PyValueError::new_err(PanicArgs)).unwrap();
        });

        // First normalization panics; swallow the panic so that we can
        // observe the state which it leaves behind.
        let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Python::attach(|py| {
                ERR.get(py).expect("is set just above").value(py);
            });
        }));
        assert!(first.is_err());

        Python::attach(|py| {
            ERR.get(py).expect("is set just above").value(py);
        });
    }

    #[test]
    #[should_panic(expected = "PyErrState poisoned by a panicking normalization")]
    #[cfg(not(target_arch = "wasm32"))] // We are building wasm Python with pthreads disabled
    fn test_poisoned_state_panics_on_other_thread() {
        // Cross-thread version of the above: a *different* thread accessing
        // the poisoned state must also panic rather than hang.
        static ERR: PyOnceLock<PyErr> = PyOnceLock::new();

        struct PanicArgs;

        impl PyErrArguments for PanicArgs {
            fn arguments(self, _py: Python<'_>) -> Py<PyAny> {
                panic!("panic inside normalization");
            }
        }

        Python::attach(|py| {
            ERR.set(py, PyValueError::new_err(PanicArgs)).unwrap();
        });

        // Another thread performs the (panicking) first normalization.
        let handle = std::thread::spawn(|| {
            let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Python::attach(|py| {
                    ERR.get(py).expect("is set just above").value(py);
                });
            }));
            assert!(first.is_err());
        });
        handle.join().unwrap();

        Python::attach(|py| {
            ERR.get(py).expect("is set just above").value(py);
        });
    }

    #[test]
    fn test_from_type_lazy_arguments_normalize_correctly() {
        Python::attach(|py| {
            // Inline `&'static str` fast path
            let err = PyErr::from_type(PyValueError::type_object(py), "static message");
            assert!(err.is_instance_of::<PyValueError>(py));
            assert_eq!(
                err.value(py).str().unwrap().to_cow().unwrap(),
                "static message"
            );

            // Inline `String` fast path
            let err =
                PyErr::from_type(PyValueError::type_object(py), String::from("owned message"));
            assert!(err.is_instance_of::<PyValueError>(py));
            assert_eq!(
                err.value(py).str().unwrap().to_cow().unwrap(),
                "owned message"
            );

            // Fallback to the boxed-closure representation (tuple arguments)
            let err = PyErr::from_type(PyValueError::type_object(py), ("tuple message", 42));
            assert!(err.is_instance_of::<PyValueError>(py));
            assert_eq!(
                err.value(py).str().unwrap().to_cow().unwrap(),
                "('tuple message', 42)"
            );
        });
    }

    #[test]
    fn test_from_type_lazy_arguments_restore() {
        Python::attach(|py| {
            // restore without a prior inspect exercises the restore path for
            // the inline representation with a runtime type object
            let err = PyErr::from_type(PyValueError::type_object(py), "restore me");
            err.restore(py);
            let fetched = PyErr::fetch(py);
            assert!(fetched.is_instance_of::<PyValueError>(py));
            assert_eq!(
                fetched.value(py).str().unwrap().to_cow().unwrap(),
                "restore me"
            );
        });
    }

    #[test]
    fn test_from_value_lazy_drop_and_restore_without_normalize() {
        Python::attach(|py| {
            // from_value of a type object takes the lazy `Py`-arguments path;
            // constructing and dropping it without ever materializing it
            // must be sound (both `Py` handles released exactly once).
            drop(PyErr::from_value(PyValueError::type_object(py).into_any()));
            drop(PyErr::from_value(PyValueError::type_object(py).into_any()));

            // Restoring without a prior normalize materializes the type
            // object into an exception instance.
            let err = PyErr::from_value(PyValueError::type_object(py).into_any());
            err.restore(py);
            let fetched = PyErr::fetch(py);
            assert!(fetched.is_instance_of::<PyValueError>(py));
            assert_eq!(fetched.value(py).str().unwrap().to_cow().unwrap(), "");
        });
    }
}
