//! Interaction with attachment of the current thread to the Python interpreter.

#[cfg(pyo3_disable_reference_pool)]
use crate::impl_::panic::PanicTrap;
use crate::{ffi, Python};

use std::cell::Cell;
#[cfg(not(pyo3_disable_reference_pool))]
use std::cell::RefCell;
#[cfg(not(pyo3_disable_reference_pool))]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(pyo3_disable_reference_pool))]
use std::sync::OnceLock;
#[cfg_attr(pyo3_disable_reference_pool, allow(unused_imports))]
use std::{mem, ptr::NonNull, sync};

std::thread_local! {
    /// This is an internal counter in pyo3 monitoring whether this thread is attached to the interpreter.
    ///
    /// It will be incremented whenever an AttachGuard is created, and decremented whenever
    /// they are dropped.
    ///
    /// As a result, if this thread is attached to the interpreter, ATTACH_COUNT is greater than zero.
    ///
    /// Additionally, we sometimes need to prevent safe access to the Python interpreter,
    /// e.g. when implementing `__traverse__`, which is represented by a negative value.
    static ATTACH_COUNT: Cell<isize> = const { Cell::new(0) };
}

const ATTACH_FORBIDDEN_DURING_TRAVERSE: isize = -1;

/// Checks whether the thread is attached to the Python interpreter.
///
/// Note: This uses pyo3's internal count rather than PyGILState_Check for two reasons:
///  1) for performance
///  2) PyGILState_Check always returns 1 if the sub-interpreter APIs have ever been called,
///     which could lead to incorrect conclusions that the thread is attached.
#[inline(always)]
pub(crate) fn thread_is_attached() -> bool {
    ATTACH_COUNT.try_with(|c| c.get() > 0).unwrap_or(false)
}

/// RAII type that represents thread attachment to the interpreter.
pub(crate) enum AttachGuard {
    /// Indicates the thread was already attached when this AttachGuard was acquired.
    Assumed,
    /// Indicates that we attached when this AttachGuard was acquired
    Ensured { gstate: ffi::PyGILState_STATE },
}

/// Possible error when calling `try_attach()`
pub(crate) enum AttachError {
    /// Forbidden during GC traversal.
    ForbiddenDuringTraverse,
    /// The interpreter is not initialized.
    NotInitialized,
    #[cfg(Py_3_13)]
    /// The interpreter is finalizing.
    Finalizing,
}

impl AttachGuard {
    /// PyO3 internal API for attaching to the Python interpreter. The public API is Python::attach.
    ///
    /// If the thread was already attached via PyO3, this returns
    /// `AttachGuard::Assumed`. Otherwise, the thread will attach now and
    /// `AttachGuard::Ensured` will be returned.
    pub(crate) fn attach() -> Self {
        match Self::try_attach() {
            Ok(guard) => guard,
            Err(AttachError::ForbiddenDuringTraverse) => {
                panic!("{}", ForbidAttaching::FORBIDDEN_DURING_TRAVERSE)
            }
            Err(AttachError::NotInitialized) => {
                // try to initialize the interpreter and try again
                crate::interpreter_lifecycle::ensure_initialized();
                unsafe { Self::do_attach_unchecked() }
            }
            #[cfg(Py_3_13)]
            Err(AttachError::Finalizing) => {
                panic!("Cannot attach to the Python interpreter while it is finalizing.");
            }
        }
    }

    /// Variant of the above which will will return gracefully if the interpreter cannot be attached to.
    pub(crate) fn try_attach() -> Result<Self, AttachError> {
        match ATTACH_COUNT.try_with(|c| c.get()) {
            Ok(i) if i > 0 => {
                // SAFETY: We just checked that the thread is already attached.
                return Ok(unsafe { Self::assume() });
            }
            // Cannot attach during GC traversal.
            Ok(ATTACH_FORBIDDEN_DURING_TRAVERSE) => {
                return Err(AttachError::ForbiddenDuringTraverse)
            }
            // other cases handled below
            _ => {}
        }

        // SAFETY: always safe to call this
        if unsafe { ffi::Py_IsInitialized() } == 0 {
            return Err(AttachError::NotInitialized);
        }

        // Calling `PyGILState_Ensure` while finalizing may crash CPython in unpredictable
        // ways, we'll make a best effort attempt here to avoid that. (There's a time of
        // check to time-of-use issue, but it's better than nothing.)
        //
        // SAFETY: always safe to call this
        #[cfg(Py_3_13)]
        if unsafe { ffi::Py_IsFinalizing() } != 0 {
            // If the interpreter is not initialized, we cannot attach.
            return Err(AttachError::Finalizing);
        }

        // SAFETY: We have done everything reasonable to ensure we're in a safe state to
        // attach to the Python interpreter.
        Ok(unsafe { Self::do_attach_unchecked() })
    }

    /// Acquires the `AttachGuard` without performing any state checking.
    ///
    /// This can be called in "unsafe" contexts where the normal interpreter state
    /// checking performed by `AttachGuard::try_attach` may fail. This includes calling
    /// as part of multi-phase interpreter initialization.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the Python interpreter is sufficiently initialized
    /// for a thread to be able to attach to it.
    pub(crate) unsafe fn attach_unchecked() -> Self {
        if thread_is_attached() {
            return unsafe { Self::assume() };
        }

        unsafe { Self::do_attach_unchecked() }
    }

    /// Attach to the interpreter, without a fast-path to check if the thread is already attached.
    #[cold]
    unsafe fn do_attach_unchecked() -> Self {
        // SAFETY: interpreter is sufficiently initialized to attach a thread.
        let gstate = unsafe { ffi::PyGILState_Ensure() };
        increment_attach_count();
        // SAFETY: just attached to the interpreter
        drop_deferred_references(unsafe { Python::assume_attached() });
        AttachGuard::Ensured { gstate }
    }

    /// Acquires the `AttachGuard` while assuming that the thread is already attached
    /// to the interpreter.
    pub(crate) unsafe fn assume() -> Self {
        increment_attach_count();
        // SAFETY: invariant of calling this function
        drop_deferred_references(unsafe { Python::assume_attached() });
        AttachGuard::Assumed
    }

    /// Gets the Python token associated with this [`AttachGuard`].
    #[inline]
    pub(crate) fn python(&self) -> Python<'_> {
        // SAFETY: this guard guarantees the thread is attached
        unsafe { Python::assume_attached() }
    }
}

/// The Drop implementation for `AttachGuard` will decrement the attach count (and potentially detach).
impl Drop for AttachGuard {
    fn drop(&mut self) {
        match self {
            AttachGuard::Assumed => {}
            AttachGuard::Ensured { gstate } => unsafe {
                ffi::PyGILState_Release(*gstate);
            },
        }
        decrement_attach_count();
    }
}

#[cfg(not(pyo3_disable_reference_pool))]
type PyObjVec = Vec<NonNull<ffi::PyObject>>;

/// Number of pending decrefs which a thread will buffer in thread-local storage
/// before flushing them into the global pool.
///
/// This amortizes the cost of the global pool's mutex: threads dropping `Py`
/// handles while detached take the mutex once per [`TLS_PENDING_DECREFS_CAPACITY`]
/// drops instead of once per drop.
#[cfg(not(pyo3_disable_reference_pool))]
const TLS_PENDING_DECREFS_CAPACITY: usize = 64;

/// Wrapper type for the thread-local buffer of pending decrefs, so that a thread
/// which exits with buffered decrefs flushes them to the global pool (raw
/// pointers do not decref themselves, so the references would otherwise be
/// leaked).
#[cfg(not(pyo3_disable_reference_pool))]
struct TlsPendingDecrefs(PyObjVec);

#[cfg(not(pyo3_disable_reference_pool))]
impl TlsPendingDecrefs {
    /// Takes the buffered objects out, leaving the buffer empty (and thereby its
    /// `Drop`, which flushes non-empty buffers, a no-op).
    fn take(&mut self) -> PyObjVec {
        mem::take(&mut self.0)
    }
}

#[cfg(not(pyo3_disable_reference_pool))]
impl Drop for TlsPendingDecrefs {
    fn drop(&mut self) {
        flush_pending_decrefs(&mut self.0);
    }
}

#[cfg(not(pyo3_disable_reference_pool))]
thread_local! {
    /// Buffer for decrefs pending on this thread, which avoids taking the global
    /// pool's mutex for every `Py` dropped while detached.
    ///
    /// The buffer is flushed to the global pool when it fills up, when the thread
    /// drains (i.e. on attach and at the end of detach), and when the thread exits.
    ///
    /// The buffer's allocation is retained across drains (up to the largest batch
    /// it has ever held) to avoid allocator churn; it is never shrunk.
    ///
    /// On wasm targets built without atomics (which have no threads), thread-exit
    /// destructors do not run, so a residual buffer at process end is simply
    /// abandoned (bounded by `TLS_PENDING_DECREFS_CAPACITY - 1` references).
    static TLS_PENDING_DECREFS: RefCell<TlsPendingDecrefs> =
        const { RefCell::new(TlsPendingDecrefs(Vec::new())) };
}

#[cfg(not(pyo3_disable_reference_pool))]
/// Flushes a batch of pending decrefs into the global pool.
fn flush_pending_decrefs(batch: &mut PyObjVec) {
    let count = batch.len();
    if count == 0 {
        return;
    }
    let pool = get_pool();
    // The count is updated before inserting and only decremented after objects
    // have been taken out, so `count >= pending_decrefs.len()` holds at all
    // times. A drain which observes `count == 0` may therefore safely skip the
    // mutex without missing objects; a stale or in-flight count merely defers
    // processing of some objects to a later drain (never drops them).
    pool.pending_count.fetch_add(count, Ordering::Relaxed);
    pool.pending_decrefs.lock().unwrap().extend(batch.drain(..));
}

#[cfg(not(pyo3_disable_reference_pool))]
/// Global storage for objects which were dec_ref while not attached, backing the
/// thread-local buffers which overflow into it when full.
struct ReferencePool {
    pending_decrefs: sync::Mutex<PyObjVec>,
    /// Number of objects currently held in `pending_decrefs`.
    ///
    /// This is a best-effort count which allows drains to skip the mutex when the
    /// pool is empty. Relaxed ordering is sufficient because a stale count merely
    /// defers processing to the next drain; no decref is ever lost.
    pending_count: AtomicUsize,
}

#[cfg(not(pyo3_disable_reference_pool))]
impl ReferencePool {
    const fn new() -> Self {
        Self {
            pending_decrefs: sync::Mutex::new(Vec::new()),
            pending_count: AtomicUsize::new(0),
        }
    }

    /// Registers a single object to have its reference count decreased by
    /// whichever thread next drains the pool.
    fn register_decref(&self, obj: NonNull<ffi::PyObject>) {
        self.pending_count.fetch_add(1, Ordering::Relaxed);
        self.pending_decrefs.lock().unwrap().push(obj);
    }

    /// Swaps the pool's pending decrefs with the given batch, returning the
    /// number of objects which the pool was holding.
    ///
    /// The given batch is expected to be empty (while owning spare capacity);
    /// after the call the pool holds the batch's allocation for reuse, and the
    /// given batch holds the objects to decref outside of the pool's mutex.
    fn swap_pending_decrefs(&self, batch: &mut PyObjVec) -> usize {
        let mut pending_decrefs = self.pending_decrefs.lock().unwrap();
        let count = pending_decrefs.len();
        if count > 0 {
            mem::swap(&mut *pending_decrefs, batch);
        }
        count
    }
}

#[cfg(not(pyo3_disable_reference_pool))]
unsafe impl Send for ReferencePool {}

#[cfg(not(pyo3_disable_reference_pool))]
unsafe impl Sync for ReferencePool {}

#[cfg(not(pyo3_disable_reference_pool))]
static POOL: OnceLock<ReferencePool> = OnceLock::new();

#[cfg(not(pyo3_disable_reference_pool))]
fn get_pool() -> &'static ReferencePool {
    POOL.get_or_init(ReferencePool::new)
}

#[cfg_attr(pyo3_disable_reference_pool, inline(always))]
#[cfg_attr(pyo3_disable_reference_pool, allow(unused_variables))]
fn drop_deferred_references(_py: Python<'_>) {
    #[cfg(not(pyo3_disable_reference_pool))]
    {
        // Take the thread-local buffer (including its allocation, to be reused
        // below) out of thread-local storage, dropping the borrow before any
        // reference counts are decreased, so that `__del__` handlers which run
        // during the decrefs cannot observe a mutably-borrowed buffer.
        let mut batch = TLS_PENDING_DECREFS
            .try_with(|pending| pending.borrow_mut().take())
            .unwrap_or_default();

        // SAFETY: all pointers in the batch are owned references which were
        // handed over to the pool by `register_decref`.
        for ptr in batch.drain(..) {
            unsafe { ffi::Py_DECREF(ptr.as_ptr()) };
        }

        // Drain the global pool, if the count suggests it is non-empty. The swap
        // leaves the pool holding this thread's spare allocation, so that the
        // pool does not reallocate on the next flush.
        if let Some(pool) = POOL.get() {
            if pool.pending_count.load(Ordering::Relaxed) > 0 {
                let count = pool.swap_pending_decrefs(&mut batch);
                pool.pending_count.fetch_sub(count, Ordering::Relaxed);
                // SAFETY: as above; decref outside of the pool's mutex so that
                // `__del__` handlers re-entering the drain cannot deadlock.
                for ptr in batch.drain(..) {
                    unsafe { ffi::Py_DECREF(ptr.as_ptr()) };
                }
            }
        }

        // Return the batch's allocation to the thread-local buffer to avoid
        // reallocating on the next drop, unless the buffer was refilled in the
        // meantime (e.g. by a `__del__` handler which detached).
        let _ = TLS_PENDING_DECREFS.try_with(|pending| {
            let mut pending = pending.borrow_mut();
            if pending.0.is_empty() {
                pending.0 = batch;
            }
        });
    }
}

/// A guard which can be used to temporarily detach from the interpreter and restore on `Drop`.
pub(crate) struct SuspendAttach {
    count: isize,
    tstate: *mut ffi::PyThreadState,
}

impl SuspendAttach {
    pub(crate) unsafe fn new() -> Self {
        let count = ATTACH_COUNT.with(|c| c.replace(0));
        let tstate = unsafe { ffi::PyEval_SaveThread() };

        Self { count, tstate }
    }
}

impl Drop for SuspendAttach {
    fn drop(&mut self) {
        ATTACH_COUNT.with(|c| c.set(self.count));
        unsafe {
            ffi::PyEval_RestoreThread(self.tstate);

            // Update counts of `Py<T>` that were dropped while not attached.
            #[cfg(not(pyo3_disable_reference_pool))]
            drop_deferred_references(Python::assume_attached());
        }
    }
}

/// Used to lock safe access to the interpreter
pub(crate) struct ForbidAttaching {
    count: isize,
}

impl ForbidAttaching {
    const FORBIDDEN_DURING_TRAVERSE: &'static str = "Attaching a thread to the interpreter is prohibited while a __traverse__ implementation is running.";

    /// Lock access to the interpreter while an implementation of `__traverse__` is running
    pub fn during_traverse() -> Self {
        Self::new(ATTACH_FORBIDDEN_DURING_TRAVERSE)
    }

    fn new(reason: isize) -> Self {
        let count = ATTACH_COUNT.with(|c| c.replace(reason));

        Self { count }
    }

    #[cold]
    fn bail(current: isize) {
        match current {
            ATTACH_FORBIDDEN_DURING_TRAVERSE => panic!("{}", Self::FORBIDDEN_DURING_TRAVERSE),
            _ => panic!("Attaching a thread to the interpreter is currently prohibited."),
        }
    }
}

impl Drop for ForbidAttaching {
    fn drop(&mut self) {
        ATTACH_COUNT.with(|c| c.set(self.count));
    }
}

/// Registers a Python object pointer to have its reference count decreased
/// the next time the thread drains pending decrefs in pyo3 (i.e. when it next
/// attaches, or at the end of a `detach`).
///
/// The object is first buffered in thread-local storage; if that buffer fills
/// up, or the thread exits before draining, it is handed over to the global
/// pool, from which any thread's next attach will release it. As a
/// consequence, a handle dropped on a thread which then stays detached is
/// released by that thread's own next drain (or its exit) rather than by
/// another thread's attach.
///
/// If the thread is attached, the reference count will be decreased immediately instead of being queued
/// for later.
///
/// # Safety
/// - The object must be an owned Python reference.
/// - The reference must not be used after calling this function.
#[inline]
pub unsafe fn register_decref(obj: NonNull<ffi::PyObject>) {
    #[cfg(not(pyo3_disable_reference_pool))]
    {
        // Fast path: buffer the decref in thread-local storage, taking no lock.
        // When the buffer fills up, flush it to the global pool (amortizing the
        // pool's mutex to one lock per `TLS_PENDING_DECREFS_CAPACITY` drops).
        if TLS_PENDING_DECREFS
            .try_with(|pending| {
                let mut pending = pending.borrow_mut();
                pending.0.push(obj);
                if pending.0.len() >= TLS_PENDING_DECREFS_CAPACITY {
                    flush_pending_decrefs(&mut pending.0);
                }
            })
            .is_err()
        {
            // Fallback: the thread-local buffer is unavailable (e.g. the thread
            // is tearing down); queue the decref in the global pool directly.
            get_pool().register_decref(obj);
        }
    }
    #[cfg(all(
        pyo3_disable_reference_pool,
        not(pyo3_leak_on_drop_without_reference_pool)
    ))]
    {
        let _trap = PanicTrap::new("Aborting the process to avoid panic-from-drop.");
        panic!("Cannot drop pointer into Python heap without the thread being attached.");
    }
}

/// Private helper function to check if we are currently in a GC traversal (as detected by PyO3).
#[cfg(any(not(Py_LIMITED_API), Py_3_11))]
pub(crate) fn is_in_gc_traversal() -> bool {
    ATTACH_COUNT
        .try_with(|c| c.get() == ATTACH_FORBIDDEN_DURING_TRAVERSE)
        .unwrap_or(false)
}

/// Increments pyo3's internal attach count - to be called whenever an AttachGuard is created.
#[inline(always)]
fn increment_attach_count() {
    // Ignores the error in case this function called from `atexit`.
    let _ = ATTACH_COUNT.try_with(|c| {
        let current = c.get();
        if current < 0 {
            ForbidAttaching::bail(current);
        }
        c.set(current + 1);
    });
}

/// Decrements pyo3's internal attach count - to be called whenever AttachGuard is dropped.
#[inline(always)]
fn decrement_attach_count() {
    // Ignores the error in case this function called from `atexit`.
    let _ = ATTACH_COUNT.try_with(|c| {
        let current = c.get();
        debug_assert!(
            current > 0,
            "Negative attach count detected. Please report this error to the PyO3 repo as a bug."
        );
        c.set(current - 1);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{types::PyAnyMethods, Py, PyAny, Python};

    fn get_object(py: Python<'_>) -> Py<PyAny> {
        py.eval(c"object()", None, None).unwrap().unbind()
    }

    #[cfg(not(pyo3_disable_reference_pool))]
    fn tls_pending_dec_refs_contain(obj: &Py<PyAny>) -> bool {
        TLS_PENDING_DECREFS
            .try_with(|pending| {
                pending
                    .borrow()
                    .0
                    .contains(&unsafe { NonNull::new_unchecked(obj.as_ptr()) })
            })
            .unwrap_or(false)
    }

    #[cfg(not(pyo3_disable_reference_pool))]
    fn pool_dec_refs_does_not_contain(obj: &Py<PyAny>) -> bool {
        let obj_ptr = unsafe { NonNull::new_unchecked(obj.as_ptr()) };
        // Check the thread-local buffer before taking the global pool's mutex,
        // mirroring the production lock order (TLS borrow never held across the
        // global lock).
        !tls_pending_dec_refs_contain(obj)
            && !get_pool()
                .pending_decrefs
                .lock()
                .unwrap()
                .contains(&obj_ptr)
    }

    // With free-threading, threads can empty the POOL at any time, so this
    // function does not test anything meaningful
    #[cfg(not(any(pyo3_disable_reference_pool, Py_GIL_DISABLED)))]
    fn pool_dec_refs_contains(obj: &Py<PyAny>) -> bool {
        let obj_ptr = unsafe { NonNull::new_unchecked(obj.as_ptr()) };
        tls_pending_dec_refs_contain(obj)
            || get_pool()
                .pending_decrefs
                .lock()
                .unwrap()
                .contains(&obj_ptr)
    }

    #[test]
    fn test_pyobject_drop_attached_decreases_refcnt() {
        Python::attach(|py| {
            let obj = get_object(py);

            // Create a reference to drop while attached.
            let reference = obj.clone_ref(py);

            assert_eq!(obj.get_refcnt(py), 2);
            #[cfg(not(pyo3_disable_reference_pool))]
            assert!(pool_dec_refs_does_not_contain(&obj));

            // While attached, reference count will be decreased immediately.
            drop(reference);

            assert_eq!(obj.get_refcnt(py), 1);
            #[cfg(not(any(pyo3_disable_reference_pool)))]
            assert!(pool_dec_refs_does_not_contain(&obj));
        });
    }

    #[test]
    #[cfg(all(not(pyo3_disable_reference_pool), not(target_arch = "wasm32")))] // We are building wasm Python with pthreads disabled
    fn test_pyobject_drop_detached_doesnt_decrease_refcnt() {
        let obj = Python::attach(|py| {
            let obj = get_object(py);
            // Create a reference to drop while detached.
            let reference = obj.clone_ref(py);

            assert_eq!(obj.get_refcnt(py), 2);
            assert!(pool_dec_refs_does_not_contain(&obj));

            // Drop reference in a separate (detached) thread.
            std::thread::spawn(move || drop(reference)).join().unwrap();

            // The reference count should not have changed, it is remembered
            // to release later.
            assert_eq!(obj.get_refcnt(py), 2);
            #[cfg(not(Py_GIL_DISABLED))]
            assert!(pool_dec_refs_contains(&obj));
            obj
        });

        // On next attach, the reference is released
        #[allow(unused)]
        Python::attach(|py| {
            // With free-threading, another thread could still be processing
            // DECREFs after releasing the lock on the POOL, so the
            // refcnt could still be 2 when this assert happens
            #[cfg(not(Py_GIL_DISABLED))]
            assert_eq!(obj.get_refcnt(py), 1);
            assert!(pool_dec_refs_does_not_contain(&obj));
        });
    }

    #[test]
    #[cfg(not(pyo3_disable_reference_pool))]
    fn test_register_decref_buffers_small_batches_out_of_global_pool() {
        Python::attach(|py| {
            let obj = get_object(py);

            // One fewer than the thread-local batch threshold: pending decrefs below
            // the threshold must be buffered in the dropping thread's thread-local
            // buffer and not contend the global pool.
            let refs: Vec<Py<PyAny>> = (0..TLS_PENDING_DECREFS_CAPACITY - 1)
                .map(|_| obj.clone_ref(py))
                .collect();
            // Capture the address as `usize` (not `NonNull`) so that the closure
            // below can be `Ungil` while detached.
            let obj_addr = obj.as_ptr() as usize;

            py.detach(move || {
                drop(refs);

                // The dropped handles are buffered in this thread's TLS and must
                // not reach the global pool. (Tests run in parallel may push their
                // own objects into the global pool, so only assert that these
                // handles are absent from it.)
                assert!(!get_pool()
                    .pending_decrefs
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|ptr| ptr.as_ptr() as usize == obj_addr));
            });

            // The end of `detach` drains this thread's pending decrefs.
            assert_eq!(obj.get_refcnt(py), 1);
            assert!(pool_dec_refs_does_not_contain(&obj));
        });
    }

    #[test]
    #[cfg(not(pyo3_disable_reference_pool))]
    fn test_register_decref_flushes_full_tls_batch_to_global_pool() {
        Python::attach(|py| {
            let obj = get_object(py);

            // Exactly `TLS_PENDING_DECREFS_CAPACITY` handles dropped in one
            // detached region fill the buffer and flush it to the global pool.
            let refs: Vec<Py<PyAny>> = (0..TLS_PENDING_DECREFS_CAPACITY)
                .map(|_| obj.clone_ref(py))
                .collect();
            // The next handle, below the threshold, is buffered again.
            let extra = obj.clone_ref(py);

            py.detach(move || {
                drop(refs);

                // The buffer was flushed when it filled up, so this thread's
                // buffer is empty again. (This thread's own TLS is not touched
                // by the tests running in parallel, so this is race-free.)
                assert!(TLS_PENDING_DECREFS
                    .try_with(|pending| pending.borrow().0.is_empty())
                    .unwrap_or(false));

                drop(extra);

                // The handle dropped after the flush is buffered in TLS again.
                assert!(TLS_PENDING_DECREFS
                    .try_with(|pending| pending.borrow().0.len() == 1)
                    .unwrap_or(false));
            });

            // The end of `detach` drains both the flushed batch and the handle
            // buffered after it: all references are released exactly once.
            assert_eq!(obj.get_refcnt(py), 1);
            assert!(pool_dec_refs_does_not_contain(&obj));
        });
    }

    #[test]
    #[cfg(all(not(pyo3_disable_reference_pool), not(target_arch = "wasm32")))]
    fn test_detached_thread_flushes_tls_buffer_to_global_pool_at_exit() {
        let obj = Python::attach(|py| {
            let obj = get_object(py);
            let reference = obj.clone_ref(py);

            // Dropping below the batch threshold while detached buffers the
            // handle in the spawned thread's TLS; the buffer must be flushed to
            // the global pool when that thread exits.
            std::thread::spawn(move || drop(reference)).join().unwrap();

            // The handle must have been flushed to the global pool at thread exit.
            // A test running in parallel may have drained the pool in the meantime,
            // in which case the reference was already released — either way it was
            // not lost.
            let obj_ptr = unsafe { NonNull::new_unchecked(obj.as_ptr()) };
            let pending_in_pool = get_pool()
                .pending_decrefs
                .lock()
                .unwrap()
                .contains(&obj_ptr);
            assert!(
                pending_in_pool || obj.get_refcnt(py) == 1,
                "reference dropped on an exited thread was lost"
            );
            obj
        });

        // On next attach, the reference is released.
        Python::attach(|py| {
            // With free-threading, another thread could still be processing the
            // DECREFs for a batch it took from the pool, so the refcount is only
            // deterministic on builds with the GIL.
            #[cfg(not(Py_GIL_DISABLED))]
            assert_eq!(obj.get_refcnt(py), 1);
            assert!(pool_dec_refs_does_not_contain(&obj));
        });
    }

    #[test]
    fn test_attach_counts() {
        // Check `attach` and AttachGuard both increase counts correctly
        let get_attach_count = || ATTACH_COUNT.with(|c| c.get());

        assert_eq!(get_attach_count(), 0);
        Python::attach(|_| {
            assert_eq!(get_attach_count(), 1);

            let pool = unsafe { AttachGuard::assume() };
            assert_eq!(get_attach_count(), 2);

            let pool2 = unsafe { AttachGuard::assume() };
            assert_eq!(get_attach_count(), 3);

            drop(pool);
            assert_eq!(get_attach_count(), 2);

            Python::attach(|_| {
                // nested `attach` updates attach count
                assert_eq!(get_attach_count(), 3);
            });
            assert_eq!(get_attach_count(), 2);

            drop(pool2);
            assert_eq!(get_attach_count(), 1);
        });
        assert_eq!(get_attach_count(), 0);
    }

    #[test]
    fn test_detach() {
        assert!(!thread_is_attached());

        Python::attach(|py| {
            assert!(thread_is_attached());

            py.detach(move || {
                assert!(!thread_is_attached());

                Python::attach(|_| assert!(thread_is_attached()));

                assert!(!thread_is_attached());
            });

            assert!(thread_is_attached());
        });

        assert!(!thread_is_attached());
    }

    #[cfg(feature = "py-clone")]
    #[test]
    #[should_panic]
    fn test_detach_updates_refcounts() {
        Python::attach(|py| {
            // Make a simple object with 1 reference
            let obj = get_object(py);
            assert_eq!(obj.get_refcnt(py), 1);
            // Cloning the object when detached should panic
            py.detach(|| obj.clone());
        });
    }

    #[test]
    fn recursive_attach_ok() {
        Python::attach(|py| {
            let obj = Python::attach(|_| py.eval(c"object()", None, None).unwrap());
            assert_eq!(obj.get_refcnt(), 1);
        })
    }

    #[cfg(feature = "py-clone")]
    #[test]
    fn test_clone_attached() {
        Python::attach(|py| {
            let obj = get_object(py);
            let count = obj.get_refcnt(py);

            // Cloning when attached should increase reference count immediately
            #[expect(clippy::redundant_clone)]
            let c = obj.clone();
            assert_eq!(count + 1, c.get_refcnt(py));
        })
    }

    #[test]
    #[cfg(not(pyo3_disable_reference_pool))]
    fn test_drop_deferred_references_does_not_deadlock() {
        // drop_deferred_references can run arbitrary Python code during Py_DECREF.
        // if the locking is implemented incorrectly, it will deadlock.

        use crate::ffi;

        Python::attach(|py| {
            let obj = get_object(py);

            unsafe extern "C" fn capsule_drop(capsule: *mut ffi::PyObject) {
                // This line will implicitly call drop_deferred_references
                // -> and so cause deadlock if drop_deferred_references is not handling recursion correctly.
                let pool = unsafe { AttachGuard::assume() };

                // Rebuild obj so that it can be dropped
                unsafe {
                    use crate::Bound;

                    Bound::from_owned_ptr(
                        pool.python(),
                        ffi::PyCapsule_GetPointer(capsule, std::ptr::null()) as _,
                    )
                };
            }

            let ptr = obj.into_ptr();

            let capsule =
                unsafe { ffi::PyCapsule_New(ptr as _, std::ptr::null(), Some(capsule_drop)) };

            // Queue the capsule through the production entry point, so that the
            // capsule is drained from the thread-local buffer and the recursive
            // drain in `capsule_drop` re-enters `drop_deferred_references` while
            // the outer drain is in progress.
            unsafe { register_decref(NonNull::new(capsule).unwrap()) };

            // Updating the counts will call decref on the capsule, which calls capsule_drop
            drop_deferred_references(py);
        })
    }

    #[test]
    #[cfg(not(pyo3_disable_reference_pool))]
    fn test_reentrant_drain_tls_refill_is_not_clobbered() {
        Python::attach(|py| {
            let obj = get_object(py);
            // The capsule stores a second reference to `obj`, which its
            // destructor adopts and drops while detached.
            let ptr = obj.clone_ref(py).into_ptr();

            unsafe extern "C" fn capsule_drop(capsule: *mut ffi::PyObject) {
                // This callback runs from the drain's decref loop, while the
                // thread is attached.
                let pool = unsafe { AttachGuard::assume() };

                // Rebuild the handle (as a detach-safe `Py`, so that it may be
                // dropped while detached) so that it can be released.
                let handle = unsafe {
                    use crate::Bound;

                    Bound::from_owned_ptr(
                        pool.python(),
                        ffi::PyCapsule_GetPointer(capsule, std::ptr::null()) as _,
                    )
                    .unbind()
                };

                // Detach and drop the handle: while the outer drain is still in
                // progress, the handle is buffered in the thread-local storage,
                // and the end of this detach runs a nested drain which must
                // release it without the outer drain clobbering the buffer.
                pool.python().detach(move || drop(handle));
            }

            let capsule =
                unsafe { ffi::PyCapsule_New(ptr as _, std::ptr::null(), Some(capsule_drop)) };

            unsafe { register_decref(NonNull::new(capsule).unwrap()) };

            drop_deferred_references(py);

            // The capsule and the handle dropped by its destructor have both
            // been released exactly once.
            assert_eq!(obj.get_refcnt(py), 1);
            assert!(pool_dec_refs_does_not_contain(&obj));
        })
    }

    #[test]
    #[cfg(not(pyo3_disable_reference_pool))]
    fn test_attach_guard_drop_deferred_references() {
        Python::attach(|py| {
            let obj = get_object(py);

            // For AttachGuard::attach

            get_pool().register_decref(NonNull::new(obj.clone_ref(py).into_ptr()).unwrap());
            #[cfg(not(Py_GIL_DISABLED))]
            assert!(pool_dec_refs_contains(&obj));
            let _guard = AttachGuard::attach();
            assert!(pool_dec_refs_does_not_contain(&obj));

            // For AttachGuard::assume

            get_pool().register_decref(NonNull::new(obj.clone_ref(py).into_ptr()).unwrap());
            #[cfg(not(Py_GIL_DISABLED))]
            assert!(pool_dec_refs_contains(&obj));
            let _guard2 = unsafe { AttachGuard::assume() };
            assert!(pool_dec_refs_does_not_contain(&obj));
        })
    }
}
