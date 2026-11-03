#[cfg(feature = "experimental-inspect")]
use crate::inspect::types::TypeInfo;
#[cfg(feature = "experimental-inspect")]
use crate::inspect::{type_hint_subscript, PyStaticExpr};
#[cfg(feature = "experimental-inspect")]
use crate::type_object::PyTypeInfo;
use crate::{
    conversion::{FromPyObjectOwned, IntoPyObject},
    instance::Bound,
    types::{any::PyAnyMethods, dict::PyDictMethods, PyDict},
    Borrowed, FromPyObject, PyAny, PyErr, Python,
};
use std::{cmp, collections, hash};

impl<'py, K, V, H> IntoPyObject<'py> for collections::HashMap<K, V, H>
where
    K: IntoPyObject<'py> + cmp::Eq + hash::Hash,
    V: IntoPyObject<'py>,
    H: hash::BuildHasher,
{
    type Target = PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    #[cfg(feature = "experimental-inspect")]
    const OUTPUT_TYPE: PyStaticExpr =
        type_hint_subscript!(PyDict::TYPE_HINT, K::OUTPUT_TYPE, V::OUTPUT_TYPE);

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        for (k, v) in self {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_output() -> TypeInfo {
        TypeInfo::dict_of(K::type_output(), V::type_output())
    }
}

impl<'a, 'py, K, V, H> IntoPyObject<'py> for &'a collections::HashMap<K, V, H>
where
    &'a K: IntoPyObject<'py> + cmp::Eq + hash::Hash,
    &'a V: IntoPyObject<'py>,
    H: hash::BuildHasher,
{
    type Target = PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    #[cfg(feature = "experimental-inspect")]
    const OUTPUT_TYPE: PyStaticExpr =
        type_hint_subscript!(PyDict::TYPE_HINT, <&K>::OUTPUT_TYPE, <&V>::OUTPUT_TYPE);

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        for (k, v) in self {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_output() -> TypeInfo {
        TypeInfo::dict_of(<&K>::type_output(), <&V>::type_output())
    }
}

impl<'py, K, V> IntoPyObject<'py> for collections::BTreeMap<K, V>
where
    K: IntoPyObject<'py> + cmp::Eq,
    V: IntoPyObject<'py>,
{
    type Target = PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    #[cfg(feature = "experimental-inspect")]
    const OUTPUT_TYPE: PyStaticExpr =
        type_hint_subscript!(PyDict::TYPE_HINT, K::OUTPUT_TYPE, V::OUTPUT_TYPE);

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        for (k, v) in self {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_output() -> TypeInfo {
        TypeInfo::dict_of(K::type_output(), V::type_output())
    }
}

impl<'a, 'py, K, V> IntoPyObject<'py> for &'a collections::BTreeMap<K, V>
where
    &'a K: IntoPyObject<'py> + cmp::Eq,
    &'a V: IntoPyObject<'py>,
    K: 'a,
    V: 'a,
{
    type Target = PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    #[cfg(feature = "experimental-inspect")]
    const OUTPUT_TYPE: PyStaticExpr =
        type_hint_subscript!(PyDict::TYPE_HINT, <&K>::OUTPUT_TYPE, <&V>::OUTPUT_TYPE);

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        for (k, v) in self {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_output() -> TypeInfo {
        TypeInfo::dict_of(<&K>::type_output(), <&V>::type_output())
    }
}

/// Extracts the `(key, value)` pairs of a dict with borrowed
/// (reference-free) access for entry shapes whose extraction is known not to
/// execute arbitrary Python code (see `FromPyObject::borrowed_item_extractor`),
/// feeding them into `insert`; other entries fall back to the owned
/// (reference-counting) path for just that entry.
///
/// Soundness: the per-entry gates guarantee that the "pure" extraction cannot
/// re-enter the interpreter while the borrowed pointers are live, so the dict
/// cannot be mutated and the pointers stay valid. Fallback entries take owned
/// references *before* running the (potentially re-entrant) extraction. Any
/// mutation performed by that re-entrant code is caught by the underlying
/// iterator's size-change guards, exactly like the owned `dict.iter()` path.
#[cfg(all(
    not(Py_LIMITED_API),
    not(PyPy),
    not(GraalPy),
    not(Py_GIL_DISABLED)
))]
#[inline(never)]
pub(crate) fn extract_borrowed_dict_into<'py, K, V, KE, VE, F>(
    dict: &Bound<'py, PyDict>,
    _key_extractor: KE,
    _value_extractor: VE,
    mut insert: F,
) -> Result<(), PyErr>
where
    K: FromPyObjectOwned<'py>,
    V: FromPyObjectOwned<'py>,
    KE: crate::conversion::PureItemExtract<Target = K>,
    VE: crate::conversion::PureItemExtract<Target = V>,
    F: FnMut(K, V),
{
    // Safety: the per-entry gates below guarantee that pure extraction
    // cannot re-enter the interpreter while the borrowed pointers are live;
    // fallback entries take owned references before running any re-entrant
    // code.
    for (key, value) in unsafe { dict.as_borrowed().iter_borrowed_guarded() } {
        let k_res = KE::extract_pure(key);
        let v_res = VE::extract_pure(value);
        match (k_res, v_res) {
            (Err(err), _) => return Err(err),
            (_, Err(err)) => return Err(err),
            (Ok(Some(k)), Ok(Some(v))) => insert(k, v),
            (Ok(k_opt), Ok(v_opt)) => {
                // At least one side is not pure-eligible for this entry:
                // take owned references first (so that re-entrant extraction
                // cannot invalidate them), then extract the missing side(s)
                // via the ordinary owned path.
                let key_owned = key.to_owned();
                let value_owned = value.to_owned();
                let k = match k_opt {
                    Some(k) => k,
                    None => match key_owned.extract::<K>() {
                        Ok(k) => k,
                        Err(err) => return Err(err.into()),
                    },
                };
                let v = match v_opt {
                    Some(v) => v,
                    None => match value_owned.extract::<V>() {
                        Ok(v) => v,
                        Err(err) => return Err(err.into()),
                    },
                };
                insert(k, v);
            }
        }
    }
    Ok(())
}

impl<'py, K, V, S> FromPyObject<'_, 'py> for collections::HashMap<K, V, S>
where
    K: FromPyObjectOwned<'py> + cmp::Eq + hash::Hash,
    V: FromPyObjectOwned<'py>,
    S: hash::BuildHasher + Default,
{
    type Error = PyErr;

    #[cfg(feature = "experimental-inspect")]
    const INPUT_TYPE: PyStaticExpr =
        type_hint_subscript!(&PyDict::TYPE_HINT, K::INPUT_TYPE, V::INPUT_TYPE);

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let dict = ob.cast::<PyDict>()?;
        let mut ret = collections::HashMap::with_capacity_and_hasher(dict.len(), S::default());
        // Borrowed fast path (no incref/decref per entry) when both the key
        // and value types have a pure item extractor.
        #[cfg(all(
            not(Py_LIMITED_API),
            not(PyPy),
            not(GraalPy),
            not(Py_GIL_DISABLED)
        ))]
        if let (Some(key_extractor), Some(value_extractor)) = (
            K::borrowed_item_extractor(crate::conversion::private::Token),
            V::borrowed_item_extractor(crate::conversion::private::Token),
        ) {
            extract_borrowed_dict_into(
                &dict,
                key_extractor,
                value_extractor,
                |k, v| {
                    ret.insert(k, v);
                },
            )?;
            return Ok(ret);
        }
        for (k, v) in dict.iter() {
            ret.insert(
                k.extract().map_err(Into::into)?,
                v.extract().map_err(Into::into)?,
            );
        }
        Ok(ret)
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_input() -> TypeInfo {
        TypeInfo::mapping_of(K::type_input(), V::type_input())
    }
}

impl<'py, K, V> FromPyObject<'_, 'py> for collections::BTreeMap<K, V>
where
    K: FromPyObjectOwned<'py> + cmp::Ord,
    V: FromPyObjectOwned<'py>,
{
    type Error = PyErr;

    #[cfg(feature = "experimental-inspect")]
    const INPUT_TYPE: PyStaticExpr =
        type_hint_subscript!(PyDict::TYPE_HINT, K::INPUT_TYPE, V::INPUT_TYPE);

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, PyErr> {
        let dict = ob.cast::<PyDict>()?;
        let mut ret = collections::BTreeMap::new();
        // Borrowed fast path (no incref/decref per entry) when both the key
        // and value types have a pure item extractor.
        #[cfg(all(
            not(Py_LIMITED_API),
            not(PyPy),
            not(GraalPy),
            not(Py_GIL_DISABLED)
        ))]
        if let (Some(key_extractor), Some(value_extractor)) = (
            K::borrowed_item_extractor(crate::conversion::private::Token),
            V::borrowed_item_extractor(crate::conversion::private::Token),
        ) {
            extract_borrowed_dict_into(
                &dict,
                key_extractor,
                value_extractor,
                |k, v| {
                    ret.insert(k, v);
                },
            )?;
            return Ok(ret);
        }
        for (k, v) in dict.iter() {
            ret.insert(
                k.extract().map_err(Into::into)?,
                v.extract().map_err(Into::into)?,
            );
        }
        Ok(ret)
    }

    #[cfg(feature = "experimental-inspect")]
    fn type_input() -> TypeInfo {
        TypeInfo::mapping_of(K::type_input(), V::type_input())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn test_extract_hashmap_of_string_int() {
        Python::attach(|py| {
            let obj = py.eval(c"{'a': 1, 'b': 2}", None, None).unwrap();
            let map: HashMap<String, i64> = obj.extract().unwrap();
            assert_eq!(map.get("a"), Some(&1));
            assert_eq!(map.get("b"), Some(&2));
        });
    }

    #[test]
    fn test_extract_hashmap_mixed_shapes() {
        Python::attach(|py| {
            // exact `int` values alongside an `__index__` object: the entry
            // with the non-exact value falls back to the owned path.
            let globals = PyDict::new(py);
            py.run(
                c"class IntLike:
    def __index__(self):
        return 42
d = {'a': 1, 'b': IntLike(), 'c': 3}",
                Some(&globals),
                Some(&globals),
            )
            .unwrap();
            let obj = py.eval(c"d", Some(&globals), Some(&globals)).unwrap();
            let map: HashMap<String, i64> = obj.extract().unwrap();
            assert_eq!(map.get("a"), Some(&1));
            assert_eq!(map.get("b"), Some(&42));
            assert_eq!(map.get("c"), Some(&3));
        });
    }

    #[test]
    fn test_extract_btreemap_mixed_shapes() {
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(
                c"class IntLike:
    def __index__(self):
        return 42
d = {'a': 1, 'b': IntLike(), 'c': 3}",
                Some(&globals),
                Some(&globals),
            )
            .unwrap();
            let obj = py.eval(c"d", Some(&globals), Some(&globals)).unwrap();
            let map: BTreeMap<String, i64> = obj.extract().unwrap();
            assert_eq!(map.get("a"), Some(&1));
            assert_eq!(map.get("b"), Some(&42));
            assert_eq!(map.get("c"), Some(&3));
        });
    }

    #[test]
    #[should_panic(expected = "dictionary changed size during iteration")]
    fn test_extract_hashmap_dict_mutated_during_extract() {
        Python::attach(|py| {
            // re-entrant value extraction mutates the dict; the borrowed
            // iterator must detect it exactly like the owned iterator.
            let globals = PyDict::new(py);
            py.run(
                c"class Evil:
    def __index__(self):
        del d['b']
        return 1
d = {'a': Evil(), 'b': Evil(), 'c': Evil()}",
                Some(&globals),
                Some(&globals),
            )
            .unwrap();
            let obj = py.eval(c"d", Some(&globals), Some(&globals)).unwrap();
            let _: HashMap<String, i64> = obj.extract().unwrap();
        });
    }

    #[test]
    fn test_hashmap_to_python() {
        Python::attach(|py| {
            let mut map = HashMap::<i32, i32>::new();
            map.insert(1, 1);

            let py_map = (&map).into_pyobject(py).unwrap();

            assert_eq!(py_map.len(), 1);
            assert!(
                py_map
                    .get_item(1)
                    .unwrap()
                    .unwrap()
                    .extract::<i32>()
                    .unwrap()
                    == 1
            );
            assert_eq!(map, py_map.extract().unwrap());
        });
    }

    #[test]
    fn test_btreemap_to_python() {
        Python::attach(|py| {
            let mut map = BTreeMap::<i32, i32>::new();
            map.insert(1, 1);

            let py_map = (&map).into_pyobject(py).unwrap();

            assert_eq!(py_map.len(), 1);
            assert!(
                py_map
                    .get_item(1)
                    .unwrap()
                    .unwrap()
                    .extract::<i32>()
                    .unwrap()
                    == 1
            );
            assert_eq!(map, py_map.extract().unwrap());
        });
    }

    #[test]
    fn test_hashmap_into_python() {
        Python::attach(|py| {
            let mut map = HashMap::<i32, i32>::new();
            map.insert(1, 1);

            let py_map = map.into_pyobject(py).unwrap();

            assert_eq!(py_map.len(), 1);
            assert!(
                py_map
                    .get_item(1)
                    .unwrap()
                    .unwrap()
                    .extract::<i32>()
                    .unwrap()
                    == 1
            );
        });
    }

    #[test]
    fn test_btreemap_into_py() {
        Python::attach(|py| {
            let mut map = BTreeMap::<i32, i32>::new();
            map.insert(1, 1);

            let py_map = map.into_pyobject(py).unwrap();

            assert_eq!(py_map.len(), 1);
            assert!(
                py_map
                    .get_item(1)
                    .unwrap()
                    .unwrap()
                    .extract::<i32>()
                    .unwrap()
                    == 1
            );
        });
    }
}
