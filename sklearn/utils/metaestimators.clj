(ns sklearn.utils.metaestimators
  "Utilities for meta-estimators"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce metaestimators (import-module "sklearn.utils.metaestimators"))

(defn abstractmethod 
  "A decorator indicating abstract methods.

    Requires that the metaclass is ABCMeta or derived from it.  A
    class that has a metaclass derived from ABCMeta cannot be
    instantiated unless all of its abstract methods are overridden.
    The abstract methods can be called using any of the normal
    'super' call mechanisms.

    Usage:

        class C(metaclass=ABCMeta):
            @abstractmethod
            def my_abstract_method(self, ...):
                ...
    "
  [ funcobj ]
  (py/call-attr metaestimators "abstractmethod"  funcobj ))

(defn if-delegate-has-method 
  "Create a decorator for methods that are delegated to a sub-estimator

    This enables ducktyping by hasattr returning True according to the
    sub-estimator.

    Parameters
    ----------
    delegate : string, list of strings or tuple of strings
        Name of the sub-estimator that can be accessed as an attribute of the
        base object. If a list or a tuple of names are provided, the first
        sub-estimator that is an attribute of the base object will be used.

    "
  [ delegate ]
  (py/call-attr metaestimators "if_delegate_has_method"  delegate ))

(defn safe-indexing 
  "Return items or rows from X using indices.

    Allows simple indexing of lists or arrays.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series.
        Data from which to sample rows or items.
    indices : array-like of int
        Indices according to which X will be subsampled.

    Returns
    -------
    subset
        Subset of X on first axis

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    "
  [ X indices ]
  (py/call-attr metaestimators "safe_indexing"  X indices ))

(defn update-wrapper 
  "Update a wrapper function to look like the wrapped function

       wrapper is the function to be updated
       wrapped is the original function
       assigned is a tuple naming the attributes assigned directly
       from the wrapped function to the wrapper function (defaults to
       functools.WRAPPER_ASSIGNMENTS)
       updated is a tuple naming the attributes of the wrapper that
       are updated with the corresponding attribute from the wrapped
       function (defaults to functools.WRAPPER_UPDATES)
    "
  [wrapper wrapped & {:keys [assigned updated]
                       :or {assigned ('__module__', '__name__', '__qualname__', '__doc__', '__annotations__') updated ('__dict__',)}} ]
    (py/call-attr-kw metaestimators "update_wrapper" [wrapper wrapped] {:assigned assigned :updated updated }))
