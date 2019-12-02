(ns sklearn
  "
Machine learning module for Python
==================================

sklearn is a Python module integrating classical machine
learning algorithms in the tightly-knit world of scientific Python
packages (numpy, scipy, matplotlib).

It aims to provide simple and efficient solutions to learning problems
that are accessible to everybody and reusable in various contexts:
machine-learning as a versatile tool for science and engineering.

See http://scikit-learn.org for complete documentation.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce sklearn (import-module "sklearn"))

(defn clone 
  "Constructs a new estimator with the same parameters.

    Clone does a deep copy of the model in an estimator
    without actually copying attached data. It yields a new estimator
    with the same parameters that has not been fit on any data.

    Parameters
    ----------
    estimator : estimator object, or list, tuple or set of objects
        The estimator or group of estimators to be cloned

    safe : boolean, optional
        If safe is false, clone will fall back to a deep copy on objects
        that are not estimators.

    "
  [estimator & {:keys [safe]
                       :or {safe true}} ]
    (py/call-attr-kw sklearn "clone" [estimator] {:safe safe }))

(defn config-context 
  "Context manager for global scikit-learn configuration

    Parameters
    ----------
    assume_finite : bool, optional
        If True, validation for finiteness will be skipped,
        saving time, but leading to potential crashes. If
        False, validation for finiteness will be performed,
        avoiding error.  Global default: False.

    working_memory : int, optional
        If set, scikit-learn will attempt to limit the size of temporary arrays
        to this number of MiB (per job when parallelised), often saving both
        computation time and memory on expensive operations that can be
        performed in chunks. Global default: 1024.

    Notes
    -----
    All settings, not just those presently modified, will be returned to
    their previous values when the context manager is exited. This is not
    thread-safe.

    Examples
    --------
    >>> import sklearn
    >>> from sklearn.utils.validation import assert_all_finite
    >>> with sklearn.config_context(assume_finite=True):
    ...     assert_all_finite([float('nan')])
    >>> with sklearn.config_context(assume_finite=True):
    ...     with sklearn.config_context(assume_finite=False):
    ...         assert_all_finite([float('nan')])
    ... # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Input contains NaN, ...

    See Also
    --------
    set_config: Set global scikit-learn configuration
    get_config: Retrieve current values of the global configuration
    "
  [  ]
  (py/call-attr sklearn "config_context"  ))

(defn get-config 
  "Retrieve current values for configuration set by :func:`set_config`

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context: Context manager for global scikit-learn configuration
    set_config: Set global scikit-learn configuration
    "
  [  ]
  (py/call-attr sklearn "get_config"  ))

(defn set-config 
  "Set global scikit-learn configuration

    .. versionadded:: 0.19

    Parameters
    ----------
    assume_finite : bool, optional
        If True, validation for finiteness will be skipped,
        saving time, but leading to potential crashes. If
        False, validation for finiteness will be performed,
        avoiding error.  Global default: False.

        .. versionadded:: 0.19

    working_memory : int, optional
        If set, scikit-learn will attempt to limit the size of temporary arrays
        to this number of MiB (per job when parallelised), often saving both
        computation time and memory on expensive operations that can be
        performed in chunks. Global default: 1024.

        .. versionadded:: 0.20

    print_changed_only : bool, optional
        If True, only the parameters that were set to non-default
        values will be printed when printing an estimator. For example,
        ``print(SVC())`` while True will only print 'SVC()' while the default
        behaviour would be to print 'SVC(C=1.0, cache_size=200, ...)' with
        all the non-changed parameters.

        .. versionadded:: 0.21

    See Also
    --------
    config_context: Context manager for global scikit-learn configuration
    get_config: Retrieve current values of the global configuration
    "
  [ assume_finite working_memory print_changed_only ]
  (py/call-attr sklearn "set_config"  assume_finite working_memory print_changed_only ))

(defn setup-module 
  "Fixture for the tests to assure globally controllable seeding of RNGs"
  [ module ]
  (py/call-attr sklearn "setup_module"  module ))

(defn show-versions 
  "Print useful debugging information"
  [  ]
  (py/call-attr sklearn "show_versions"  ))
