(ns sklearn.base
  "Base classes for all estimators."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.base"))

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
  [ & {:keys [estimator safe]
       :or {safe true}} ]
  
   (py/call-attr-kw base "clone" [] {:estimator estimator :safe safe }))

(defn is-classifier 
  "Returns True if the given estimator is (probably) a classifier.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a classifier and False otherwise.
    "
  [ & {:keys [estimator]} ]
   (py/call-attr-kw base "is_classifier" [] {:estimator estimator }))

(defn is-outlier-detector 
  "Returns True if the given estimator is (probably) an outlier detector.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is an outlier detector and False otherwise.
    "
  [ & {:keys [estimator]} ]
   (py/call-attr-kw base "is_outlier_detector" [] {:estimator estimator }))

(defn is-regressor 
  "Returns True if the given estimator is (probably) a regressor.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a regressor and False otherwise.
    "
  [ & {:keys [estimator]} ]
   (py/call-attr-kw base "is_regressor" [] {:estimator estimator }))
