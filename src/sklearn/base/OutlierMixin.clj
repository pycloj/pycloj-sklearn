(ns sklearn.base.OutlierMixin
  "Mixin class for all outlier detection estimators in scikit-learn."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.base"))

(defn OutlierMixin 
  "Mixin class for all outlier detection estimators in scikit-learn."
  [  ]
  (py/call-attr base "OutlierMixin"  ))
(defn fit-predict 
  "Performs fit on X and returns labels for X.

        Returns -1 for outliers and 1 for inliers.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : Ignored
            not used, present for API consistency by convention.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            1 for inliers, -1 for outliers.
        "
  [self X  & {:keys [y]} ]
    (py/call-attr-kw self "fit_predict" [X] {:y y }))