(ns sklearn.base.ClusterMixin
  "Mixin class for all cluster estimators in scikit-learn."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.base"))

(defn ClusterMixin 
  "Mixin class for all cluster estimators in scikit-learn."
  [  ]
  (py/call-attr base "ClusterMixin"   ))

(defn fit-predict 
  "Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : Ignored
            not used, present for API consistency by convention.

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            cluster labels
        "
  [self  & {:keys [X y]} ]
    (py/call-attr-kw base "fit_predict" [self] {:X X :y y }))
