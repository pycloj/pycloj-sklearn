(ns sklearn.cluster.birch.ClusterMixin
  "Mixin class for all cluster estimators in scikit-learn."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce birch (import-module "sklearn.cluster.birch"))

(defn ClusterMixin 
  "Mixin class for all cluster estimators in scikit-learn."
  [  ]
  (py/call-attr birch "ClusterMixin"  ))

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
  [ self X y ]
  (py/call-attr self "fit_predict"  self X y ))
