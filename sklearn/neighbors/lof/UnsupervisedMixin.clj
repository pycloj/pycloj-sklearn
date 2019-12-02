(ns sklearn.neighbors.lof.UnsupervisedMixin
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lof (import-module "sklearn.neighbors.lof"))

(defn UnsupervisedMixin 
  ""
  [  ]
  (py/call-attr lof "UnsupervisedMixin"  ))

(defn fit 
  "Fit the model using X as training data

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.
        "
  [ self X y ]
  (py/call-attr self "fit"  self X y ))
