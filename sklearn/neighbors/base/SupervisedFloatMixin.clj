(ns sklearn.neighbors.base.SupervisedFloatMixin
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.neighbors.base"))

(defn SupervisedFloatMixin 
  ""
  [  ]
  (py/call-attr base "SupervisedFloatMixin"  ))

(defn fit 
  "Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.

        y : {array-like, sparse matrix}
            Target values, array of float values, shape = [n_samples]
             or [n_samples, n_outputs]
        "
  [ self X y ]
  (py/call-attr self "fit"  self X y ))
