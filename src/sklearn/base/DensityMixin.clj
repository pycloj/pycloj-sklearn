(ns sklearn.base.DensityMixin
  "Mixin class for all density estimators in scikit-learn."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.base"))

(defn DensityMixin 
  "Mixin class for all density estimators in scikit-learn."
  [  ]
  (py/call-attr base "DensityMixin"   ))

(defn score 
  "Returns the score of the model on the data X

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        score : float
        "
  [self  & {:keys [X y]} ]
    (py/call-attr-kw base "score" [self] {:X X :y y }))
