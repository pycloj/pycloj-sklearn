(ns sklearn.feature-extraction.hashing.TransformerMixin
  "Mixin class for all transformers in scikit-learn."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce hashing (import-module "sklearn.feature_extraction.hashing"))

(defn TransformerMixin 
  "Mixin class for all transformers in scikit-learn."
  [  ]
  (py/call-attr hashing "TransformerMixin"  ))

(defn fit-transform 
  "Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        "
  [ self X y ]
  (py/call-attr self "fit_transform"  self X y ))
