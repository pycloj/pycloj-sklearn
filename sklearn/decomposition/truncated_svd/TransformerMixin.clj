(ns sklearn.decomposition.truncated-svd.TransformerMixin
  "Mixin class for all transformers in scikit-learn."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce truncated-svd (import-module "sklearn.decomposition.truncated_svd"))

(defn TransformerMixin 
  "Mixin class for all transformers in scikit-learn."
  [  ]
  (py/call-attr truncated-svd "TransformerMixin"  ))

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
