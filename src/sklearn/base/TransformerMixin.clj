(ns sklearn.base.TransformerMixin
  "Mixin class for all transformers in scikit-learn."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.base"))

(defn TransformerMixin 
  "Mixin class for all transformers in scikit-learn."
  [  ]
  (py/call-attr base "TransformerMixin"  ))
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
  [self X  & {:keys [y]} ]
    (py/call-attr-kw self "fit_transform" [X] {:y y }))
