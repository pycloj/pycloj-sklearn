(ns sklearn.ensemble.gradient-boosting.ZeroEstimator
  "An estimator that simply predicts zero.

    .. deprecated:: 0.21
        Using ``ZeroEstimator`` or ``init='zero'`` is deprecated in version
        0.21 and will be removed in version 0.23.

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce gradient-boosting (import-module "sklearn.ensemble.gradient_boosting"))

(defn ZeroEstimator 
  "An estimator that simply predicts zero.

    .. deprecated:: 0.21
        Using ``ZeroEstimator`` or ``init='zero'`` is deprecated in version
        0.21 and will be removed in version 0.23.

    "
  [  ]
  (py/call-attr gradient-boosting "ZeroEstimator"  ))

(defn fit 
  "Fit the estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data

        y : numpy, shape (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary

        sample_weight : array, shape (n_samples,)
            Individual weights for each sample
        "
  [ self X y sample_weight ]
  (py/call-attr self "fit"  self X y sample_weight ))

(defn predict 
  "Predict labels

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y : array, shape (n_samples,)
            Returns predicted values.
        "
  [ self X ]
  (py/call-attr self "predict"  self X ))

(defn predict-proba 
  ""
  [ self X ]
  (py/call-attr self "predict_proba"  self X ))
