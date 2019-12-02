(ns sklearn.ensemble.gradient-boosting.MeanEstimator
  "An estimator predicting the mean of the training targets."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce gradient-boosting (import-module "sklearn.ensemble.gradient_boosting"))

(defn MeanEstimator 
  "An estimator predicting the mean of the training targets."
  [  ]
  (py/call-attr gradient-boosting "MeanEstimator"  ))

(defn fit 
  "Fit the estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data

        y : array, shape (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape (n_samples,)
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