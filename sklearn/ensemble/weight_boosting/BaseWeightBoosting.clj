(ns sklearn.ensemble.weight-boosting.BaseWeightBoosting
  "Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce weight-boosting (import-module "sklearn.ensemble.weight_boosting"))

(defn BaseWeightBoosting 
  "Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    "
  [base_estimator & {:keys [n_estimators estimator_params learning_rate random_state]
                       :or {n_estimators 50 estimator_params () learning_rate 1.0}} ]
    (py/call-attr-kw weight-boosting "BaseWeightBoosting" [base_estimator] {:n_estimators n_estimators :estimator_params estimator_params :learning_rate learning_rate :random_state random_state }))

(defn feature-importances- 
  "Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        "
  [ self ]
    (py/call-attr self "feature_importances_"))

(defn fit 
  "Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
        "
  [ self X y sample_weight ]
  (py/call-attr self "fit"  self X y sample_weight ))

(defn get-params 
  "Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        "
  [self  & {:keys [deep]
                       :or {deep true}} ]
    (py/call-attr-kw self "get_params" [] {:deep deep }))

(defn set-params 
  "Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        "
  [ self  ]
  (py/call-attr self "set_params"  self  ))

(defn staged-score 
  "Return staged scores for X, y.

        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like, shape = [n_samples]
            Labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        z : float
        "
  [ self X y sample_weight ]
  (py/call-attr self "staged_score"  self X y sample_weight ))
