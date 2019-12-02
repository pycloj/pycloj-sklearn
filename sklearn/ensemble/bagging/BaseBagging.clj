(ns sklearn.ensemble.bagging.BaseBagging
  "Base class for Bagging meta-estimator.

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
(defonce bagging (import-module "sklearn.ensemble.bagging"))

(defn BaseBagging 
  "Base class for Bagging meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    "
  [base_estimator & {:keys [n_estimators max_samples max_features bootstrap bootstrap_features oob_score warm_start n_jobs random_state verbose]
                       :or {n_estimators 10 max_samples 1.0 max_features 1.0 bootstrap true bootstrap_features false oob_score false warm_start false verbose 0}} ]
    (py/call-attr-kw bagging "BaseBagging" [base_estimator] {:n_estimators n_estimators :max_samples max_samples :max_features max_features :bootstrap bootstrap :bootstrap_features bootstrap_features :oob_score oob_score :warm_start warm_start :n_jobs n_jobs :random_state random_state :verbose verbose }))

(defn estimators-samples- 
  "The subset of drawn samples for each base estimator.

        Returns a dynamically generated list of indices identifying
        the samples used for fitting each member of the ensemble, i.e.,
        the in-bag samples.

        Note: the list is re-created at each call to the property in order
        to reduce the object memory footprint by not storing the sampling
        data. Thus fetching the property may be slower than expected.
        "
  [ self ]
    (py/call-attr self "estimators_samples_"))

(defn fit 
  "Build a Bagging ensemble of estimators from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

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
