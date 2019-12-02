(ns sklearn.svm.base.BaseLibSVM
  "Base class for estimators that use libsvm as backing library

    This implements support vector machine classification and regression.

    Parameter documentation is in the derived `SVC` class.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.svm.base"))

(defn BaseLibSVM 
  "Base class for estimators that use libsvm as backing library

    This implements support vector machine classification and regression.

    Parameter documentation is in the derived `SVC` class.
    "
  [ kernel degree gamma coef0 tol C nu epsilon shrinking probability cache_size class_weight verbose max_iter random_state ]
  (py/call-attr base "BaseLibSVM"  kernel degree gamma coef0 tol C nu epsilon shrinking probability cache_size class_weight verbose max_iter random_state ))

(defn coef- 
  ""
  [ self ]
    (py/call-attr self "coef_"))

(defn fit 
  "Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
            For kernel=\"precomputed\", the expected shape of X is
            (n_samples, n_samples).

        y : array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression)

        sample_weight : array-like, shape (n_samples,)
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : object

        Notes
        -----
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.
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

(defn predict 
  "Perform regression on samples in X.

        For an one-class model, +1 (inlier) or -1 (outlier) is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel=\"precomputed\", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : array, shape (n_samples,)
        "
  [ self X ]
  (py/call-attr self "predict"  self X ))

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
