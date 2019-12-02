(ns sklearn.linear-model.coordinate-descent.LinearModelCV
  "Base class for iterative model fitting along a regularization path"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce coordinate-descent (import-module "sklearn.linear_model.coordinate_descent"))

(defn LinearModelCV 
  "Base class for iterative model fitting along a regularization path"
  [ & {:keys [eps n_alphas alphas fit_intercept normalize precompute max_iter tol copy_X cv verbose n_jobs positive random_state selection]
       :or {eps 0.001 n_alphas 100 fit_intercept true normalize false precompute "auto" max_iter 1000 tol 0.0001 copy_X true cv "warn" verbose false positive false selection "cyclic"}} ]
  
   (py/call-attr-kw coordinate-descent "LinearModelCV" [] {:eps eps :n_alphas n_alphas :alphas alphas :fit_intercept fit_intercept :normalize normalize :precompute precompute :max_iter max_iter :tol tol :copy_X copy_X :cv cv :verbose verbose :n_jobs n_jobs :positive positive :random_state random_state :selection selection }))

(defn fit 
  "Fit linear model with coordinate descent

        Fit is on grid of alphas and best alpha estimated by cross-validation.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data. Pass directly as Fortran-contiguous data
            to avoid unnecessary memory duplication. If y is mono-output,
            X can be sparse.

        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values
        "
  [ self X y ]
  (py/call-attr self "fit"  self X y ))

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
  "Predict using the linear model

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
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
