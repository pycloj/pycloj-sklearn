(ns sklearn.linear-model.stochastic-gradient.BaseSGD
  "Base class for SGD classification and regression."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce stochastic-gradient (import-module "sklearn.linear_model.stochastic_gradient"))

(defn BaseSGD 
  "Base class for SGD classification and regression."
  [loss & {:keys [penalty alpha C l1_ratio fit_intercept max_iter tol shuffle verbose epsilon random_state learning_rate eta0 power_t early_stopping validation_fraction n_iter_no_change warm_start average]
                       :or {penalty "l2" alpha 0.0001 C 1.0 l1_ratio 0.15 fit_intercept true max_iter 1000 tol 0.001 shuffle true verbose 0 epsilon 0.1 learning_rate "optimal" eta0 0.0 power_t 0.5 early_stopping false validation_fraction 0.1 n_iter_no_change 5 warm_start false average false}} ]
    (py/call-attr-kw stochastic-gradient "BaseSGD" [loss] {:penalty penalty :alpha alpha :C C :l1_ratio l1_ratio :fit_intercept fit_intercept :max_iter max_iter :tol tol :shuffle shuffle :verbose verbose :epsilon epsilon :random_state random_state :learning_rate learning_rate :eta0 eta0 :power_t power_t :early_stopping early_stopping :validation_fraction validation_fraction :n_iter_no_change n_iter_no_change :warm_start warm_start :average average }))

(defn densify 
  "Convert coefficient matrix to dense array format.

        Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
        default format of ``coef_`` and is required for fitting, so calling
        this method is only required on models that have previously been
        sparsified; otherwise, it is a no-op.

        Returns
        -------
        self : estimator
        "
  [ self  ]
  (py/call-attr self "densify"  self  ))

(defn fit 
  "Fit model."
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

(defn set-params 
  ""
  [ self  ]
  (py/call-attr self "set_params"  self  ))

(defn sparsify 
  "Convert coefficient matrix to sparse format.

        Converts the ``coef_`` member to a scipy.sparse matrix, which for
        L1-regularized models can be much more memory- and storage-efficient
        than the usual numpy.ndarray representation.

        The ``intercept_`` member is not converted.

        Notes
        -----
        For non-sparse models, i.e. when there are not many zeros in ``coef_``,
        this may actually *increase* memory usage, so use this method with
        care. A rule of thumb is that the number of zero elements, which can
        be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
        to provide significant benefits.

        After calling this method, further fitting with the partial_fit
        method (if any) will not work until you call densify.

        Returns
        -------
        self : estimator
        "
  [ self  ]
  (py/call-attr self "sparsify"  self  ))
