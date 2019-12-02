(ns sklearn.linear-model.Lars
  "Least Angle Regression model a.k.a. LAR

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    fit_intercept : boolean
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    verbose : boolean or integer, optional
        Sets the verbosity amount

    normalize : boolean, optional, default True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    n_nonzero_coefs : int, optional
        Target number of non-zero coefficients. Use ``np.inf`` for no limit.

    eps : float, optional
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    fit_path : boolean
        If True the full path is stored in the ``coef_path_`` attribute.
        If you compute the solution for a large problem or many targets,
        setting ``fit_path`` to ``False`` will lead to a speedup, especially
        with a small alpha.

    positive : boolean (default=False)
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.

        .. deprecated:: 0.20

            The option is broken and deprecated. It will be removed in v0.22.

    Attributes
    ----------
    alphas_ : array, shape (n_alphas + 1,) | list of n_targets such arrays
        Maximum of covariances (in absolute value) at each iteration.         ``n_alphas`` is either ``n_nonzero_coefs`` or ``n_features``,         whichever is smaller.

    active_ : list, length = n_alphas | list of n_targets such lists
        Indices of active variables at the end of the path.

    coef_path_ : array, shape (n_features, n_alphas + 1)         | list of n_targets such arrays
        The varying values of the coefficients along the path. It is not
        present if the ``fit_path`` parameter is ``False``.

    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the formulation formula).

    intercept_ : float | array, shape (n_targets,)
        Independent term in decision function.

    n_iter_ : array-like or int
        The number of iterations taken by lars_path to find the
        grid of alphas for each target.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> reg = linear_model.Lars(n_nonzero_coefs=1)
    >>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111])
    ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    Lars(copy_X=True, eps=..., fit_intercept=True, fit_path=True,
       n_nonzero_coefs=1, normalize=True, positive=False, precompute='auto',
       verbose=False)
    >>> print(reg.coef_) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    [ 0. -1.11...]

    See also
    --------
    lars_path, LarsCV
    sklearn.decomposition.sparse_encode

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce linear-model (import-module "sklearn.linear_model"))

(defn Lars 
  "Least Angle Regression model a.k.a. LAR

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    fit_intercept : boolean
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    verbose : boolean or integer, optional
        Sets the verbosity amount

    normalize : boolean, optional, default True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    n_nonzero_coefs : int, optional
        Target number of non-zero coefficients. Use ``np.inf`` for no limit.

    eps : float, optional
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    fit_path : boolean
        If True the full path is stored in the ``coef_path_`` attribute.
        If you compute the solution for a large problem or many targets,
        setting ``fit_path`` to ``False`` will lead to a speedup, especially
        with a small alpha.

    positive : boolean (default=False)
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.

        .. deprecated:: 0.20

            The option is broken and deprecated. It will be removed in v0.22.

    Attributes
    ----------
    alphas_ : array, shape (n_alphas + 1,) | list of n_targets such arrays
        Maximum of covariances (in absolute value) at each iteration.         ``n_alphas`` is either ``n_nonzero_coefs`` or ``n_features``,         whichever is smaller.

    active_ : list, length = n_alphas | list of n_targets such lists
        Indices of active variables at the end of the path.

    coef_path_ : array, shape (n_features, n_alphas + 1)         | list of n_targets such arrays
        The varying values of the coefficients along the path. It is not
        present if the ``fit_path`` parameter is ``False``.

    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the formulation formula).

    intercept_ : float | array, shape (n_targets,)
        Independent term in decision function.

    n_iter_ : array-like or int
        The number of iterations taken by lars_path to find the
        grid of alphas for each target.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> reg = linear_model.Lars(n_nonzero_coefs=1)
    >>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111])
    ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    Lars(copy_X=True, eps=..., fit_intercept=True, fit_path=True,
       n_nonzero_coefs=1, normalize=True, positive=False, precompute='auto',
       verbose=False)
    >>> print(reg.coef_) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    [ 0. -1.11...]

    See also
    --------
    lars_path, LarsCV
    sklearn.decomposition.sparse_encode

    "
  [ & {:keys [fit_intercept verbose normalize precompute n_nonzero_coefs eps copy_X fit_path positive]
       :or {fit_intercept true verbose false normalize true precompute "auto" n_nonzero_coefs 500 eps 2.220446049250313e-16 copy_X true fit_path true positive false}} ]
  
   (py/call-attr-kw linear-model "Lars" [] {:fit_intercept fit_intercept :verbose verbose :normalize normalize :precompute precompute :n_nonzero_coefs n_nonzero_coefs :eps eps :copy_X copy_X :fit_path fit_path :positive positive }))

(defn fit 
  "Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Xy : array-like, shape (n_samples,) or (n_samples, n_targets),                 optional
            Xy = np.dot(X.T, y) that can be precomputed. It is useful
            only when the Gram matrix is precomputed.

        Returns
        -------
        self : object
            returns an instance of self.
        "
  [ self X y Xy ]
  (py/call-attr self "fit"  self X y Xy ))

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

(defn score 
  "Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples. For some estimators this may be a
            precomputed kernel matrix instead, shape = (n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for the estimator.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.

        Notes
        -----
        The R2 score used when calling ``score`` on a regressor will use
        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
        with `metrics.r2_score`. This will influence the ``score`` method of
        all the multioutput regressors (except for
        `multioutput.MultiOutputRegressor`). To specify the default value
        manually and avoid the warning, please either call `metrics.r2_score`
        directly or make a custom scorer with `metrics.make_scorer` (the
        built-in scorer ``'r2'`` uses ``multioutput='uniform_average'``).
        "
  [ self X y sample_weight ]
  (py/call-attr self "score"  self X y sample_weight ))

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
