(ns sklearn.linear-model.least-angle.LarsCV
  "Cross-validated Least Angle Regression model.

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    verbose : boolean or integer, optional
        Sets the verbosity amount

    max_iter : integer, optional
        Maximum number of iterations to perform.

    normalize : boolean, optional, default True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram matrix
        cannot be passed as argument since we will use only subsets of X.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.20
            ``cv`` default value if None will change from 3-fold to 5-fold
            in v0.22.

    max_n_alphas : integer, optional
        The maximum number of points on the path used to compute the
        residuals in the cross-validation

    n_jobs : int or None, optional (default=None)
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    eps : float, optional
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    positive : boolean (default=False)
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.

        .. deprecated:: 0.20
            The option is broken and deprecated. It will be removed in v0.22.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the formulation formula)

    intercept_ : float
        independent term in decision function

    coef_path_ : array, shape (n_features, n_alphas)
        the varying values of the coefficients along the path

    alpha_ : float
        the estimated regularization parameter alpha

    alphas_ : array, shape (n_alphas,)
        the different values of alpha along the path

    cv_alphas_ : array, shape (n_cv_alphas,)
        all the values of alpha along the path for the different folds

    mse_path_ : array, shape (n_folds, n_cv_alphas)
        the mean square error on left-out for each fold along the path
        (alpha values given by ``cv_alphas``)

    n_iter_ : array-like or int
        the number of iterations run by Lars with the optimal alpha.

    Examples
    --------
    >>> from sklearn.linear_model import LarsCV
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, noise=4.0, random_state=0)
    >>> reg = LarsCV(cv=5).fit(X, y)
    >>> reg.score(X, y) # doctest: +ELLIPSIS
    0.9996...
    >>> reg.alpha_
    0.0254...
    >>> reg.predict(X[:1,])
    array([154.0842...])

    See also
    --------
    lars_path, LassoLars, LassoLarsCV
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce least-angle (import-module "sklearn.linear_model.least_angle"))

(defn LarsCV 
  "Cross-validated Least Angle Regression model.

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    verbose : boolean or integer, optional
        Sets the verbosity amount

    max_iter : integer, optional
        Maximum number of iterations to perform.

    normalize : boolean, optional, default True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram matrix
        cannot be passed as argument since we will use only subsets of X.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.20
            ``cv`` default value if None will change from 3-fold to 5-fold
            in v0.22.

    max_n_alphas : integer, optional
        The maximum number of points on the path used to compute the
        residuals in the cross-validation

    n_jobs : int or None, optional (default=None)
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    eps : float, optional
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    positive : boolean (default=False)
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.

        .. deprecated:: 0.20
            The option is broken and deprecated. It will be removed in v0.22.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the formulation formula)

    intercept_ : float
        independent term in decision function

    coef_path_ : array, shape (n_features, n_alphas)
        the varying values of the coefficients along the path

    alpha_ : float
        the estimated regularization parameter alpha

    alphas_ : array, shape (n_alphas,)
        the different values of alpha along the path

    cv_alphas_ : array, shape (n_cv_alphas,)
        all the values of alpha along the path for the different folds

    mse_path_ : array, shape (n_folds, n_cv_alphas)
        the mean square error on left-out for each fold along the path
        (alpha values given by ``cv_alphas``)

    n_iter_ : array-like or int
        the number of iterations run by Lars with the optimal alpha.

    Examples
    --------
    >>> from sklearn.linear_model import LarsCV
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, noise=4.0, random_state=0)
    >>> reg = LarsCV(cv=5).fit(X, y)
    >>> reg.score(X, y) # doctest: +ELLIPSIS
    0.9996...
    >>> reg.alpha_
    0.0254...
    >>> reg.predict(X[:1,])
    array([154.0842...])

    See also
    --------
    lars_path, LassoLars, LassoLarsCV
    "
  [ & {:keys [fit_intercept verbose max_iter normalize precompute cv max_n_alphas n_jobs eps copy_X positive]
       :or {fit_intercept true verbose false max_iter 500 normalize true precompute "auto" cv "warn" max_n_alphas 1000 eps 2.220446049250313e-16 copy_X true positive false}} ]
  
   (py/call-attr-kw least-angle "LarsCV" [] {:fit_intercept fit_intercept :verbose verbose :max_iter max_iter :normalize normalize :precompute precompute :cv cv :max_n_alphas max_n_alphas :n_jobs n_jobs :eps eps :copy_X copy_X :positive positive }))

(defn fit 
  "Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            returns an instance of self.
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
