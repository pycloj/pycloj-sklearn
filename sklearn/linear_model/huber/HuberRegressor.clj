(ns sklearn.linear-model.huber.HuberRegressor
  "Linear regression model that is robust to outliers.

    The Huber Regressor optimizes the squared loss for the samples where
    ``|(y - X'w) / sigma| < epsilon`` and the absolute loss for the samples
    where ``|(y - X'w) / sigma| > epsilon``, where w and sigma are parameters
    to be optimized. The parameter sigma makes sure that if y is scaled up
    or down by a certain factor, one does not need to rescale epsilon to
    achieve the same robustness. Note that this does not take into account
    the fact that the different features of X may be of different scales.

    This makes sure that the loss function is not heavily influenced by the
    outliers while not completely ignoring their effect.

    Read more in the :ref:`User Guide <huber_regression>`

    .. versionadded:: 0.18

    Parameters
    ----------
    epsilon : float, greater than 1.0, default 1.35
        The parameter epsilon controls the number of samples that should be
        classified as outliers. The smaller the epsilon, the more robust it is
        to outliers.

    max_iter : int, default 100
        Maximum number of iterations that scipy.optimize.fmin_l_bfgs_b
        should run for.

    alpha : float, default 0.0001
        Regularization parameter.

    warm_start : bool, default False
        This is useful if the stored attributes of a previously used model
        has to be reused. If set to False, then the coefficients will
        be rewritten for every call to fit.
        See :term:`the Glossary <warm_start>`.

    fit_intercept : bool, default True
        Whether or not to fit the intercept. This can be set to False
        if the data is already centered around the origin.

    tol : float, default 1e-5
        The iteration will stop when
        ``max{|proj g_i | i = 1, ..., n}`` <= ``tol``
        where pg_i is the i-th component of the projected gradient.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Features got by optimizing the Huber loss.

    intercept_ : float
        Bias.

    scale_ : float
        The value by which ``|y - X'w - c|`` is scaled down.

    n_iter_ : int
        Number of iterations that fmin_l_bfgs_b has run for.

        .. versionchanged:: 0.20

            In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
            ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.

    outliers_ : array, shape (n_samples,)
        A boolean mask which is set to True where the samples are identified
        as outliers.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import HuberRegressor, LinearRegression
    >>> from sklearn.datasets import make_regression
    >>> rng = np.random.RandomState(0)
    >>> X, y, coef = make_regression(
    ...     n_samples=200, n_features=2, noise=4.0, coef=True, random_state=0)
    >>> X[:4] = rng.uniform(10, 20, (4, 2))
    >>> y[:4] = rng.uniform(10, 20, 4)
    >>> huber = HuberRegressor().fit(X, y)
    >>> huber.score(X, y) # doctest: +ELLIPSIS
    -7.284608623514573
    >>> huber.predict(X[:1,])
    array([806.7200...])
    >>> linear = LinearRegression().fit(X, y)
    >>> print(\"True coefficients:\", coef)
    True coefficients: [20.4923...  34.1698...]
    >>> print(\"Huber coefficients:\", huber.coef_)
    Huber coefficients: [17.7906... 31.0106...]
    >>> print(\"Linear Regression coefficients:\", linear.coef_)
    Linear Regression coefficients: [-1.9221...  7.0226...]

    References
    ----------
    .. [1] Peter J. Huber, Elvezio M. Ronchetti, Robust Statistics
           Concomitant scale estimates, pg 172
    .. [2] Art B. Owen (2006), A robust hybrid of lasso and ridge regression.
           https://statweb.stanford.edu/~owen/reports/hhu.pdf
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce huber (import-module "sklearn.linear_model.huber"))

(defn HuberRegressor 
  "Linear regression model that is robust to outliers.

    The Huber Regressor optimizes the squared loss for the samples where
    ``|(y - X'w) / sigma| < epsilon`` and the absolute loss for the samples
    where ``|(y - X'w) / sigma| > epsilon``, where w and sigma are parameters
    to be optimized. The parameter sigma makes sure that if y is scaled up
    or down by a certain factor, one does not need to rescale epsilon to
    achieve the same robustness. Note that this does not take into account
    the fact that the different features of X may be of different scales.

    This makes sure that the loss function is not heavily influenced by the
    outliers while not completely ignoring their effect.

    Read more in the :ref:`User Guide <huber_regression>`

    .. versionadded:: 0.18

    Parameters
    ----------
    epsilon : float, greater than 1.0, default 1.35
        The parameter epsilon controls the number of samples that should be
        classified as outliers. The smaller the epsilon, the more robust it is
        to outliers.

    max_iter : int, default 100
        Maximum number of iterations that scipy.optimize.fmin_l_bfgs_b
        should run for.

    alpha : float, default 0.0001
        Regularization parameter.

    warm_start : bool, default False
        This is useful if the stored attributes of a previously used model
        has to be reused. If set to False, then the coefficients will
        be rewritten for every call to fit.
        See :term:`the Glossary <warm_start>`.

    fit_intercept : bool, default True
        Whether or not to fit the intercept. This can be set to False
        if the data is already centered around the origin.

    tol : float, default 1e-5
        The iteration will stop when
        ``max{|proj g_i | i = 1, ..., n}`` <= ``tol``
        where pg_i is the i-th component of the projected gradient.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Features got by optimizing the Huber loss.

    intercept_ : float
        Bias.

    scale_ : float
        The value by which ``|y - X'w - c|`` is scaled down.

    n_iter_ : int
        Number of iterations that fmin_l_bfgs_b has run for.

        .. versionchanged:: 0.20

            In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
            ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.

    outliers_ : array, shape (n_samples,)
        A boolean mask which is set to True where the samples are identified
        as outliers.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import HuberRegressor, LinearRegression
    >>> from sklearn.datasets import make_regression
    >>> rng = np.random.RandomState(0)
    >>> X, y, coef = make_regression(
    ...     n_samples=200, n_features=2, noise=4.0, coef=True, random_state=0)
    >>> X[:4] = rng.uniform(10, 20, (4, 2))
    >>> y[:4] = rng.uniform(10, 20, 4)
    >>> huber = HuberRegressor().fit(X, y)
    >>> huber.score(X, y) # doctest: +ELLIPSIS
    -7.284608623514573
    >>> huber.predict(X[:1,])
    array([806.7200...])
    >>> linear = LinearRegression().fit(X, y)
    >>> print(\"True coefficients:\", coef)
    True coefficients: [20.4923...  34.1698...]
    >>> print(\"Huber coefficients:\", huber.coef_)
    Huber coefficients: [17.7906... 31.0106...]
    >>> print(\"Linear Regression coefficients:\", linear.coef_)
    Linear Regression coefficients: [-1.9221...  7.0226...]

    References
    ----------
    .. [1] Peter J. Huber, Elvezio M. Ronchetti, Robust Statistics
           Concomitant scale estimates, pg 172
    .. [2] Art B. Owen (2006), A robust hybrid of lasso and ridge regression.
           https://statweb.stanford.edu/~owen/reports/hhu.pdf
    "
  [ & {:keys [epsilon max_iter alpha warm_start fit_intercept tol]
       :or {epsilon 1.35 max_iter 100 alpha 0.0001 warm_start false fit_intercept true tol 1e-05}} ]
  
   (py/call-attr-kw huber "HuberRegressor" [] {:epsilon epsilon :max_iter max_iter :alpha alpha :warm_start warm_start :fit_intercept fit_intercept :tol tol }))

(defn fit 
  "Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,)
            Weight given to each sample.

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
