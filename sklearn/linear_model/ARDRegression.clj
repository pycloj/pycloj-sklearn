(ns sklearn.linear-model.ARDRegression
  "Bayesian ARD regression.

    Fit the weights of a regression model, using an ARD prior. The weights of
    the regression model are assumed to be in Gaussian distributions.
    Also estimate the parameters lambda (precisions of the distributions of the
    weights) and alpha (precision of the distribution of the noise).
    The estimation is done by an iterative procedures (Evidence Maximization)

    Read more in the :ref:`User Guide <bayesian_regression>`.

    Parameters
    ----------
    n_iter : int, optional
        Maximum number of iterations. Default is 300

    tol : float, optional
        Stop the algorithm if w has converged. Default is 1.e-3.

    alpha_1 : float, optional
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter. Default is 1.e-6.

    alpha_2 : float, optional
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter. Default is 1.e-6.

    lambda_1 : float, optional
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter. Default is 1.e-6.

    lambda_2 : float, optional
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter. Default is 1.e-6.

    compute_score : boolean, optional
        If True, compute the objective function at each step of the model.
        Default is False.

    threshold_lambda : float, optional
        threshold for removing (pruning) weights with high precision from
        the computation. Default is 1.e+4.

    fit_intercept : boolean, optional
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
        Default is True.

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    copy_X : boolean, optional, default True.
        If True, X will be copied; else, it may be overwritten.

    verbose : boolean, optional, default False
        Verbose mode when fitting the model.

    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of distribution)

    alpha_ : float
       estimated precision of the noise.

    lambda_ : array, shape = (n_features)
       estimated precisions of the weights.

    sigma_ : array, shape = (n_features, n_features)
        estimated variance-covariance matrix of the weights

    scores_ : float
        if computed, value of the objective function (to be maximized)

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.ARDRegression()
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    ... # doctest: +NORMALIZE_WHITESPACE
    ARDRegression(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
            copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,
            n_iter=300, normalize=False, threshold_lambda=10000.0, tol=0.001,
            verbose=False)
    >>> clf.predict([[1, 1]])
    array([1.])

    Notes
    -----
    For an example, see :ref:`examples/linear_model/plot_ard.py
    <sphx_glr_auto_examples_linear_model_plot_ard.py>`.

    References
    ----------
    D. J. C. MacKay, Bayesian nonlinear modeling for the prediction
    competition, ASHRAE Transactions, 1994.

    R. Salakhutdinov, Lecture notes on Statistical Machine Learning,
    http://www.utstat.toronto.edu/~rsalakhu/sta4273/notes/Lecture2.pdf#page=15
    Their beta is our ``self.alpha_``
    Their alpha is our ``self.lambda_``
    ARD is a little different than the slide: only dimensions/features for
    which ``self.lambda_ < self.threshold_lambda`` are kept and the rest are
    discarded.
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

(defn ARDRegression 
  "Bayesian ARD regression.

    Fit the weights of a regression model, using an ARD prior. The weights of
    the regression model are assumed to be in Gaussian distributions.
    Also estimate the parameters lambda (precisions of the distributions of the
    weights) and alpha (precision of the distribution of the noise).
    The estimation is done by an iterative procedures (Evidence Maximization)

    Read more in the :ref:`User Guide <bayesian_regression>`.

    Parameters
    ----------
    n_iter : int, optional
        Maximum number of iterations. Default is 300

    tol : float, optional
        Stop the algorithm if w has converged. Default is 1.e-3.

    alpha_1 : float, optional
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter. Default is 1.e-6.

    alpha_2 : float, optional
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter. Default is 1.e-6.

    lambda_1 : float, optional
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter. Default is 1.e-6.

    lambda_2 : float, optional
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter. Default is 1.e-6.

    compute_score : boolean, optional
        If True, compute the objective function at each step of the model.
        Default is False.

    threshold_lambda : float, optional
        threshold for removing (pruning) weights with high precision from
        the computation. Default is 1.e+4.

    fit_intercept : boolean, optional
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
        Default is True.

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    copy_X : boolean, optional, default True.
        If True, X will be copied; else, it may be overwritten.

    verbose : boolean, optional, default False
        Verbose mode when fitting the model.

    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of distribution)

    alpha_ : float
       estimated precision of the noise.

    lambda_ : array, shape = (n_features)
       estimated precisions of the weights.

    sigma_ : array, shape = (n_features, n_features)
        estimated variance-covariance matrix of the weights

    scores_ : float
        if computed, value of the objective function (to be maximized)

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.ARDRegression()
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    ... # doctest: +NORMALIZE_WHITESPACE
    ARDRegression(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
            copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,
            n_iter=300, normalize=False, threshold_lambda=10000.0, tol=0.001,
            verbose=False)
    >>> clf.predict([[1, 1]])
    array([1.])

    Notes
    -----
    For an example, see :ref:`examples/linear_model/plot_ard.py
    <sphx_glr_auto_examples_linear_model_plot_ard.py>`.

    References
    ----------
    D. J. C. MacKay, Bayesian nonlinear modeling for the prediction
    competition, ASHRAE Transactions, 1994.

    R. Salakhutdinov, Lecture notes on Statistical Machine Learning,
    http://www.utstat.toronto.edu/~rsalakhu/sta4273/notes/Lecture2.pdf#page=15
    Their beta is our ``self.alpha_``
    Their alpha is our ``self.lambda_``
    ARD is a little different than the slide: only dimensions/features for
    which ``self.lambda_ < self.threshold_lambda`` are kept and the rest are
    discarded.
    "
  [ & {:keys [n_iter tol alpha_1 alpha_2 lambda_1 lambda_2 compute_score threshold_lambda fit_intercept normalize copy_X verbose]
       :or {n_iter 300 tol 0.001 alpha_1 1e-06 alpha_2 1e-06 lambda_1 1e-06 lambda_2 1e-06 compute_score false threshold_lambda 10000.0 fit_intercept true normalize false copy_X true verbose false}} ]
  
   (py/call-attr-kw linear-model "ARDRegression" [] {:n_iter n_iter :tol tol :alpha_1 alpha_1 :alpha_2 alpha_2 :lambda_1 lambda_1 :lambda_2 lambda_2 :compute_score compute_score :threshold_lambda threshold_lambda :fit_intercept fit_intercept :normalize normalize :copy_X copy_X :verbose verbose }))

(defn fit 
  "Fit the ARDRegression model according to the given training data
        and parameters.

        Iterative procedure to maximize the evidence

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array, shape = [n_samples]
            Target values (integers). Will be cast to X's dtype if necessary

        Returns
        -------
        self : returns an instance of self.
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
  "Predict using the linear model.

        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        return_std : boolean, optional
            Whether to return the standard deviation of posterior prediction.

        Returns
        -------
        y_mean : array, shape = (n_samples,)
            Mean of predictive distribution of query points.

        y_std : array, shape = (n_samples,)
            Standard deviation of predictive distribution of query points.
        "
  [self X & {:keys [return_std]
                       :or {return_std false}} ]
    (py/call-attr-kw self "predict" [X] {:return_std return_std }))

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
