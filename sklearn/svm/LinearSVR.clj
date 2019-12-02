(ns sklearn.svm.LinearSVR
  "Linear Support Vector Regression.

    Similar to SVR with parameter kernel='linear', but implemented in terms of
    liblinear rather than libsvm, so it has more flexibility in the choice of
    penalties and loss functions and should scale better to large numbers of
    samples.

    This class supports both dense and sparse input.

    Read more in the :ref:`User Guide <svm_regression>`.

    Parameters
    ----------
    epsilon : float, optional (default=0.0)
        Epsilon parameter in the epsilon-insensitive loss function. Note
        that the value of this parameter depends on the scale of the target
        variable y. If unsure, set ``epsilon=0``.

    tol : float, optional (default=1e-4)
        Tolerance for stopping criteria.

    C : float, optional (default=1.0)
        Penalty parameter C of the error term. The penalty is a squared
        l2 penalty. The bigger this parameter, the less regularization is used.

    loss : string, optional (default='epsilon_insensitive')
        Specifies the loss function. The epsilon-insensitive loss
        (standard SVR) is the L1 loss, while the squared epsilon-insensitive
        loss ('squared_epsilon_insensitive') is the L2 loss.

    fit_intercept : boolean, optional (default=True)
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be already centered).

    intercept_scaling : float, optional (default=1)
        When self.fit_intercept is True, instance vector x becomes
        [x, self.intercept_scaling],
        i.e. a \"synthetic\" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    dual : bool, (default=True)
        Select the algorithm to either solve the dual or primal
        optimization problem. Prefer dual=False when n_samples > n_features.

    verbose : int, (default=0)
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    max_iter : int, (default=1000)
        The maximum number of iterations to be run.

    Attributes
    ----------
    coef_ : array, shape = [n_features] if n_classes == 2 else [n_classes, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is a readonly property derived from `raw_coef_` that
        follows the internal memory layout of liblinear.

    intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    Examples
    --------
    >>> from sklearn.svm import LinearSVR
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=4, random_state=0)
    >>> regr = LinearSVR(random_state=0, tol=1e-5)
    >>> regr.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
    LinearSVR(C=1.0, dual=True, epsilon=0.0, fit_intercept=True,
         intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
         random_state=0, tol=1e-05, verbose=0)
    >>> print(regr.coef_)
    [16.35... 26.91... 42.30... 60.47...]
    >>> print(regr.intercept_)
    [-4.29...]
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-4.29...]

    See also
    --------
    LinearSVC
        Implementation of Support Vector Machine classifier using the
        same library as this class (liblinear).

    SVR
        Implementation of Support Vector Machine regression using libsvm:
        the kernel can be non-linear but its SMO algorithm does not
        scale to large number of samples as LinearSVC does.

    sklearn.linear_model.SGDRegressor
        SGDRegressor can optimize the same cost function as LinearSVR
        by adjusting the penalty and loss parameters. In addition it requires
        less memory, allows incremental (online) learning, and implements
        various loss functions and regularization regimes.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce svm (import-module "sklearn.svm"))

(defn LinearSVR 
  "Linear Support Vector Regression.

    Similar to SVR with parameter kernel='linear', but implemented in terms of
    liblinear rather than libsvm, so it has more flexibility in the choice of
    penalties and loss functions and should scale better to large numbers of
    samples.

    This class supports both dense and sparse input.

    Read more in the :ref:`User Guide <svm_regression>`.

    Parameters
    ----------
    epsilon : float, optional (default=0.0)
        Epsilon parameter in the epsilon-insensitive loss function. Note
        that the value of this parameter depends on the scale of the target
        variable y. If unsure, set ``epsilon=0``.

    tol : float, optional (default=1e-4)
        Tolerance for stopping criteria.

    C : float, optional (default=1.0)
        Penalty parameter C of the error term. The penalty is a squared
        l2 penalty. The bigger this parameter, the less regularization is used.

    loss : string, optional (default='epsilon_insensitive')
        Specifies the loss function. The epsilon-insensitive loss
        (standard SVR) is the L1 loss, while the squared epsilon-insensitive
        loss ('squared_epsilon_insensitive') is the L2 loss.

    fit_intercept : boolean, optional (default=True)
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be already centered).

    intercept_scaling : float, optional (default=1)
        When self.fit_intercept is True, instance vector x becomes
        [x, self.intercept_scaling],
        i.e. a \"synthetic\" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    dual : bool, (default=True)
        Select the algorithm to either solve the dual or primal
        optimization problem. Prefer dual=False when n_samples > n_features.

    verbose : int, (default=0)
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    max_iter : int, (default=1000)
        The maximum number of iterations to be run.

    Attributes
    ----------
    coef_ : array, shape = [n_features] if n_classes == 2 else [n_classes, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is a readonly property derived from `raw_coef_` that
        follows the internal memory layout of liblinear.

    intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    Examples
    --------
    >>> from sklearn.svm import LinearSVR
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=4, random_state=0)
    >>> regr = LinearSVR(random_state=0, tol=1e-5)
    >>> regr.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
    LinearSVR(C=1.0, dual=True, epsilon=0.0, fit_intercept=True,
         intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
         random_state=0, tol=1e-05, verbose=0)
    >>> print(regr.coef_)
    [16.35... 26.91... 42.30... 60.47...]
    >>> print(regr.intercept_)
    [-4.29...]
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-4.29...]

    See also
    --------
    LinearSVC
        Implementation of Support Vector Machine classifier using the
        same library as this class (liblinear).

    SVR
        Implementation of Support Vector Machine regression using libsvm:
        the kernel can be non-linear but its SMO algorithm does not
        scale to large number of samples as LinearSVC does.

    sklearn.linear_model.SGDRegressor
        SGDRegressor can optimize the same cost function as LinearSVR
        by adjusting the penalty and loss parameters. In addition it requires
        less memory, allows incremental (online) learning, and implements
        various loss functions and regularization regimes.
    "
  [ & {:keys [epsilon tol C loss fit_intercept intercept_scaling dual verbose random_state max_iter]
       :or {epsilon 0.0 tol 0.0001 C 1.0 loss "epsilon_insensitive" fit_intercept true intercept_scaling 1.0 dual true verbose 0 max_iter 1000}} ]
  
   (py/call-attr-kw svm "LinearSVR" [] {:epsilon epsilon :tol tol :C C :loss loss :fit_intercept fit_intercept :intercept_scaling intercept_scaling :dual dual :verbose verbose :random_state random_state :max_iter max_iter }))

(defn fit 
  "Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target vector relative to X

        sample_weight : array-like, shape = [n_samples], optional
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.

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
