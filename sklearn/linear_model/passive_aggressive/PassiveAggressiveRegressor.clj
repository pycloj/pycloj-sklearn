(ns sklearn.linear-model.passive-aggressive.PassiveAggressiveRegressor
  "Passive Aggressive Regressor

    Read more in the :ref:`User Guide <passive_aggressive>`.

    Parameters
    ----------

    C : float
        Maximum step size (regularization). Defaults to 1.0.

    fit_intercept : bool
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. Defaults to True.

    max_iter : int, optional (default=1000)
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        `partial_fit`.

        .. versionadded:: 0.19

    tol : float or None, optional (default=1e-3)
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > previous_loss - tol).

        .. versionadded:: 0.19

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation.
        score is not improving. If set to True, it will automatically set aside
        a fraction of training data as validation and terminate
        training when validation score is not improving by at least tol for
        n_iter_no_change consecutive epochs.

        .. versionadded:: 0.20

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before early stopping.

        .. versionadded:: 0.20

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : integer, optional
        The verbosity level

    loss : string, optional
        The loss function to be used:
        epsilon_insensitive: equivalent to PA-I in the reference paper.
        squared_epsilon_insensitive: equivalent to PA-II in the reference
        paper.

    epsilon : float
        If the difference between the current prediction and the correct label
        is below this threshold, the model is not updated.

    random_state : int, RandomState instance or None, optional, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.

    average : bool or int, optional
        When set to True, computes the averaged SGD weights and stores the
        result in the ``coef_`` attribute. If set to an int greater than 1,
        averaging will begin once the total number of samples seen reaches
        average. So average=10 will begin averaging after seeing 10 samples.

        .. versionadded:: 0.19
           parameter *average* to use weights averaging in SGD

    Attributes
    ----------
    coef_ : array, shape = [1, n_features] if n_classes == 2 else [n_classes,            n_features]
        Weights assigned to the features.

    intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

    Examples
    --------
    >>> from sklearn.linear_model import PassiveAggressiveRegressor
    >>> from sklearn.datasets import make_regression

    >>> X, y = make_regression(n_features=4, random_state=0)
    >>> regr = PassiveAggressiveRegressor(max_iter=100, random_state=0,
    ... tol=1e-3)
    >>> regr.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
    PassiveAggressiveRegressor(C=1.0, average=False, early_stopping=False,
                  epsilon=0.1, fit_intercept=True, loss='epsilon_insensitive',
                  max_iter=100, n_iter_no_change=5, random_state=0,
                  shuffle=True, tol=0.001, validation_fraction=0.1,
                  verbose=0, warm_start=False)
    >>> print(regr.coef_)
    [20.48736655 34.18818427 67.59122734 87.94731329]
    >>> print(regr.intercept_)
    [-0.02306214]
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-0.02306214]

    See also
    --------

    SGDRegressor

    References
    ----------
    Online Passive-Aggressive Algorithms
    <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>
    K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR (2006)

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce passive-aggressive (import-module "sklearn.linear_model.passive_aggressive"))

(defn PassiveAggressiveRegressor 
  "Passive Aggressive Regressor

    Read more in the :ref:`User Guide <passive_aggressive>`.

    Parameters
    ----------

    C : float
        Maximum step size (regularization). Defaults to 1.0.

    fit_intercept : bool
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. Defaults to True.

    max_iter : int, optional (default=1000)
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        `partial_fit`.

        .. versionadded:: 0.19

    tol : float or None, optional (default=1e-3)
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > previous_loss - tol).

        .. versionadded:: 0.19

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation.
        score is not improving. If set to True, it will automatically set aside
        a fraction of training data as validation and terminate
        training when validation score is not improving by at least tol for
        n_iter_no_change consecutive epochs.

        .. versionadded:: 0.20

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before early stopping.

        .. versionadded:: 0.20

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : integer, optional
        The verbosity level

    loss : string, optional
        The loss function to be used:
        epsilon_insensitive: equivalent to PA-I in the reference paper.
        squared_epsilon_insensitive: equivalent to PA-II in the reference
        paper.

    epsilon : float
        If the difference between the current prediction and the correct label
        is below this threshold, the model is not updated.

    random_state : int, RandomState instance or None, optional, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.

    average : bool or int, optional
        When set to True, computes the averaged SGD weights and stores the
        result in the ``coef_`` attribute. If set to an int greater than 1,
        averaging will begin once the total number of samples seen reaches
        average. So average=10 will begin averaging after seeing 10 samples.

        .. versionadded:: 0.19
           parameter *average* to use weights averaging in SGD

    Attributes
    ----------
    coef_ : array, shape = [1, n_features] if n_classes == 2 else [n_classes,            n_features]
        Weights assigned to the features.

    intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

    Examples
    --------
    >>> from sklearn.linear_model import PassiveAggressiveRegressor
    >>> from sklearn.datasets import make_regression

    >>> X, y = make_regression(n_features=4, random_state=0)
    >>> regr = PassiveAggressiveRegressor(max_iter=100, random_state=0,
    ... tol=1e-3)
    >>> regr.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
    PassiveAggressiveRegressor(C=1.0, average=False, early_stopping=False,
                  epsilon=0.1, fit_intercept=True, loss='epsilon_insensitive',
                  max_iter=100, n_iter_no_change=5, random_state=0,
                  shuffle=True, tol=0.001, validation_fraction=0.1,
                  verbose=0, warm_start=False)
    >>> print(regr.coef_)
    [20.48736655 34.18818427 67.59122734 87.94731329]
    >>> print(regr.intercept_)
    [-0.02306214]
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-0.02306214]

    See also
    --------

    SGDRegressor

    References
    ----------
    Online Passive-Aggressive Algorithms
    <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>
    K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR (2006)

    "
  [ & {:keys [C fit_intercept max_iter tol early_stopping validation_fraction n_iter_no_change shuffle verbose loss epsilon random_state warm_start average]
       :or {C 1.0 fit_intercept true max_iter 1000 tol 0.001 early_stopping false validation_fraction 0.1 n_iter_no_change 5 shuffle true verbose 0 loss "epsilon_insensitive" epsilon 0.1 warm_start false average false}} ]
  
   (py/call-attr-kw passive-aggressive "PassiveAggressiveRegressor" [] {:C C :fit_intercept fit_intercept :max_iter max_iter :tol tol :early_stopping early_stopping :validation_fraction validation_fraction :n_iter_no_change n_iter_no_change :shuffle shuffle :verbose verbose :loss loss :epsilon epsilon :random_state random_state :warm_start warm_start :average average }))

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
  "Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : numpy array of shape [n_samples]
            Target values

        coef_init : array, shape = [n_features]
            The initial coefficients to warm-start the optimization.

        intercept_init : array, shape = [1]
            The initial intercept to warm-start the optimization.

        Returns
        -------
        self : returns an instance of self.
        "
  [ self X y coef_init intercept_init ]
  (py/call-attr self "fit"  self X y coef_init intercept_init ))

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

(defn partial-fit 
  "Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Subset of training data

        y : numpy array of shape [n_samples]
            Subset of target values

        Returns
        -------
        self : returns an instance of self.
        "
  [ self X y ]
  (py/call-attr self "partial_fit"  self X y ))

(defn predict 
  "Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        array, shape (n_samples,)
           Predicted target values per element in X.
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
