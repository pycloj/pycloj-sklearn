(ns sklearn.linear-model.Perceptron
  "Perceptron

    Read more in the :ref:`User Guide <perceptron>`.

    Parameters
    ----------

    penalty : None, 'l2' or 'l1' or 'elasticnet'
        The penalty (aka regularization term) to be used. Defaults to None.

    alpha : float
        Constant that multiplies the regularization term if regularization is
        used. Defaults to 0.0001

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

    shuffle : bool, optional, default True
        Whether or not the training data should be shuffled after each epoch.

    verbose : integer, optional
        The verbosity level

    eta0 : double
        Constant by which the updates are multiplied. Defaults to 1.

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation.
        score is not improving. If set to True, it will automatically set aside
        a stratified fraction of training data as validation and terminate
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

    class_weight : dict, {class_label: weight} or \"balanced\" or None, optional
        Preset for the class_weight fit parameter.

        Weights associated with classes. If not given, all classes
        are supposed to have weight one.

        The \"balanced\" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution. See
        :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    coef_ : array, shape = [1, n_features] if n_classes == 2 else [n_classes,            n_features]
        Weights assigned to the features.

    intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.
        For multiclass fits, it is the maximum over every binary fit.

    Notes
    -----

    ``Perceptron`` is a classification algorithm which shares the same
    underlying implementation with ``SGDClassifier``. In fact,
    ``Perceptron()`` is equivalent to `SGDClassifier(loss=\"perceptron\",
    eta0=1, learning_rate=\"constant\", penalty=None)`.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.linear_model import Perceptron
    >>> X, y = load_digits(return_X_y=True)
    >>> clf = Perceptron(tol=1e-3, random_state=0)
    >>> clf.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
    Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
          fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
          penalty=None, random_state=0, shuffle=True, tol=0.001,
          validation_fraction=0.1, verbose=0, warm_start=False)
    >>> clf.score(X, y) # doctest: +ELLIPSIS
    0.939...

    See also
    --------

    SGDClassifier

    References
    ----------

    https://en.wikipedia.org/wiki/Perceptron and references therein.
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

(defn Perceptron 
  "Perceptron

    Read more in the :ref:`User Guide <perceptron>`.

    Parameters
    ----------

    penalty : None, 'l2' or 'l1' or 'elasticnet'
        The penalty (aka regularization term) to be used. Defaults to None.

    alpha : float
        Constant that multiplies the regularization term if regularization is
        used. Defaults to 0.0001

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

    shuffle : bool, optional, default True
        Whether or not the training data should be shuffled after each epoch.

    verbose : integer, optional
        The verbosity level

    eta0 : double
        Constant by which the updates are multiplied. Defaults to 1.

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation.
        score is not improving. If set to True, it will automatically set aside
        a stratified fraction of training data as validation and terminate
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

    class_weight : dict, {class_label: weight} or \"balanced\" or None, optional
        Preset for the class_weight fit parameter.

        Weights associated with classes. If not given, all classes
        are supposed to have weight one.

        The \"balanced\" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution. See
        :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    coef_ : array, shape = [1, n_features] if n_classes == 2 else [n_classes,            n_features]
        Weights assigned to the features.

    intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.
        For multiclass fits, it is the maximum over every binary fit.

    Notes
    -----

    ``Perceptron`` is a classification algorithm which shares the same
    underlying implementation with ``SGDClassifier``. In fact,
    ``Perceptron()`` is equivalent to `SGDClassifier(loss=\"perceptron\",
    eta0=1, learning_rate=\"constant\", penalty=None)`.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.linear_model import Perceptron
    >>> X, y = load_digits(return_X_y=True)
    >>> clf = Perceptron(tol=1e-3, random_state=0)
    >>> clf.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
    Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
          fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
          penalty=None, random_state=0, shuffle=True, tol=0.001,
          validation_fraction=0.1, verbose=0, warm_start=False)
    >>> clf.score(X, y) # doctest: +ELLIPSIS
    0.939...

    See also
    --------

    SGDClassifier

    References
    ----------

    https://en.wikipedia.org/wiki/Perceptron and references therein.
    "
  [penalty & {:keys [alpha fit_intercept max_iter tol shuffle verbose eta0 n_jobs random_state early_stopping validation_fraction n_iter_no_change class_weight warm_start]
                       :or {alpha 0.0001 fit_intercept true max_iter 1000 tol 0.001 shuffle true verbose 0 eta0 1.0 random_state 0 early_stopping false validation_fraction 0.1 n_iter_no_change 5 warm_start false}} ]
    (py/call-attr-kw linear-model "Perceptron" [penalty] {:alpha alpha :fit_intercept fit_intercept :max_iter max_iter :tol tol :shuffle shuffle :verbose verbose :eta0 eta0 :n_jobs n_jobs :random_state random_state :early_stopping early_stopping :validation_fraction validation_fraction :n_iter_no_change n_iter_no_change :class_weight class_weight :warm_start warm_start }))

(defn decision-function 
  "Predict confidence scores for samples.

        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        "
  [ self X ]
  (py/call-attr self "decision_function"  self X ))

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
  "Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data

        y : numpy array, shape (n_samples,)
            Target values

        coef_init : array, shape (n_classes, n_features)
            The initial coefficients to warm-start the optimization.

        intercept_init : array, shape (n_classes,)
            The initial intercept to warm-start the optimization.

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed. These weights will
            be multiplied with class_weight (passed through the
            constructor) if class_weight is specified

        Returns
        -------
        self : returns an instance of self.
        "
  [ self X y coef_init intercept_init sample_weight ]
  (py/call-attr self "fit"  self X y coef_init intercept_init sample_weight ))

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
  "Perform one epoch of stochastic gradient descent on given samples.

        Internally, this method uses ``max_iter = 1``. Therefore, it is not
        guaranteed that a minimum of the cost function is reached after calling
        it once. Matters such as objective convergence and early stopping
        should be handled by the user.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of the training data

        y : numpy array, shape (n_samples,)
            Subset of the target values

        classes : array, shape (n_classes,)
            Classes across all calls to partial_fit.
            Can be obtained by via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : returns an instance of self.
        "
  [ self X y classes sample_weight ]
  (py/call-attr self "partial_fit"  self X y classes sample_weight ))

(defn predict 
  "Predict class labels for samples in X.

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        "
  [ self X ]
  (py/call-attr self "predict"  self X ))

(defn score 
  "Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

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
