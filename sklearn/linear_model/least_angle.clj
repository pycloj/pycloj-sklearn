(ns sklearn.linear-model.least-angle
  "
Least Angle Regression algorithm. See the documentation on the
Generalized Linear Model for a complete discussion.
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

(defn as-float-array 
  "Converts an array-like to an array of floats.

    The new dtype will be np.float32 or np.float64, depending on the original
    type. The function can create a copy or modify the argument depending
    on the argument copy.

    Parameters
    ----------
    X : {array-like, sparse matrix}

    copy : bool, optional
        If True, a copy of X will be created. If False, a copy may still be
        returned if X's dtype is not a floating point type.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. The possibilities
        are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan': accept only np.nan values in X. Values cannot be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    Returns
    -------
    XT : {array, sparse matrix}
        An array of type np.float
    "
  [X & {:keys [copy force_all_finite]
                       :or {copy true force_all_finite true}} ]
    (py/call-attr-kw least-angle "as_float_array" [X] {:copy copy :force_all_finite force_all_finite }))

(defn check-X-y 
  "Input validation for standard estimators.

    Checks X and y for consistent length, enforces X to be 2D and y 1D. By
    default, X is checked to be non-empty and containing only finite values.
    Standard input checks are also applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2D and sparse y. If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.

    y : nd-array, list or sparse matrix
        Labels.

    accept_sparse : string, boolean or list of string (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default=\"numeric\")
        Data type of result. If None, the dtype of the input is preserved.
        If \"numeric\", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. This parameter
        does not influence whether y can have np.inf or np.nan values.
        The possibilities are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan': accept only np.nan values in X. Values cannot be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    ensure_2d : boolean (default=True)
        Whether to raise a value error if X is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    multi_output : boolean (default=False)
        Whether to allow 2D y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.

    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    warn_on_dtype : boolean or None, optional (default=None)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

        .. deprecated:: 0.21
            ``warn_on_dtype`` is deprecated in version 0.21 and will be
             removed in 0.23.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    "
  [X y & {:keys [accept_sparse accept_large_sparse dtype order copy force_all_finite ensure_2d allow_nd multi_output ensure_min_samples ensure_min_features y_numeric warn_on_dtype estimator]
                       :or {accept_sparse false accept_large_sparse true dtype "numeric" copy false force_all_finite true ensure_2d true allow_nd false multi_output false ensure_min_samples 1 ensure_min_features 1 y_numeric false}} ]
    (py/call-attr-kw least-angle "check_X_y" [X y] {:accept_sparse accept_sparse :accept_large_sparse accept_large_sparse :dtype dtype :order order :copy copy :force_all_finite force_all_finite :ensure_2d ensure_2d :allow_nd allow_nd :multi_output multi_output :ensure_min_samples ensure_min_samples :ensure_min_features ensure_min_features :y_numeric y_numeric :warn_on_dtype warn_on_dtype :estimator estimator }))

(defn check-cv 
  "Input checker utility for building a cross-validator

    Parameters
    ----------
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if classifier is True and ``y`` is either
        binary or multiclass, :class:`StratifiedKFold` is used. In all other
        cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.20
            ``cv`` default value will change from 3-fold to 5-fold in v0.22.

    y : array-like, optional
        The target variable for supervised learning problems.

    classifier : boolean, optional, default False
        Whether the task is a classification task, in which case
        stratified KFold will be used.

    Returns
    -------
    checked_cv : a cross-validator instance.
        The return value is a cross-validator which generates the train/test
        splits via the ``split`` method.
    "
  [ & {:keys [cv y classifier]
       :or {cv "warn" classifier false}} ]
  
   (py/call-attr-kw least-angle "check_cv" [] {:cv cv :y y :classifier classifier }))

(defn delayed 
  "Decorator used to capture the arguments of a function."
  [ function check_pickle ]
  (py/call-attr least-angle "delayed"  function check_pickle ))

(defn get-lapack-funcs 
  "Return available LAPACK function objects from names.

    Arrays are used to determine the optimal prefix of LAPACK routines.

    Parameters
    ----------
    names : str or sequence of str
        Name(s) of LAPACK functions without type prefix.

    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of LAPACK
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.

    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.

    Returns
    -------
    funcs : list
        List containing the found function(s).

    Notes
    -----
    This routine automatically chooses between Fortran/C
    interfaces. Fortran code is used whenever possible for arrays with
    column major order. In all other cases, C code is preferred.

    In LAPACK, the naming convention is that all functions start with a
    type prefix, which depends on the type of the principal
    matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
    types {float32, float64, complex64, complex128} respectively, and
    are stored in attribute ``typecode`` of the returned functions.

    Examples
    --------
    Suppose we would like to use '?lange' routine which computes the selected
    norm of an array. We pass our array in order to get the correct 'lange'
    flavor.

    >>> import scipy.linalg as LA
    >>> a = np.random.rand(3,2)
    >>> x_lange = LA.get_lapack_funcs('lange', (a,))
    >>> x_lange.typecode
    'd'
    >>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
    >>> x_lange.typecode
    'z'

    Several LAPACK routines work best when its internal WORK array has
    the optimal size (big enough for fast computation and small enough to
    avoid waste of memory). This size is determined also by a dedicated query
    to the function which is often wrapped as a standalone function and
    commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

    >>> import scipy.linalg as LA
    >>> a = np.random.rand(1000,1000)
    >>> b = np.random.rand(1000,1)*1j
    >>> # We pick up zsysv and zsysv_lwork due to b array
    ... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
    >>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
    >>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))

    "
  [names & {:keys [arrays dtype]
                       :or {arrays ()}} ]
    (py/call-attr-kw least-angle "get_lapack_funcs" [names] {:arrays arrays :dtype dtype }))

(defn lars-path 
  "Compute Least Angle Regression or Lasso path using LARS algorithm [1]

    The optimization objective for the case method='lasso' is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    in the case of method='lars', the objective function is only known in
    the form of an implicit equation (see discussion in [1])

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    X : None or array, shape (n_samples, n_features)
        Input data. Note that if X is None then the Gram matrix must be
        specified, i.e., cannot be None or False.

        .. deprecated:: 0.21

           The use of ``X`` is ``None`` in combination with ``Gram`` is not
           ``None`` will be removed in v0.23. Use :func:`lars_path_gram`
           instead.

    y : None or array, shape (n_samples,)
        Input targets.

    Xy : array-like, shape (n_samples,) or (n_samples, n_targets), optional
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    Gram : None, 'auto', array, shape (n_features, n_features), optional
        Precomputed Gram matrix (X' * X), if ``'auto'``, the Gram
        matrix is precomputed from the given X, if there are more samples
        than features.

        .. deprecated:: 0.21

           The use of ``X`` is ``None`` in combination with ``Gram`` is not
           None will be removed in v0.23. Use :func:`lars_path_gram` instead.

    max_iter : integer, optional (default=500)
        Maximum number of iterations to perform, set to infinity for no limit.

    alpha_min : float, optional (default=0)
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.

    method : {'lar', 'lasso'}, optional (default='lar')
        Specifies the returned model. Select ``'lar'`` for Least Angle
        Regression, ``'lasso'`` for the Lasso.

    copy_X : bool, optional (default=True)
        If ``False``, ``X`` is overwritten.

    eps : float, optional (default=``np.finfo(np.float).eps``)
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems.

    copy_Gram : bool, optional (default=True)
        If ``False``, ``Gram`` is overwritten.

    verbose : int (default=0)
        Controls output verbosity.

    return_path : bool, optional (default=True)
        If ``return_path==True`` returns the entire path, else returns only the
        last point of the path.

    return_n_iter : bool, optional (default=False)
        Whether to return the number of iterations.

    positive : boolean (default=False)
        Restrict coefficients to be >= 0.
        This option is only allowed with method 'lasso'. Note that the model
        coefficients will not converge to the ordinary-least-squares solution
        for small values of alpha. Only coefficients up to the smallest alpha
        value (``alphas_[alphas_ > 0.].min()`` when fit_path=True) reached by
        the stepwise Lars-Lasso algorithm are typically in congruence with the
        solution of the coordinate descent lasso_path function.

    Returns
    -------
    alphas : array, shape (n_alphas + 1,)
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller.

    active : array, shape [n_alphas]
        Indices of active variables at the end of the path.

    coefs : array, shape (n_features, n_alphas + 1)
        Coefficients along the path

    n_iter : int
        Number of iterations run. Returned only if return_n_iter is set
        to True.

    See also
    --------
    lars_path_gram
    lasso_path
    lasso_path_gram
    LassoLars
    Lars
    LassoLarsCV
    LarsCV
    sklearn.decomposition.sparse_encode

    References
    ----------
    .. [1] \"Least Angle Regression\", Efron et al.
           http://statweb.stanford.edu/~tibs/ftp/lars.pdf

    .. [2] `Wikipedia entry on the Least-angle regression
           <https://en.wikipedia.org/wiki/Least-angle_regression>`_

    .. [3] `Wikipedia entry on the Lasso
           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_

    "
  [X y Xy Gram & {:keys [max_iter alpha_min method copy_X eps copy_Gram verbose return_path return_n_iter positive]
                       :or {max_iter 500 alpha_min 0 method "lar" copy_X true eps 2.220446049250313e-16 copy_Gram true verbose 0 return_path true return_n_iter false positive false}} ]
    (py/call-attr-kw least-angle "lars_path" [X y Xy Gram] {:max_iter max_iter :alpha_min alpha_min :method method :copy_X copy_X :eps eps :copy_Gram copy_Gram :verbose verbose :return_path return_path :return_n_iter return_n_iter :positive positive }))

(defn lars-path-gram 
  "lars_path in the sufficient stats mode [1]

    The optimization objective for the case method='lasso' is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    in the case of method='lars', the objective function is only known in
    the form of an implicit equation (see discussion in [1])

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    Xy : array-like, shape (n_samples,) or (n_samples, n_targets)
        Xy = np.dot(X.T, y).

    Gram : array, shape (n_features, n_features)
        Gram = np.dot(X.T * X).

    n_samples : integer or float
        Equivalent size of sample.

    max_iter : integer, optional (default=500)
        Maximum number of iterations to perform, set to infinity for no limit.

    alpha_min : float, optional (default=0)
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.

    method : {'lar', 'lasso'}, optional (default='lar')
        Specifies the returned model. Select ``'lar'`` for Least Angle
        Regression, ``'lasso'`` for the Lasso.

    copy_X : bool, optional (default=True)
        If ``False``, ``X`` is overwritten.

    eps : float, optional (default=``np.finfo(np.float).eps``)
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems.

    copy_Gram : bool, optional (default=True)
        If ``False``, ``Gram`` is overwritten.

    verbose : int (default=0)
        Controls output verbosity.

    return_path : bool, optional (default=True)
        If ``return_path==True`` returns the entire path, else returns only the
        last point of the path.

    return_n_iter : bool, optional (default=False)
        Whether to return the number of iterations.

    positive : boolean (default=False)
        Restrict coefficients to be >= 0.
        This option is only allowed with method 'lasso'. Note that the model
        coefficients will not converge to the ordinary-least-squares solution
        for small values of alpha. Only coefficients up to the smallest alpha
        value (``alphas_[alphas_ > 0.].min()`` when fit_path=True) reached by
        the stepwise Lars-Lasso algorithm are typically in congruence with the
        solution of the coordinate descent lasso_path function.

    Returns
    -------
    alphas : array, shape (n_alphas + 1,)
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller.

    active : array, shape [n_alphas]
        Indices of active variables at the end of the path.

    coefs : array, shape (n_features, n_alphas + 1)
        Coefficients along the path

    n_iter : int
        Number of iterations run. Returned only if return_n_iter is set
        to True.

    See also
    --------
    lars_path
    lasso_path
    lasso_path_gram
    LassoLars
    Lars
    LassoLarsCV
    LarsCV
    sklearn.decomposition.sparse_encode

    References
    ----------
    .. [1] \"Least Angle Regression\", Efron et al.
           http://statweb.stanford.edu/~tibs/ftp/lars.pdf

    .. [2] `Wikipedia entry on the Least-angle regression
           <https://en.wikipedia.org/wiki/Least-angle_regression>`_

    .. [3] `Wikipedia entry on the Lasso
           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_

    "
  [Xy Gram n_samples & {:keys [max_iter alpha_min method copy_X eps copy_Gram verbose return_path return_n_iter positive]
                       :or {max_iter 500 alpha_min 0 method "lar" copy_X true eps 2.220446049250313e-16 copy_Gram true verbose 0 return_path true return_n_iter false positive false}} ]
    (py/call-attr-kw least-angle "lars_path_gram" [Xy Gram n_samples] {:max_iter max_iter :alpha_min alpha_min :method method :copy_X copy_X :eps eps :copy_Gram copy_Gram :verbose verbose :return_path return_path :return_n_iter return_n_iter :positive positive }))
