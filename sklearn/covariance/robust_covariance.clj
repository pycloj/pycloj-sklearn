(ns sklearn.covariance.robust-covariance
  "
Robust location and covariance estimators.

Here are implemented estimators that are resistant to outliers.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce robust-covariance (import-module "sklearn.covariance.robust_covariance"))

(defn c-step 
  "C_step procedure described in [Rouseeuw1984]_ aiming at computing MCD.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data set in which we look for the n_support observations whose
        scatter matrix has minimum determinant.

    n_support : int, > n_samples / 2
        Number of observations to compute the robust estimates of location
        and covariance from.

    remaining_iterations : int, optional
        Number of iterations to perform.
        According to [Rouseeuw1999]_, two iterations are sufficient to get
        close to the minimum, and we never need more than 30 to reach
        convergence.

    initial_estimates : 2-tuple, optional
        Initial estimates of location and shape from which to run the c_step
        procedure:
        - initial_estimates[0]: an initial location estimate
        - initial_estimates[1]: an initial covariance estimate

    verbose : boolean, optional
        Verbose mode.

    cov_computation_method : callable, default empirical_covariance
        The function which will be used to compute the covariance.
        Must return shape (n_features, n_features)

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    location : array-like, shape (n_features,)
        Robust location estimates.

    covariance : array-like, shape (n_features, n_features)
        Robust covariance estimates.

    support : array-like, shape (n_samples,)
        A mask for the `n_support` observations whose scatter matrix has
        minimum determinant.

    References
    ----------
    .. [Rouseeuw1999] A Fast Algorithm for the Minimum Covariance Determinant
        Estimator, 1999, American Statistical Association and the American
        Society for Quality, TECHNOMETRICS

    "
  [X n_support & {:keys [remaining_iterations initial_estimates verbose cov_computation_method random_state]
                       :or {remaining_iterations 30 verbose false cov_computation_method <function empirical_covariance at 0x1a222677a0>}} ]
    (py/call-attr-kw robust-covariance "c_step" [X n_support] {:remaining_iterations remaining_iterations :initial_estimates initial_estimates :verbose verbose :cov_computation_method cov_computation_method :random_state random_state }))

(defn check-array 
  "Input validation on an array, list, sparse matrix or similar.

    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default=\"numeric\")
        Data type of result. If None, the dtype of the input is preserved.
        If \"numeric\", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.
        - 'allow-nan': accept only np.nan values in array. Values cannot
          be infinite.

        For object dtyped data, only np.nan is checked and not np.inf.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    ensure_2d : boolean (default=True)
        Whether to raise a value error if array is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow array.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

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
    array_converted : object
        The converted and validated array.
    "
  [array & {:keys [accept_sparse accept_large_sparse dtype order copy force_all_finite ensure_2d allow_nd ensure_min_samples ensure_min_features warn_on_dtype estimator]
                       :or {accept_sparse false accept_large_sparse true dtype "numeric" copy false force_all_finite true ensure_2d true allow_nd false ensure_min_samples 1 ensure_min_features 1}} ]
    (py/call-attr-kw robust-covariance "check_array" [array] {:accept_sparse accept_sparse :accept_large_sparse accept_large_sparse :dtype dtype :order order :copy copy :force_all_finite force_all_finite :ensure_2d ensure_2d :allow_nd allow_nd :ensure_min_samples ensure_min_samples :ensure_min_features ensure_min_features :warn_on_dtype warn_on_dtype :estimator estimator }))

(defn check-random-state 
  "Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    "
  [ seed ]
  (py/call-attr robust-covariance "check_random_state"  seed ))

(defn empirical-covariance 
  "Computes the Maximum likelihood covariance estimator


    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data from which to compute the covariance estimate

    assume_centered : boolean
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data will be centered before computation.

    Returns
    -------
    covariance : 2D ndarray, shape (n_features, n_features)
        Empirical covariance (Maximum Likelihood Estimator).

    "
  [X & {:keys [assume_centered]
                       :or {assume_centered false}} ]
    (py/call-attr-kw robust-covariance "empirical_covariance" [X] {:assume_centered assume_centered }))

(defn fast-logdet 
  "Compute log(det(A)) for A symmetric

    Equivalent to : np.log(nl.det(A)) but more robust.
    It returns -Inf if det(A) is non positive or is not defined.

    Parameters
    ----------
    A : array_like
        The matrix
    "
  [ A ]
  (py/call-attr robust-covariance "fast_logdet"  A ))

(defn fast-mcd 
  "Estimates the Minimum Covariance Determinant matrix.

    Read more in the :ref:`User Guide <robust_covariance>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
      The data matrix, with p features and n samples.

    support_fraction : float, 0 < support_fraction < 1
          The proportion of points to be included in the support of the raw
          MCD estimate. Default is None, which implies that the minimum
          value of support_fraction will be used within the algorithm:
          `[n_sample + n_features + 1] / 2`.

    cov_computation_method : callable, default empirical_covariance
        The function which will be used to compute the covariance.
        Must return shape (n_features, n_features)

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Notes
    -----
    The FastMCD algorithm has been introduced by Rousseuw and Van Driessen
    in \"A Fast Algorithm for the Minimum Covariance Determinant Estimator,
    1999, American Statistical Association and the American Society
    for Quality, TECHNOMETRICS\".
    The principle is to compute robust estimates and random subsets before
    pooling them into a larger subsets, and finally into the full data set.
    Depending on the size of the initial sample, we have one, two or three
    such computation levels.

    Note that only raw estimates are returned. If one is interested in
    the correction and reweighting steps described in [RouseeuwVan]_,
    see the MinCovDet object.

    References
    ----------

    .. [RouseeuwVan] A Fast Algorithm for the Minimum Covariance
        Determinant Estimator, 1999, American Statistical Association
        and the American Society for Quality, TECHNOMETRICS

    .. [Butler1993] R. W. Butler, P. L. Davies and M. Jhun,
        Asymptotics For The Minimum Covariance Determinant Estimator,
        The Annals of Statistics, 1993, Vol. 21, No. 3, 1385-1400

    Returns
    -------
    location : array-like, shape (n_features,)
        Robust location of the data.

    covariance : array-like, shape (n_features, n_features)
        Robust covariance of the features.

    support : array-like, type boolean, shape (n_samples,)
        A mask of the observations that have been used to compute
        the robust location and covariance estimates of the data set.

    "
  [X support_fraction & {:keys [cov_computation_method random_state]
                       :or {cov_computation_method <function empirical_covariance at 0x1a222677a0>}} ]
    (py/call-attr-kw robust-covariance "fast_mcd" [X support_fraction] {:cov_computation_method cov_computation_method :random_state random_state }))

(defn select-candidates 
  "Finds the best pure subset of observations to compute MCD from it.

    The purpose of this function is to find the best sets of n_support
    observations with respect to a minimization of their covariance
    matrix determinant. Equivalently, it removes n_samples-n_support
    observations to construct what we call a pure data set (i.e. not
    containing outliers). The list of the observations of the pure
    data set is referred to as the `support`.

    Starting from a random support, the pure data set is found by the
    c_step procedure introduced by Rousseeuw and Van Driessen in
    [RV]_.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data (sub)set in which we look for the n_support purest observations.

    n_support : int, [(n + p + 1)/2] < n_support < n
        The number of samples the pure data set must contain.

    n_trials : int, nb_trials > 0 or 2-tuple
        Number of different initial sets of observations from which to
        run the algorithm.
        Instead of giving a number of trials to perform, one can provide a
        list of initial estimates that will be used to iteratively run
        c_step procedures. In this case:
        - n_trials[0]: array-like, shape (n_trials, n_features)
          is the list of `n_trials` initial location estimates
        - n_trials[1]: array-like, shape (n_trials, n_features, n_features)
          is the list of `n_trials` initial covariances estimates

    select : int, int > 0
        Number of best candidates results to return.

    n_iter : int, nb_iter > 0
        Maximum number of iterations for the c_step procedure.
        (2 is enough to be close to the final solution. \"Never\" exceeds 20).

    verbose : boolean, default False
        Control the output verbosity.

    cov_computation_method : callable, default empirical_covariance
        The function which will be used to compute the covariance.
        Must return shape (n_features, n_features)

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    See Also
    ---------
    c_step

    Returns
    -------
    best_locations : array-like, shape (select, n_features)
        The `select` location estimates computed from the `select` best
        supports found in the data set (`X`).

    best_covariances : array-like, shape (select, n_features, n_features)
        The `select` covariance estimates computed from the `select`
        best supports found in the data set (`X`).

    best_supports : array-like, shape (select, n_samples)
        The `select` best supports found in the data set (`X`).

    References
    ----------
    .. [RV] A Fast Algorithm for the Minimum Covariance Determinant
        Estimator, 1999, American Statistical Association and the American
        Society for Quality, TECHNOMETRICS

    "
  [X n_support n_trials & {:keys [select n_iter verbose cov_computation_method random_state]
                       :or {select 1 n_iter 30 verbose false cov_computation_method <function empirical_covariance at 0x1a222677a0>}} ]
    (py/call-attr-kw robust-covariance "select_candidates" [X n_support n_trials] {:select select :n_iter n_iter :verbose verbose :cov_computation_method cov_computation_method :random_state random_state }))
