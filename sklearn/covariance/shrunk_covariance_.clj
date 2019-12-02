(ns sklearn.covariance.shrunk-covariance-
  "
Covariance estimators using shrinkage.

Shrinkage corresponds to regularising `cov` using a convex combination:
shrunk_cov = (1-shrinkage)*cov + shrinkage*structured_estimate.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce shrunk-covariance- (import-module "sklearn.covariance.shrunk_covariance_"))

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
    (py/call-attr-kw shrunk-covariance- "check_array" [array] {:accept_sparse accept_sparse :accept_large_sparse accept_large_sparse :dtype dtype :order order :copy copy :force_all_finite force_all_finite :ensure_2d ensure_2d :allow_nd allow_nd :ensure_min_samples ensure_min_samples :ensure_min_features ensure_min_features :warn_on_dtype warn_on_dtype :estimator estimator }))

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
    (py/call-attr-kw shrunk-covariance- "empirical_covariance" [X] {:assume_centered assume_centered }))

(defn ledoit-wolf 
  "Estimates the shrunk Ledoit-Wolf covariance matrix.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data from which to compute the covariance estimate

    assume_centered : boolean, default=False
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.

    block_size : int, default=1000
        Size of the blocks into which the covariance matrix will be split.
        This is purely a memory optimization and does not affect results.

    Returns
    -------
    shrunk_cov : array-like, shape (n_features, n_features)
        Shrunk covariance.

    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularized (shrunk) covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features

    "
  [X & {:keys [assume_centered block_size]
                       :or {assume_centered false block_size 1000}} ]
    (py/call-attr-kw shrunk-covariance- "ledoit_wolf" [X] {:assume_centered assume_centered :block_size block_size }))

(defn ledoit-wolf-shrinkage 
  "Estimates the shrunk Ledoit-Wolf covariance matrix.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data from which to compute the Ledoit-Wolf shrunk covariance shrinkage.

    assume_centered : bool
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.

    block_size : int
        Size of the blocks into which the covariance matrix will be split.

    Returns
    -------
    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularized (shrunk) covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features

    "
  [X & {:keys [assume_centered block_size]
                       :or {assume_centered false block_size 1000}} ]
    (py/call-attr-kw shrunk-covariance- "ledoit_wolf_shrinkage" [X] {:assume_centered assume_centered :block_size block_size }))

(defn oas 
  "Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    assume_centered : boolean
      If True, data will not be centered before computation.
      Useful to work with data whose mean is significantly equal to
      zero but is not exactly zero.
      If False, data will be centered before computation.

    Returns
    -------
    shrunk_cov : array-like, shape (n_features, n_features)
        Shrunk covariance.

    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularised (shrunk) covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features

    The formula we used to implement the OAS is slightly modified compared
    to the one given in the article. See :class:`OAS` for more details.

    "
  [X & {:keys [assume_centered]
                       :or {assume_centered false}} ]
    (py/call-attr-kw shrunk-covariance- "oas" [X] {:assume_centered assume_centered }))

(defn shrunk-covariance 
  "Calculates a covariance matrix shrunk on the diagonal

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    emp_cov : array-like, shape (n_features, n_features)
        Covariance matrix to be shrunk

    shrinkage : float, 0 <= shrinkage <= 1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Returns
    -------
    shrunk_cov : array-like
        Shrunk covariance.

    Notes
    -----
    The regularized (shrunk) covariance is given by:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features

    "
  [emp_cov & {:keys [shrinkage]
                       :or {shrinkage 0.1}} ]
    (py/call-attr-kw shrunk-covariance- "shrunk_covariance" [emp_cov] {:shrinkage shrinkage }))
