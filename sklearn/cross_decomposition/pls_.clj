(ns sklearn.cross-decomposition.pls-
  "
The :mod:`sklearn.pls` module implements Partial Least Squares (PLS).
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce pls- (import-module "sklearn.cross_decomposition.pls_"))

(defn abstractmethod 
  "A decorator indicating abstract methods.

    Requires that the metaclass is ABCMeta or derived from it.  A
    class that has a metaclass derived from ABCMeta cannot be
    instantiated unless all of its abstract methods are overridden.
    The abstract methods can be called using any of the normal
    'super' call mechanisms.

    Usage:

        class C(metaclass=ABCMeta):
            @abstractmethod
            def my_abstract_method(self, ...):
                ...
    "
  [ funcobj ]
  (py/call-attr pls- "abstractmethod"  funcobj ))

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
    (py/call-attr-kw pls- "check_array" [array] {:accept_sparse accept_sparse :accept_large_sparse accept_large_sparse :dtype dtype :order order :copy copy :force_all_finite force_all_finite :ensure_2d ensure_2d :allow_nd allow_nd :ensure_min_samples ensure_min_samples :ensure_min_features ensure_min_features :warn_on_dtype warn_on_dtype :estimator estimator }))

(defn check-consistent-length 
  "Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    "
  [  ]
  (py/call-attr pls- "check_consistent_length"  ))

(defn check-is-fitted 
  "Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    \"all_or_any\" of the passed attributes and raises a NotFittedError with the
    given message.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg.:
            ``[\"coef_\", \"estimator_\", ...], \"coef_\"``

    msg : string
        The default error message is, \"This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method.\"

        For custom messages if \"%(name)s\" is present in the message string,
        it is substituted for the estimator name.

        Eg. : \"Estimator, %(name)s, must be fitted before sparsifying\".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        If the attributes are not found.
    "
  [estimator attributes msg & {:keys [all_or_any]
                       :or {all_or_any <built-in function all>}} ]
    (py/call-attr-kw pls- "check_is_fitted" [estimator attributes msg] {:all_or_any all_or_any }))

(defn pinv2 
  "
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate a generalized inverse of a matrix using its
    singular-value decomposition and including all 'large' singular
    values.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be pseudo-inverted.
    cond, rcond : float or None
        Cutoff for 'small' singular values; singular values smaller than this
        value are considered as zero. If both are omitted, the default value
        ``max(M,N)*largest_singular_value*eps`` is used where ``eps`` is the
        machine precision value of the datatype of ``a``.

        .. versionchanged:: 1.3.0
            Previously the default cutoff value was just ``eps*f`` where ``f``
            was ``1e3`` for single precision and ``1e6`` for double precision.

    return_rank : bool, optional
        If True, return the effective rank of the matrix.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    B : (N, M) ndarray
        The pseudo-inverse of matrix `a`.
    rank : int
        The effective rank of the matrix.  Returned if `return_rank` is True.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    Examples
    --------
    >>> from scipy import linalg
    >>> a = np.random.randn(9, 6)
    >>> B = linalg.pinv2(a)
    >>> np.allclose(a, np.dot(a, np.dot(B, a)))
    True
    >>> np.allclose(B, np.dot(B, np.dot(a, B)))
    True

    "
  [a cond rcond & {:keys [return_rank check_finite]
                       :or {return_rank false check_finite true}} ]
    (py/call-attr-kw pls- "pinv2" [a cond rcond] {:return_rank return_rank :check_finite check_finite }))

(defn svd 
  "
    Singular Value Decomposition.

    Factorizes the matrix `a` into two unitary matrices ``U`` and ``Vh``, and
    a 1-D array ``s`` of singular values (real, non-negative) such that
    ``a == U @ S @ Vh``, where ``S`` is a suitably shaped matrix of zeros with
    main diagonal ``s``.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to decompose.
    full_matrices : bool, optional
        If True (default), `U` and `Vh` are of shape ``(M, M)``, ``(N, N)``.
        If False, the shapes are ``(M, K)`` and ``(K, N)``, where
        ``K = min(M, N)``.
    compute_uv : bool, optional
        Whether to compute also ``U`` and ``Vh`` in addition to ``s``.
        Default is True.
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.
        Default is False.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    lapack_driver : {'gesdd', 'gesvd'}, optional
        Whether to use the more efficient divide-and-conquer approach
        (``'gesdd'``) or general rectangular approach (``'gesvd'``)
        to compute the SVD. MATLAB and Octave use the ``'gesvd'`` approach.
        Default is ``'gesdd'``.

        .. versionadded:: 0.18

    Returns
    -------
    U : ndarray
        Unitary matrix having left singular vectors as columns.
        Of shape ``(M, M)`` or ``(M, K)``, depending on `full_matrices`.
    s : ndarray
        The singular values, sorted in non-increasing order.
        Of shape (K,), with ``K = min(M, N)``.
    Vh : ndarray
        Unitary matrix having right singular vectors as rows.
        Of shape ``(N, N)`` or ``(K, N)`` depending on `full_matrices`.

    For ``compute_uv=False``, only ``s`` is returned.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    See also
    --------
    svdvals : Compute singular values of a matrix.
    diagsvd : Construct the Sigma matrix, given the vector s.

    Examples
    --------
    >>> from scipy import linalg
    >>> m, n = 9, 6
    >>> a = np.random.randn(m, n) + 1.j*np.random.randn(m, n)
    >>> U, s, Vh = linalg.svd(a)
    >>> U.shape,  s.shape, Vh.shape
    ((9, 9), (6,), (6, 6))

    Reconstruct the original matrix from the decomposition:

    >>> sigma = np.zeros((m, n))
    >>> for i in range(min(m, n)):
    ...     sigma[i, i] = s[i]
    >>> a1 = np.dot(U, np.dot(sigma, Vh))
    >>> np.allclose(a, a1)
    True

    Alternatively, use ``full_matrices=False`` (notice that the shape of
    ``U`` is then ``(m, n)`` instead of ``(m, m)``):

    >>> U, s, Vh = linalg.svd(a, full_matrices=False)
    >>> U.shape, s.shape, Vh.shape
    ((9, 6), (6,), (6, 6))
    >>> S = np.diag(s)
    >>> np.allclose(a, np.dot(U, np.dot(S, Vh)))
    True

    >>> s2 = linalg.svd(a, compute_uv=False)
    >>> np.allclose(s, s2)
    True

    "
  [a & {:keys [full_matrices compute_uv overwrite_a check_finite lapack_driver]
                       :or {full_matrices true compute_uv true overwrite_a false check_finite true lapack_driver "gesdd"}} ]
    (py/call-attr-kw pls- "svd" [a] {:full_matrices full_matrices :compute_uv compute_uv :overwrite_a overwrite_a :check_finite check_finite :lapack_driver lapack_driver }))

(defn svd-flip 
  "Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u : ndarray
        u and v are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
        so one can compute `np.dot(u * s, v)`.

    v : ndarray
        u and v are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
        so one can compute `np.dot(u * s, v)`.

    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.


    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.

    "
  [u v & {:keys [u_based_decision]
                       :or {u_based_decision true}} ]
    (py/call-attr-kw pls- "svd_flip" [u v] {:u_based_decision u_based_decision }))

(defn svds 
  "Compute the largest k singular values/vectors for a sparse matrix.

    Parameters
    ----------
    A : {sparse matrix, LinearOperator}
        Array to compute the SVD on, of shape (M, N)
    k : int, optional
        Number of singular values and vectors to compute.
        Must be 1 <= k < min(A.shape).
    ncv : int, optional
        The number of Lanczos vectors generated
        ncv must be greater than k+1 and smaller than n;
        it is recommended that ncv > 2*k
        Default: ``min(n, max(2*k + 1, 20))``
    tol : float, optional
        Tolerance for singular values. Zero (default) means machine precision.
    which : str, ['LM' | 'SM'], optional
        Which `k` singular values to find:

            - 'LM' : largest singular values
            - 'SM' : smallest singular values

        .. versionadded:: 0.12.0
    v0 : ndarray, optional
        Starting vector for iteration, of length min(A.shape). Should be an
        (approximate) left singular vector if N > M and a right singular
        vector otherwise.
        Default: random

        .. versionadded:: 0.12.0
    maxiter : int, optional
        Maximum number of iterations.

        .. versionadded:: 0.12.0
    return_singular_vectors : bool or str, optional
        - True: return singular vectors (True) in addition to singular values.

        .. versionadded:: 0.12.0

        - \"u\": only return the u matrix, without computing vh (if N > M).
        - \"vh\": only return the vh matrix, without computing u (if N <= M).

        .. versionadded:: 0.16.0

    Returns
    -------
    u : ndarray, shape=(M, k)
        Unitary matrix having left singular vectors as columns.
        If `return_singular_vectors` is \"vh\", this variable is not computed,
        and None is returned instead.
    s : ndarray, shape=(k,)
        The singular values.
    vt : ndarray, shape=(k, N)
        Unitary matrix having right singular vectors as rows.
        If `return_singular_vectors` is \"u\", this variable is not computed,
        and None is returned instead.


    Notes
    -----
    This is a naive implementation using ARPACK as an eigensolver
    on A.H * A or A * A.H, depending on which one is more efficient.

    Examples
    --------
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import svds, eigs
    >>> A = csc_matrix([[1, 0, 0], [5, 0, 2], [0, -1, 0], [0, 0, 3]], dtype=float)
    >>> u, s, vt = svds(A, k=2)
    >>> s
    array([ 2.75193379,  5.6059665 ])
    >>> np.sqrt(eigs(A.dot(A.T), k=2)[0]).real
    array([ 5.6059665 ,  2.75193379])
    "
  [A & {:keys [k ncv tol which v0 maxiter return_singular_vectors]
                       :or {k 6 tol 0 which "LM" return_singular_vectors true}} ]
    (py/call-attr-kw pls- "svds" [A] {:k k :ncv ncv :tol tol :which which :v0 v0 :maxiter maxiter :return_singular_vectors return_singular_vectors }))
