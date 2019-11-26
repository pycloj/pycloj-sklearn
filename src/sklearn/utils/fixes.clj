(ns sklearn.utils.fixes
  "Compatibility fixes for older version of python, numpy and scipy

If you add content to this file, please give the version of the package
at which the fixe is no longer needed.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce fixes (import-module "sklearn.utils.fixes"))

(defn comb 
  "The number of combinations of N things taken k at a time.

    This is often expressed as \"N choose k\".

    Parameters
    ----------
    N : int, ndarray
        Number of things.
    k : int, ndarray
        Number of elements taken.
    exact : bool, optional
        If `exact` is False, then floating point precision is used, otherwise
        exact long integer is computed.
    repetition : bool, optional
        If `repetition` is True, then the number of combinations with
        repetition is computed.

    Returns
    -------
    val : int, float, ndarray
        The total number of combinations.

    See Also
    --------
    binom : Binomial coefficient ufunc

    Notes
    -----
    - Array arguments accepted only for exact=False case.
    - If k > N, N < 0, or k < 0, then a 0 is returned.

    Examples
    --------
    >>> from scipy.special import comb
    >>> k = np.array([3, 4])
    >>> n = np.array([10, 10])
    >>> comb(n, k, exact=False)
    array([ 120.,  210.])
    >>> comb(10, 3, exact=True)
    120L
    >>> comb(10, 3, exact=True, repetition=True)
    220L

    "
  [N k & {:keys [exact repetition]
                       :or {exact false repetition false}} ]
    (py/call-attr-kw fixes "comb" [N k] {:exact exact :repetition repetition }))

(defn logsumexp 
  "Compute the log of the sum of exponentials of input elements.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.

        .. versionadded:: 0.11.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.

        .. versionadded:: 0.15.0
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.

        .. versionadded:: 0.12.0
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).

        .. versionadded:: 0.16.0

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    sgn : ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign
        of the result. If False, only one result is returned.

    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2

    Notes
    -----
    NumPy has a logaddexp function which is very similar to `logsumexp`, but
    only handles two arguments. `logaddexp.reduce` is similar to this
    function, but may be less stable.

    Examples
    --------
    >>> from scipy.special import logsumexp
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107

    With weights

    >>> a = np.arange(10)
    >>> b = np.arange(10, 0, -1)
    >>> logsumexp(a, b=b)
    9.9170178533034665
    >>> np.log(np.sum(b*np.exp(a)))
    9.9170178533034647

    Returning a sign flag

    >>> logsumexp([1,2],b=[1,-1],return_sign=True)
    (1.5413248546129181, -1.0)

    Notice that `logsumexp` does not directly support masked arrays. To use it
    on a masked array, convert the mask into zero weights:

    >>> a = np.ma.array([np.log(2), 2, np.log(3)],
    ...                  mask=[False, True, False])
    >>> b = (~a.mask).astype(int)
    >>> logsumexp(a.data, b=b), np.log(5)
    1.6094379124341005, 1.6094379124341005

    "
  [a & {:keys [axis b keepdims return_sign]
                       :or {keepdims false return_sign false}} ]
    (py/call-attr-kw fixes "logsumexp" [a] {:axis axis :b b :keepdims keepdims :return_sign return_sign }))

(defn parallel-helper 
  "Workaround for Python 2 limitations of pickling instance methods

    Parameters
    ----------
    obj
    methodname
    *args
    **kwargs

    "
  [ obj methodname ]
  (py/call-attr fixes "parallel_helper"  obj methodname ))

(defn pinvh 
  "
    Compute the (Moore-Penrose) pseudo-inverse of a Hermitian matrix.

    Copied in from scipy==1.2.2, in order to preserve the default choice of the
    `cond` and `above_cutoff` values which determine which values of the matrix
    inversion lie below threshold and are so set to zero. Changes in scipy 1.3
    resulted in a smaller default threshold and thus slower convergence of
    dependent algorithms in some cases (see Sklearn github issue #14055).

    Calculate a generalized inverse of a Hermitian or real symmetric matrix
    using its eigenvalue decomposition and including all eigenvalues with
    'large' absolute value.

    Parameters
    ----------
    a : (N, N) array_like
        Real symmetric or complex hermetian matrix to be pseudo-inverted
    cond, rcond : float or None
        Cutoff for 'small' eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are considered
        zero.

        If None or -1, suitable machine precision is used.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower or upper
        triangle of a. (Default: lower)
    return_rank : bool, optional
        if True, return the effective rank of the matrix
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    B : (N, N) ndarray
        The pseudo-inverse of matrix `a`.
    rank : int
        The effective rank of the matrix.  Returned if return_rank == True

    Raises
    ------
    LinAlgError
        If eigenvalue does not converge

    Examples
    --------
    >>> from scipy.linalg import pinvh
    >>> a = np.random.randn(9, 6)
    >>> a = np.dot(a, a.T)
    >>> B = pinvh(a)
    >>> np.allclose(a, np.dot(a, np.dot(B, a)))
    True
    >>> np.allclose(B, np.dot(B, np.dot(a, B)))
    True

    "
  [a & {:keys [cond rcond lower return_rank check_finite]
                       :or {lower true return_rank false check_finite true}} ]
    (py/call-attr-kw fixes "pinvh" [a] {:cond cond :rcond rcond :lower lower :return_rank return_rank :check_finite check_finite }))

(defn sparse-lsqr 
  "Find the least-squares solution to a large, sparse, linear system
    of equations.

    The function solves ``Ax = b``  or  ``min ||b - Ax||^2`` or
    ``min ||Ax - b||^2 + d^2 ||x||^2``.

    The matrix A may be square or rectangular (over-determined or
    under-determined), and may have any rank.

    ::

      1. Unsymmetric equations --    solve  A*x = b

      2. Linear least squares  --    solve  A*x = b
                                     in the least-squares sense

      3. Damped least squares  --    solve  (   A    )*x = ( b )
                                            ( damp*I )     ( 0 )
                                     in the least-squares sense

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        Representation of an m-by-n matrix.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` and ``A^T x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : array_like, shape (m,)
        Right-hand side vector ``b``.
    damp : float
        Damping coefficient.
    atol, btol : float, optional
        Stopping tolerances. If both are 1.0e-9 (say), the final
        residual norm should be accurate to about 9 digits.  (The
        final x will usually have fewer correct digits, depending on
        cond(A) and the size of damp.)
    conlim : float, optional
        Another stopping tolerance.  lsqr terminates if an estimate of
        ``cond(A)`` exceeds `conlim`.  For compatible systems ``Ax =
        b``, `conlim` could be as large as 1.0e+12 (say).  For
        least-squares problems, conlim should be less than 1.0e+8.
        Maximum precision can be obtained by setting ``atol = btol =
        conlim = zero``, but the number of iterations may then be
        excessive.
    iter_lim : int, optional
        Explicit limitation on number of iterations (for safety).
    show : bool, optional
        Display an iteration log.
    calc_var : bool, optional
        Whether to estimate diagonals of ``(A'A + damp^2*I)^{-1}``.
    x0 : array_like, shape (n,), optional
        Initial guess of x, if None zeros are used.

        .. versionadded:: 1.0.0

    Returns
    -------
    x : ndarray of float
        The final solution.
    istop : int
        Gives the reason for termination.
        1 means x is an approximate solution to Ax = b.
        2 means x approximately solves the least-squares problem.
    itn : int
        Iteration number upon termination.
    r1norm : float
        ``norm(r)``, where ``r = b - Ax``.
    r2norm : float
        ``sqrt( norm(r)^2  +  damp^2 * norm(x)^2 )``.  Equal to `r1norm` if
        ``damp == 0``.
    anorm : float
        Estimate of Frobenius norm of ``Abar = [[A]; [damp*I]]``.
    acond : float
        Estimate of ``cond(Abar)``.
    arnorm : float
        Estimate of ``norm(A'*r - damp^2*x)``.
    xnorm : float
        ``norm(x)``
    var : ndarray of float
        If ``calc_var`` is True, estimates all diagonals of
        ``(A'A)^{-1}`` (if ``damp == 0``) or more generally ``(A'A +
        damp^2*I)^{-1}``.  This is well defined if A has full column
        rank or ``damp > 0``.  (Not sure what var means if ``rank(A)
        < n`` and ``damp = 0.``)

    Notes
    -----
    LSQR uses an iterative method to approximate the solution.  The
    number of iterations required to reach a certain accuracy depends
    strongly on the scaling of the problem.  Poor scaling of the rows
    or columns of A should therefore be avoided where possible.

    For example, in problem 1 the solution is unaltered by
    row-scaling.  If a row of A is very small or large compared to
    the other rows of A, the corresponding row of ( A  b ) should be
    scaled up or down.

    In problems 1 and 2, the solution x is easily recovered
    following column-scaling.  Unless better information is known,
    the nonzero columns of A should be scaled so that they all have
    the same Euclidean norm (e.g., 1.0).

    In problem 3, there is no freedom to re-scale if damp is
    nonzero.  However, the value of damp should be assigned only
    after attention has been paid to the scaling of A.

    The parameter damp is intended to help regularize
    ill-conditioned systems, by preventing the true solution from
    being very large.  Another aid to regularization is provided by
    the parameter acond, which may be used to terminate iterations
    before the computed solution becomes very large.

    If some initial estimate ``x0`` is known and if ``damp == 0``,
    one could proceed as follows:

      1. Compute a residual vector ``r0 = b - A*x0``.
      2. Use LSQR to solve the system  ``A*dx = r0``.
      3. Add the correction dx to obtain a final solution ``x = x0 + dx``.

    This requires that ``x0`` be available before and after the call
    to LSQR.  To judge the benefits, suppose LSQR takes k1 iterations
    to solve A*x = b and k2 iterations to solve A*dx = r0.
    If x0 is \"good\", norm(r0) will be smaller than norm(b).
    If the same stopping tolerances atol and btol are used for each
    system, k1 and k2 will be similar, but the final solution x0 + dx
    should be more accurate.  The only way to reduce the total work
    is to use a larger stopping tolerance for the second system.
    If some value btol is suitable for A*x = b, the larger value
    btol*norm(b)/norm(r0)  should be suitable for A*dx = r0.

    Preconditioning is another way to reduce the number of iterations.
    If it is possible to solve a related system ``M*x = b``
    efficiently, where M approximates A in some helpful way (e.g. M -
    A has low rank or its elements are small relative to those of A),
    LSQR may converge more rapidly on the system ``A*M(inverse)*z =
    b``, after which x can be recovered by solving M*x = z.

    If A is symmetric, LSQR should not be used!

    Alternatives are the symmetric conjugate-gradient method (cg)
    and/or SYMMLQ.  SYMMLQ is an implementation of symmetric cg that
    applies to any symmetric A and will converge more rapidly than
    LSQR.  If A is positive definite, there are other implementations
    of symmetric cg that require slightly less work per iteration than
    SYMMLQ (but will take the same number of iterations).

    References
    ----------
    .. [1] C. C. Paige and M. A. Saunders (1982a).
           \"LSQR: An algorithm for sparse linear equations and
           sparse least squares\", ACM TOMS 8(1), 43-71.
    .. [2] C. C. Paige and M. A. Saunders (1982b).
           \"Algorithm 583.  LSQR: Sparse linear equations and least
           squares problems\", ACM TOMS 8(2), 195-209.
    .. [3] M. A. Saunders (1995).  \"Solution of sparse rectangular
           systems using LSQR and CRAIG\", BIT 35, 588-604.

    Examples
    --------
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import lsqr
    >>> A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)

    The first example has the trivial solution `[0, 0]`

    >>> b = np.array([0., 0., 0.], dtype=float)
    >>> x, istop, itn, normr = lsqr(A, b)[:4]
    The exact solution is  x = 0
    >>> istop
    0
    >>> x
    array([ 0.,  0.])

    The stopping code `istop=0` returned indicates that a vector of zeros was
    found as a solution. The returned solution `x` indeed contains `[0., 0.]`.
    The next example has a non-trivial solution:

    >>> b = np.array([1., 0., -1.], dtype=float)
    >>> x, istop, itn, r1norm = lsqr(A, b)[:4]
    >>> istop
    1
    >>> x
    array([ 1., -1.])
    >>> itn
    1
    >>> r1norm
    4.440892098500627e-16

    As indicated by `istop=1`, `lsqr` found a solution obeying the tolerance
    limits. The given solution `[1., -1.]` obviously solves the equation. The
    remaining return values include information about the number of iterations
    (`itn=1`) and the remaining difference of left and right side of the solved
    equation.
    The final example demonstrates the behavior in the case where there is no
    solution for the equation:

    >>> b = np.array([1., 0.01, -1.], dtype=float)
    >>> x, istop, itn, r1norm = lsqr(A, b)[:4]
    >>> istop
    2
    >>> x
    array([ 1.00333333, -0.99666667])
    >>> A.dot(x)-b
    array([ 0.00333333, -0.00333333,  0.00333333])
    >>> r1norm
    0.005773502691896255

    `istop` indicates that the system is inconsistent and thus `x` is rather an
    approximate solution to the corresponding least-squares problem. `r1norm`
    contains the norm of the minimal residual that was found.
    "
  [A b & {:keys [damp atol btol conlim iter_lim show calc_var x0]
                       :or {damp 0.0 atol 1e-08 btol 1e-08 conlim 100000000.0 show false calc_var false}} ]
    (py/call-attr-kw fixes "sparse_lsqr" [A b] {:damp damp :atol atol :btol btol :conlim conlim :iter_lim iter_lim :show show :calc_var calc_var :x0 x0 }))
