(ns sklearn.covariance
  "
The :mod:`sklearn.covariance` module includes methods and algorithms to
robustly estimate the covariance of features given a set of points. The
precision matrix defined as the inverse of the covariance is also estimated.
Covariance estimation is closely related to the theory of Gaussian Graphical
Models.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce covariance (import-module "sklearn.covariance"))

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
    (py/call-attr-kw covariance "empirical_covariance" [X] {:assume_centered assume_centered }))

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
    (py/call-attr-kw covariance "fast_mcd" [X support_fraction] {:cov_computation_method cov_computation_method :random_state random_state }))

(defn graph-lasso 
  "DEPRECATED: The 'graph_lasso' was renamed to 'graphical_lasso' in version 0.20 and will be removed in 0.22.

l1-penalized covariance estimator

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    Parameters
    ----------
    emp_cov : 2D ndarray, shape (n_features, n_features)
        Empirical covariance from which to compute the covariance estimate.

    alpha : positive float
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.

    cov_init : 2D array (n_features, n_features), optional
        The initial guess for the covariance.

    mode : {'cd', 'lars'}
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    tol : positive float, optional
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped.

    enet_tol : positive float, optional
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'.

    max_iter : integer, optional
        The maximum number of iterations.

    verbose : boolean, optional
        If verbose is True, the objective function and dual gap are
        printed at each iteration.

    return_costs : boolean, optional
        If return_costs is True, the objective function and dual gap
        at each iteration are returned.

    eps : float, optional
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems.

    return_n_iter : bool, optional
        Whether or not to return the number of iterations.

    Returns
    -------
    covariance : 2D ndarray, shape (n_features, n_features)
        The estimated covariance matrix.

    precision : 2D ndarray, shape (n_features, n_features)
        The estimated (sparse) precision matrix.

    costs : list of (objective, dual_gap) pairs
        The list of values of the objective function and the dual gap at
        each iteration. Returned only if return_costs is True.

    n_iter : int
        Number of iterations. Returned only if `return_n_iter` is set to True.

    See Also
    --------
    GraphLasso, GraphLassoCV

    Notes
    -----
    The algorithm employed to solve this problem is the GLasso algorithm,
    from the Friedman 2008 Biostatistics paper. It is the same algorithm
    as in the R `glasso` package.

    One possible difference with the `glasso` R package is that the
    diagonal coefficients are not penalized.

    "
  [emp_cov alpha cov_init & {:keys [mode tol enet_tol max_iter verbose return_costs eps return_n_iter]
                       :or {mode "cd" tol 0.0001 enet_tol 0.0001 max_iter 100 verbose false return_costs false eps 2.220446049250313e-16 return_n_iter false}} ]
    (py/call-attr-kw covariance "graph_lasso" [emp_cov alpha cov_init] {:mode mode :tol tol :enet_tol enet_tol :max_iter max_iter :verbose verbose :return_costs return_costs :eps eps :return_n_iter return_n_iter }))

(defn graphical-lasso 
  "l1-penalized covariance estimator

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    Parameters
    ----------
    emp_cov : 2D ndarray, shape (n_features, n_features)
        Empirical covariance from which to compute the covariance estimate.

    alpha : positive float
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.

    cov_init : 2D array (n_features, n_features), optional
        The initial guess for the covariance.

    mode : {'cd', 'lars'}
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    tol : positive float, optional
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped.

    enet_tol : positive float, optional
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'.

    max_iter : integer, optional
        The maximum number of iterations.

    verbose : boolean, optional
        If verbose is True, the objective function and dual gap are
        printed at each iteration.

    return_costs : boolean, optional
        If return_costs is True, the objective function and dual gap
        at each iteration are returned.

    eps : float, optional
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems.

    return_n_iter : bool, optional
        Whether or not to return the number of iterations.

    Returns
    -------
    covariance : 2D ndarray, shape (n_features, n_features)
        The estimated covariance matrix.

    precision : 2D ndarray, shape (n_features, n_features)
        The estimated (sparse) precision matrix.

    costs : list of (objective, dual_gap) pairs
        The list of values of the objective function and the dual gap at
        each iteration. Returned only if return_costs is True.

    n_iter : int
        Number of iterations. Returned only if `return_n_iter` is set to True.

    See Also
    --------
    GraphicalLasso, GraphicalLassoCV

    Notes
    -----
    The algorithm employed to solve this problem is the GLasso algorithm,
    from the Friedman 2008 Biostatistics paper. It is the same algorithm
    as in the R `glasso` package.

    One possible difference with the `glasso` R package is that the
    diagonal coefficients are not penalized.

    "
  [emp_cov alpha cov_init & {:keys [mode tol enet_tol max_iter verbose return_costs eps return_n_iter]
                       :or {mode "cd" tol 0.0001 enet_tol 0.0001 max_iter 100 verbose false return_costs false eps 2.220446049250313e-16 return_n_iter false}} ]
    (py/call-attr-kw covariance "graphical_lasso" [emp_cov alpha cov_init] {:mode mode :tol tol :enet_tol enet_tol :max_iter max_iter :verbose verbose :return_costs return_costs :eps eps :return_n_iter return_n_iter }))

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
    (py/call-attr-kw covariance "ledoit_wolf" [X] {:assume_centered assume_centered :block_size block_size }))

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
    (py/call-attr-kw covariance "ledoit_wolf_shrinkage" [X] {:assume_centered assume_centered :block_size block_size }))

(defn log-likelihood 
  "Computes the sample mean of the log_likelihood under a covariance model

    computes the empirical expected log-likelihood (accounting for the
    normalization terms and scaling), allowing for universal comparison (beyond
    this software package)

    Parameters
    ----------
    emp_cov : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance

    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the covariance model to be tested

    Returns
    -------
    sample mean of the log-likelihood
    "
  [ emp_cov precision ]
  (py/call-attr covariance "log_likelihood"  emp_cov precision ))

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
    (py/call-attr-kw covariance "oas" [X] {:assume_centered assume_centered }))

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
    (py/call-attr-kw covariance "shrunk_covariance" [emp_cov] {:shrinkage shrinkage }))
