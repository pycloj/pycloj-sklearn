(ns sklearn.decomposition
  "
The :mod:`sklearn.decomposition` module includes matrix decomposition
algorithms, including among others PCA, NMF or ICA. Most of the algorithms of
this module can be regarded as dimensionality reduction techniques.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce decomposition (import-module "sklearn.decomposition"))

(defn dict-learning 
  "Solves a dictionary learning matrix factorization problem.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                     (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Data matrix.

    n_components : int,
        Number of dictionary atoms to extract.

    alpha : int,
        Sparsity controlling parameter.

    max_iter : int,
        Maximum number of iterations to perform.

    tol : float,
        Tolerance for the stopping condition.

    method : {'lars', 'cd'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    n_jobs : int or None, optional (default=None)
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    dict_init : array of shape (n_components, n_features),
        Initial value for the dictionary for warm restart scenarios.

    code_init : array of shape (n_samples, n_components),
        Initial value for the sparse code for warm restart scenarios.

    callback : callable or None, optional (default: None)
        Callable that gets invoked every five iterations

    verbose : bool, optional (default: False)
        To control the verbosity of the procedure.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    return_n_iter : bool
        Whether or not to return the number of iterations.

    positive_dict : bool
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    positive_code : bool
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    Returns
    -------
    code : array of shape (n_samples, n_components)
        The sparse code factor in the matrix factorization.

    dictionary : array of shape (n_components, n_features),
        The dictionary factor in the matrix factorization.

    errors : array
        Vector of errors at each iteration.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.

    See also
    --------
    dict_learning_online
    DictionaryLearning
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    "
  [X n_components alpha & {:keys [max_iter tol method n_jobs dict_init code_init callback verbose random_state return_n_iter positive_dict positive_code]
                       :or {max_iter 100 tol 1e-08 method "lars" verbose false return_n_iter false positive_dict false positive_code false}} ]
    (py/call-attr-kw decomposition "dict_learning" [X n_components alpha] {:max_iter max_iter :tol tol :method method :n_jobs n_jobs :dict_init dict_init :code_init code_init :callback callback :verbose verbose :random_state random_state :return_n_iter return_n_iter :positive_dict positive_dict :positive_code positive_code }))

(defn dict-learning-online 
  "Solves a dictionary learning matrix factorization problem online.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                     (U,V)
                     with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code. This is
    accomplished by repeatedly iterating over mini-batches by slicing
    the input data.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Data matrix.

    n_components : int,
        Number of dictionary atoms to extract.

    alpha : float,
        Sparsity controlling parameter.

    n_iter : int,
        Number of iterations to perform.

    return_code : boolean,
        Whether to also return the code U or just the dictionary V.

    dict_init : array of shape (n_components, n_features),
        Initial value for the dictionary for warm restart scenarios.

    callback : callable or None, optional (default: None)
        callable that gets invoked every five iterations

    batch_size : int,
        The number of samples to take in each batch.

    verbose : bool, optional (default: False)
        To control the verbosity of the procedure.

    shuffle : boolean,
        Whether to shuffle the data before splitting it in batches.

    n_jobs : int or None, optional (default=None)
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    method : {'lars', 'cd'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    iter_offset : int, default 0
        Number of previous iterations completed on the dictionary used for
        initialization.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    return_inner_stats : boolean, optional
        Return the inner statistics A (dictionary covariance) and B
        (data approximation). Useful to restart the algorithm in an
        online setting. If return_inner_stats is True, return_code is
        ignored

    inner_stats : tuple of (A, B) ndarrays
        Inner sufficient statistics that are kept by the algorithm.
        Passing them at initialization is useful in online settings, to
        avoid loosing the history of the evolution.
        A (n_components, n_components) is the dictionary covariance matrix.
        B (n_features, n_components) is the data approximation matrix

    return_n_iter : bool
        Whether or not to return the number of iterations.

    positive_dict : bool
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    positive_code : bool
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    Returns
    -------
    code : array of shape (n_samples, n_components),
        the sparse code (only returned if `return_code=True`)

    dictionary : array of shape (n_components, n_features),
        the solutions to the dictionary learning problem

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to `True`.

    See also
    --------
    dict_learning
    DictionaryLearning
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA

    "
  [X & {:keys [n_components alpha n_iter return_code dict_init callback batch_size verbose shuffle n_jobs method iter_offset random_state return_inner_stats inner_stats return_n_iter positive_dict positive_code]
                       :or {n_components 2 alpha 1 n_iter 100 return_code true batch_size 3 verbose false shuffle true method "lars" iter_offset 0 return_inner_stats false return_n_iter false positive_dict false positive_code false}} ]
    (py/call-attr-kw decomposition "dict_learning_online" [X] {:n_components n_components :alpha alpha :n_iter n_iter :return_code return_code :dict_init dict_init :callback callback :batch_size batch_size :verbose verbose :shuffle shuffle :n_jobs n_jobs :method method :iter_offset iter_offset :random_state random_state :return_inner_stats return_inner_stats :inner_stats inner_stats :return_n_iter return_n_iter :positive_dict positive_dict :positive_code positive_code }))

(defn fastica 
  "Perform Fast Independent Component Analysis.

    Read more in the :ref:`User Guide <ICA>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    n_components : int, optional
        Number of components to extract. If None no dimension reduction
        is performed.

    algorithm : {'parallel', 'deflation'}, optional
        Apply a parallel or deflational FASTICA algorithm.

    whiten : boolean, optional
        If True perform an initial whitening of the data.
        If False, the data is assumed to have already been
        preprocessed: it should be centered, normed and white.
        Otherwise you will get incorrect results.
        In this case the parameter n_components will be ignored.

    fun : string or function, optional. Default: 'logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. The derivative should be averaged along its last dimension.
        Example:

        def my_g(x):
            return x ** 3, np.mean(3 * x ** 2, axis=-1)

    fun_args : dictionary, optional
        Arguments to send to the functional form.
        If empty or None and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}

    max_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged.

    w_init : (n_components, n_components) array, optional
        Initial un-mixing array of dimension (n.comp,n.comp).
        If None (default) then an array of normal r.v.'s is used.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    return_X_mean : bool, optional
        If True, X_mean is returned too.

    compute_sources : bool, optional
        If False, sources are not computed, but only the rotation matrix.
        This can save memory when working with big data. Defaults to True.

    return_n_iter : bool, optional
        Whether or not to return the number of iterations.

    Returns
    -------
    K : array, shape (n_components, n_features) | None.
        If whiten is 'True', K is the pre-whitening matrix that projects data
        onto the first n_components principal components. If whiten is 'False',
        K is 'None'.

    W : array, shape (n_components, n_components)
        Estimated un-mixing matrix.
        The mixing matrix can be obtained by::

            w = np.dot(W, K.T)
            A = w.T * (w * w.T).I

    S : array, shape (n_samples, n_components) | None
        Estimated source matrix

    X_mean : array, shape (n_features, )
        The mean over features. Returned only if return_X_mean is True.

    n_iter : int
        If the algorithm is \"deflation\", n_iter is the
        maximum number of iterations run across all components. Else
        they are just the number of iterations taken to converge. This is
        returned only when return_n_iter is set to `True`.

    Notes
    -----

    The data matrix X is considered to be a linear combination of
    non-Gaussian (independent) components i.e. X = AS where columns of S
    contain the independent components and A is a linear mixing
    matrix. In short ICA attempts to `un-mix' the data by estimating an
    un-mixing matrix W where ``S = W K X.``

    This implementation was originally made for data of shape
    [n_features, n_samples]. Now the input is transposed
    before the algorithm is applied. This makes it slightly
    faster for Fortran-ordered input.

    Implemented using FastICA:
    *A. Hyvarinen and E. Oja, Independent Component Analysis:
    Algorithms and Applications, Neural Networks, 13(4-5), 2000,
    pp. 411-430*

    "
  [X n_components & {:keys [algorithm whiten fun fun_args max_iter tol w_init random_state return_X_mean compute_sources return_n_iter]
                       :or {algorithm "parallel" whiten true fun "logcosh" max_iter 200 tol 0.0001 return_X_mean false compute_sources true return_n_iter false}} ]
    (py/call-attr-kw decomposition "fastica" [X n_components] {:algorithm algorithm :whiten whiten :fun fun :fun_args fun_args :max_iter max_iter :tol tol :w_init w_init :random_state random_state :return_X_mean return_X_mean :compute_sources compute_sources :return_n_iter return_n_iter }))

(defn non-negative-factorization 
  "Compute Non-negative Matrix Factorization (NMF)

    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is::

        0.5 * ||X - WH||_Fro^2
        + alpha * l1_ratio * ||vec(W)||_1
        + alpha * l1_ratio * ||vec(H)||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
        + 0.5 * alpha * (1 - l1_ratio) * ||H||_Fro^2

    Where::

        ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
        ||vec(A)||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)

    For multiplicative-update ('mu') solver, the Frobenius norm
    (0.5 * ||X - WH||_Fro^2) can be changed into another beta-divergence loss,
    by changing the beta_loss parameter.

    The objective function is minimized with an alternating minimization of W
    and H. If H is given and update_H=False, it solves for W only.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Constant matrix.

    W : array-like, shape (n_samples, n_components)
        If init='custom', it is used as initial guess for the solution.

    H : array-like, shape (n_components, n_features)
        If init='custom', it is used as initial guess for the solution.
        If update_H=False, it is used as a constant, to solve for W only.

    n_components : integer
        Number of components, if n_components is not set all features
        are kept.

    init : None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar' | 'custom'
        Method used to initialize the procedure.
        Default: 'random'.

        The default value will change from 'random' to None in version 0.23
        to make it consistent with decomposition.NMF.

        Valid options:

        - None: 'nndsvd' if n_components < n_features, otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

    update_H : boolean, default: True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    solver : 'cd' | 'mu'
        Numerical solver to use:
        'cd' is a Coordinate Descent solver that uses Fast Hierarchical
            Alternating Least Squares (Fast HALS).
        'mu' is a Multiplicative Update solver.

        .. versionadded:: 0.17
           Coordinate Descent solver.

        .. versionadded:: 0.19
           Multiplicative Update solver.

    beta_loss : float or string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.

        .. versionadded:: 0.19

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before timing out.

    alpha : double, default: 0.
        Constant that multiplies the regularization terms.

    l1_ratio : double, default: 0.
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    regularization : 'both' | 'components' | 'transformation' | None
        Select whether the regularization affects the components (H), the
        transformation (W), both or none of them.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : integer, default: 0
        The verbosity level.

    shuffle : boolean, default: False
        If true, randomize the order of coordinates in the CD solver.

    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : array-like, shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        Actual number of iterations.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import non_negative_factorization
    >>> W, H, n_iter = non_negative_factorization(X, n_components=2,
    ... init='random', random_state=0)

    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. \"Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations.\"
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.

    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    "
  [X W H n_components & {:keys [init update_H solver beta_loss tol max_iter alpha l1_ratio regularization random_state verbose shuffle]
                       :or {init "warn" update_H true solver "cd" beta_loss "frobenius" tol 0.0001 max_iter 200 alpha 0.0 l1_ratio 0.0 verbose 0 shuffle false}} ]
    (py/call-attr-kw decomposition "non_negative_factorization" [X W H n_components] {:init init :update_H update_H :solver solver :beta_loss beta_loss :tol tol :max_iter max_iter :alpha alpha :l1_ratio l1_ratio :regularization regularization :random_state random_state :verbose verbose :shuffle shuffle }))

(defn randomized-svd 
  "Computes a truncated randomized SVD

    Parameters
    ----------
    M : ndarray or sparse matrix
        Matrix to decompose

    n_components : int
        Number of singular values and vectors to extract.

    n_oversamples : int (default is 10)
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.

    n_iter : int or 'auto' (default is 'auto')
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
        This improves precision with few components.

        .. versionchanged:: 0.18

    power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

        .. versionadded:: 0.18

    transpose : True, False or 'auto' (default)
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.

        .. versionchanged:: 0.18

    flip_sign : boolean, (True by default)
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision).

    References
    ----------
    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions
      Halko, et al., 2009 https://arxiv.org/abs/0909.4061

    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    "
  [M n_components & {:keys [n_oversamples n_iter power_iteration_normalizer transpose flip_sign random_state]
                       :or {n_oversamples 10 n_iter "auto" power_iteration_normalizer "auto" transpose "auto" flip_sign true random_state 0}} ]
    (py/call-attr-kw decomposition "randomized_svd" [M n_components] {:n_oversamples n_oversamples :n_iter n_iter :power_iteration_normalizer power_iteration_normalizer :transpose transpose :flip_sign flip_sign :random_state random_state }))

(defn sparse-encode 
  "Sparse coding

    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::

        X ~= code * dictionary

    Read more in the :ref:`User Guide <SparseCoder>`.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Data matrix

    dictionary : array of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows for meaningful
        output.

    gram : array, shape=(n_components, n_components)
        Precomputed Gram matrix, dictionary * dictionary'

    cov : array, shape=(n_components, n_samples)
        Precomputed covariance, dictionary' * X

    algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated components are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than alpha from
        the projection dictionary * X'

    n_nonzero_coefs : int, 0.1 * n_features by default
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case.

    alpha : float, 1. by default
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.

    copy_cov : boolean, optional
        Whether to copy the precomputed covariance matrix; if False, it may be
        overwritten.

    init : array of shape (n_samples, n_components)
        Initialization value of the sparse codes. Only used if
        `algorithm='lasso_cd'`.

    max_iter : int, 1000 by default
        Maximum number of iterations to perform if `algorithm='lasso_cd'`.

    n_jobs : int or None, optional (default=None)
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    check_input : boolean, optional
        If False, the input arrays X and dictionary will not be checked.

    verbose : int, optional
        Controls the verbosity; the higher, the more messages. Defaults to 0.

    positive : boolean, optional
        Whether to enforce positivity when finding the encoding.

        .. versionadded:: 0.20

    Returns
    -------
    code : array of shape (n_samples, n_components)
        The sparse codes

    See also
    --------
    sklearn.linear_model.lars_path
    sklearn.linear_model.orthogonal_mp
    sklearn.linear_model.Lasso
    SparseCoder
    "
  [X dictionary gram cov & {:keys [algorithm n_nonzero_coefs alpha copy_cov init max_iter n_jobs check_input verbose positive]
                       :or {algorithm "lasso_lars" copy_cov true max_iter 1000 check_input true verbose 0 positive false}} ]
    (py/call-attr-kw decomposition "sparse_encode" [X dictionary gram cov] {:algorithm algorithm :n_nonzero_coefs n_nonzero_coefs :alpha alpha :copy_cov copy_cov :init init :max_iter max_iter :n_jobs n_jobs :check_input check_input :verbose verbose :positive positive }))
