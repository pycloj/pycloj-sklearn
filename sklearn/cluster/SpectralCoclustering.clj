(ns sklearn.cluster.SpectralCoclustering
  "Spectral Co-Clustering algorithm (Dhillon, 2001).

    Clusters rows and columns of an array `X` to solve the relaxed
    normalized cut of the bipartite graph created from `X` as follows:
    the edge between row vertex `i` and column vertex `j` has weight
    `X[i, j]`.

    The resulting bicluster structure is block-diagonal, since each
    row and each column belongs to exactly one bicluster.

    Supports sparse matrices, as long as they are nonnegative.

    Read more in the :ref:`User Guide <spectral_coclustering>`.

    Parameters
    ----------
    n_clusters : integer, optional, default: 3
        The number of biclusters to find.

    svd_method : string, optional, default: 'randomized'
        Selects the algorithm for finding singular vectors. May be
        'randomized' or 'arpack'. If 'randomized', use
        :func:`sklearn.utils.extmath.randomized_svd`, which may be faster
        for large matrices. If 'arpack', use
        :func:`scipy.sparse.linalg.svds`, which is more accurate, but
        possibly slower in some cases.

    n_svd_vecs : int, optional, default: None
        Number of vectors to use in calculating the SVD. Corresponds
        to `ncv` when `svd_method=arpack` and `n_oversamples` when
        `svd_method` is 'randomized`.

    mini_batch : bool, optional, default: False
        Whether to use mini-batch k-means, which is faster but may get
        different results.

    init : {'k-means++', 'random' or an ndarray}
         Method for initialization of k-means algorithm; defaults to
         'k-means++'.

    n_init : int, optional, default: 10
        Number of random initializations that are tried with the
        k-means algorithm.

        If mini-batch k-means is used, the best initialization is
        chosen and the algorithm runs once. Otherwise, the algorithm
        is run for each initialization and the best solution chosen.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None (default)
        Used for randomizing the singular value decomposition and the k-means
        initialization. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    rows_ : array-like, shape (n_row_clusters, n_rows)
        Results of the clustering. `rows[i, r]` is True if
        cluster `i` contains row `r`. Available only after calling ``fit``.

    columns_ : array-like, shape (n_column_clusters, n_columns)
        Results of the clustering, like `rows`.

    row_labels_ : array-like, shape (n_rows,)
        The bicluster label of each row.

    column_labels_ : array-like, shape (n_cols,)
        The bicluster label of each column.

    Examples
    --------
    >>> from sklearn.cluster import SpectralCoclustering
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = SpectralCoclustering(n_clusters=2, random_state=0).fit(X)
    >>> clustering.row_labels_
    array([0, 1, 1, 0, 0, 0], dtype=int32)
    >>> clustering.column_labels_
    array([0, 0], dtype=int32)
    >>> clustering # doctest: +NORMALIZE_WHITESPACE
    SpectralCoclustering(init='k-means++', mini_batch=False, n_clusters=2,
               n_init=10, n_jobs=None, n_svd_vecs=None, random_state=0,
               svd_method='randomized')

    References
    ----------

    * Dhillon, Inderjit S, 2001. `Co-clustering documents and words using
      bipartite spectral graph partitioning
      <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.140.3011>`__.

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cluster (import-module "sklearn.cluster"))

(defn SpectralCoclustering 
  "Spectral Co-Clustering algorithm (Dhillon, 2001).

    Clusters rows and columns of an array `X` to solve the relaxed
    normalized cut of the bipartite graph created from `X` as follows:
    the edge between row vertex `i` and column vertex `j` has weight
    `X[i, j]`.

    The resulting bicluster structure is block-diagonal, since each
    row and each column belongs to exactly one bicluster.

    Supports sparse matrices, as long as they are nonnegative.

    Read more in the :ref:`User Guide <spectral_coclustering>`.

    Parameters
    ----------
    n_clusters : integer, optional, default: 3
        The number of biclusters to find.

    svd_method : string, optional, default: 'randomized'
        Selects the algorithm for finding singular vectors. May be
        'randomized' or 'arpack'. If 'randomized', use
        :func:`sklearn.utils.extmath.randomized_svd`, which may be faster
        for large matrices. If 'arpack', use
        :func:`scipy.sparse.linalg.svds`, which is more accurate, but
        possibly slower in some cases.

    n_svd_vecs : int, optional, default: None
        Number of vectors to use in calculating the SVD. Corresponds
        to `ncv` when `svd_method=arpack` and `n_oversamples` when
        `svd_method` is 'randomized`.

    mini_batch : bool, optional, default: False
        Whether to use mini-batch k-means, which is faster but may get
        different results.

    init : {'k-means++', 'random' or an ndarray}
         Method for initialization of k-means algorithm; defaults to
         'k-means++'.

    n_init : int, optional, default: 10
        Number of random initializations that are tried with the
        k-means algorithm.

        If mini-batch k-means is used, the best initialization is
        chosen and the algorithm runs once. Otherwise, the algorithm
        is run for each initialization and the best solution chosen.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None (default)
        Used for randomizing the singular value decomposition and the k-means
        initialization. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    rows_ : array-like, shape (n_row_clusters, n_rows)
        Results of the clustering. `rows[i, r]` is True if
        cluster `i` contains row `r`. Available only after calling ``fit``.

    columns_ : array-like, shape (n_column_clusters, n_columns)
        Results of the clustering, like `rows`.

    row_labels_ : array-like, shape (n_rows,)
        The bicluster label of each row.

    column_labels_ : array-like, shape (n_cols,)
        The bicluster label of each column.

    Examples
    --------
    >>> from sklearn.cluster import SpectralCoclustering
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = SpectralCoclustering(n_clusters=2, random_state=0).fit(X)
    >>> clustering.row_labels_
    array([0, 1, 1, 0, 0, 0], dtype=int32)
    >>> clustering.column_labels_
    array([0, 0], dtype=int32)
    >>> clustering # doctest: +NORMALIZE_WHITESPACE
    SpectralCoclustering(init='k-means++', mini_batch=False, n_clusters=2,
               n_init=10, n_jobs=None, n_svd_vecs=None, random_state=0,
               svd_method='randomized')

    References
    ----------

    * Dhillon, Inderjit S, 2001. `Co-clustering documents and words using
      bipartite spectral graph partitioning
      <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.140.3011>`__.

    "
  [ & {:keys [n_clusters svd_method n_svd_vecs mini_batch init n_init n_jobs random_state]
       :or {n_clusters 3 svd_method "randomized" mini_batch false init "k-means++" n_init 10}} ]
  
   (py/call-attr-kw cluster "SpectralCoclustering" [] {:n_clusters n_clusters :svd_method svd_method :n_svd_vecs n_svd_vecs :mini_batch mini_batch :init init :n_init n_init :n_jobs n_jobs :random_state random_state }))

(defn biclusters- 
  "Convenient way to get row and column indicators together.

        Returns the ``rows_`` and ``columns_`` members.
        "
  [ self ]
    (py/call-attr self "biclusters_"))

(defn fit 
  "Creates a biclustering for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        y : Ignored

        "
  [ self X y ]
  (py/call-attr self "fit"  self X y ))

(defn get-indices 
  "Row and column indices of the i'th bicluster.

        Only works if ``rows_`` and ``columns_`` attributes exist.

        Parameters
        ----------
        i : int
            The index of the cluster.

        Returns
        -------
        row_ind : np.array, dtype=np.intp
            Indices of rows in the dataset that belong to the bicluster.
        col_ind : np.array, dtype=np.intp
            Indices of columns in the dataset that belong to the bicluster.

        "
  [ self i ]
  (py/call-attr self "get_indices"  self i ))

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

(defn get-shape 
  "Shape of the i'th bicluster.

        Parameters
        ----------
        i : int
            The index of the cluster.

        Returns
        -------
        shape : (int, int)
            Number of rows and columns (resp.) in the bicluster.
        "
  [ self i ]
  (py/call-attr self "get_shape"  self i ))

(defn get-submatrix 
  "Returns the submatrix corresponding to bicluster `i`.

        Parameters
        ----------
        i : int
            The index of the cluster.
        data : array
            The data.

        Returns
        -------
        submatrix : array
            The submatrix corresponding to bicluster i.

        Notes
        -----
        Works with sparse matrices. Only works if ``rows_`` and
        ``columns_`` attributes exist.
        "
  [ self i data ]
  (py/call-attr self "get_submatrix"  self i data ))

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
