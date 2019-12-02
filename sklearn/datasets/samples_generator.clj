(ns sklearn.datasets.samples-generator
  "
Generate samples of synthetic data sets.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce samples-generator (import-module "sklearn.datasets.samples_generator"))

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
    (py/call-attr-kw samples-generator "check_array" [array] {:accept_sparse accept_sparse :accept_large_sparse accept_large_sparse :dtype dtype :order order :copy copy :force_all_finite force_all_finite :ensure_2d ensure_2d :allow_nd allow_nd :ensure_min_samples ensure_min_samples :ensure_min_features ensure_min_features :warn_on_dtype warn_on_dtype :estimator estimator }))

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
  (py/call-attr samples-generator "check_random_state"  seed ))

(defn make-biclusters 
  "Generate an array with constant block diagonal structure for
    biclustering.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    shape : iterable (n_rows, n_cols)
        The shape of the result.

    n_clusters : integer
        The number of biclusters.

    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise.

    minval : int, optional (default=10)
        Minimum value of a bicluster.

    maxval : int, optional (default=100)
        Maximum value of a bicluster.

    shuffle : boolean, optional (default=True)
        Shuffle the samples.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape `shape`
        The generated array.

    rows : array of shape (n_clusters, X.shape[0],)
        The indicators for cluster membership of each row.

    cols : array of shape (n_clusters, X.shape[1],)
        The indicators for cluster membership of each column.

    References
    ----------

    .. [1] Dhillon, I. S. (2001, August). Co-clustering documents and
        words using bipartite spectral graph partitioning. In Proceedings
        of the seventh ACM SIGKDD international conference on Knowledge
        discovery and data mining (pp. 269-274). ACM.

    See also
    --------
    make_checkerboard
    "
  [shape n_clusters & {:keys [noise minval maxval shuffle random_state]
                       :or {noise 0.0 minval 10 maxval 100 shuffle true}} ]
    (py/call-attr-kw samples-generator "make_biclusters" [shape n_clusters] {:noise noise :minval minval :maxval maxval :shuffle shuffle :random_state random_state }))

(defn make-blobs 
  "Generate isotropic Gaussian blobs for clustering.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int or array-like, optional (default=100)
        If int, it is the total number of points equally divided among
        clusters.
        If array-like, each element of the sequence indicates
        the number of samples per cluster.

    n_features : int, optional (default=2)
        The number of features for each sample.

    centers : int or array of shape [n_centers, n_features], optional
        (default=None)
        The number of centers to generate, or the fixed center locations.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be
        either None or an array of length equal to the length of n_samples.

    cluster_std : float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.

    center_box : pair of floats (min, max), optional (default=(-10.0, 10.0))
        The bounding box for each cluster center when centers are
        generated at random.

    shuffle : boolean, optional (default=True)
        Shuffle the samples.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.

    Examples
    --------
    >>> from sklearn.datasets.samples_generator import make_blobs
    >>> X, y = make_blobs(n_samples=10, centers=3, n_features=2,
    ...                   random_state=0)
    >>> print(X.shape)
    (10, 2)
    >>> y
    array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])
    >>> X, y = make_blobs(n_samples=[3, 3, 4], centers=None, n_features=2,
    ...                   random_state=0)
    >>> print(X.shape)
    (10, 2)
    >>> y
    array([0, 1, 2, 0, 2, 2, 2, 1, 1, 0])

    See also
    --------
    make_classification: a more intricate variant
    "
  [ & {:keys [n_samples n_features centers cluster_std center_box shuffle random_state]
       :or {n_samples 100 n_features 2 cluster_std 1.0 center_box (-10.0, 10.0) shuffle true}} ]
  
   (py/call-attr-kw samples-generator "make_blobs" [] {:n_samples n_samples :n_features n_features :centers centers :cluster_std cluster_std :center_box center_box :shuffle shuffle :random_state random_state }))

(defn make-checkerboard 
  "Generate an array with block checkerboard structure for
    biclustering.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    shape : iterable (n_rows, n_cols)
        The shape of the result.

    n_clusters : integer or iterable (n_row_clusters, n_column_clusters)
        The number of row and column clusters.

    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise.

    minval : int, optional (default=10)
        Minimum value of a bicluster.

    maxval : int, optional (default=100)
        Maximum value of a bicluster.

    shuffle : boolean, optional (default=True)
        Shuffle the samples.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape `shape`
        The generated array.

    rows : array of shape (n_clusters, X.shape[0],)
        The indicators for cluster membership of each row.

    cols : array of shape (n_clusters, X.shape[1],)
        The indicators for cluster membership of each column.


    References
    ----------

    .. [1] Kluger, Y., Basri, R., Chang, J. T., & Gerstein, M. (2003).
        Spectral biclustering of microarray data: coclustering genes
        and conditions. Genome research, 13(4), 703-716.

    See also
    --------
    make_biclusters
    "
  [shape n_clusters & {:keys [noise minval maxval shuffle random_state]
                       :or {noise 0.0 minval 10 maxval 100 shuffle true}} ]
    (py/call-attr-kw samples-generator "make_checkerboard" [shape n_clusters] {:noise noise :minval minval :maxval maxval :shuffle shuffle :random_state random_state }))

(defn make-circles 
  "Make a large circle containing a smaller circle in 2d.

    A simple toy dataset to visualize clustering and classification
    algorithms.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated. If odd, the inner circle will
        have one point more than the outer circle.

    shuffle : bool, optional (default=True)
        Whether to shuffle the samples.

    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    factor : 0 < double < 1 (default=.8)
        Scale factor between inner and outer circle.

    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    "
  [ & {:keys [n_samples shuffle noise random_state factor]
       :or {n_samples 100 shuffle true factor 0.8}} ]
  
   (py/call-attr-kw samples-generator "make_circles" [] {:n_samples n_samples :shuffle shuffle :noise noise :random_state random_state :factor factor }))

(defn make-classification 
  "Generate a random n-class classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an ``n_informative``-dimensional hypercube with sides of
    length ``2*class_sep`` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Without shuffling, ``X`` horizontally stacks features in the following
    order: the primary ``n_informative`` features, followed by ``n_redundant``
    linear combinations of the informative features, followed by ``n_repeated``
    duplicates, drawn randomly with replacement from the informative and
    redundant features. The remaining features are filled with random noise.
    Thus, without shuffling, all useful features are contained in the columns
    ``X[:, :n_informative + n_redundant + n_repeated]``.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=20)
        The total number of features. These comprise ``n_informative``
        informative features, ``n_redundant`` redundant features,
        ``n_repeated`` duplicated features and
        ``n_features-n_informative-n_redundant-n_repeated`` useless features
        drawn at random.

    n_informative : int, optional (default=2)
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension ``n_informative``. For each cluster,
        informative features are drawn independently from  N(0, 1) and then
        randomly linearly combined within each cluster in order to add
        covariance. The clusters are then placed on the vertices of the
        hypercube.

    n_redundant : int, optional (default=2)
        The number of redundant features. These features are generated as
        random linear combinations of the informative features.

    n_repeated : int, optional (default=0)
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.

    n_classes : int, optional (default=2)
        The number of classes (or labels) of the classification problem.

    n_clusters_per_class : int, optional (default=2)
        The number of clusters per class.

    weights : list of floats or None (default=None)
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if ``len(weights) == n_classes - 1``,
        then the last class weight is automatically inferred.
        More than ``n_samples`` samples may be returned if the sum of
        ``weights`` exceeds 1.

    flip_y : float, optional (default=0.01)
        The fraction of samples whose class are randomly exchanged. Larger
        values introduce noise in the labels and make the classification
        task harder.

    class_sep : float, optional (default=1.0)
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.

    hypercube : boolean, optional (default=True)
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    shift : float, array of shape [n_features] or None, optional (default=0.0)
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    scale : float, array of shape [n_features] or None, optional (default=1.0)
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for class membership of each sample.

    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the \"Madelon\" dataset.

    References
    ----------
    .. [1] I. Guyon, \"Design of experiments for the NIPS 2003 variable
           selection benchmark\", 2003.

    See also
    --------
    make_blobs: simplified variant
    make_multilabel_classification: unrelated generator for multilabel tasks
    "
  [ & {:keys [n_samples n_features n_informative n_redundant n_repeated n_classes n_clusters_per_class weights flip_y class_sep hypercube shift scale shuffle random_state]
       :or {n_samples 100 n_features 20 n_informative 2 n_redundant 2 n_repeated 0 n_classes 2 n_clusters_per_class 2 flip_y 0.01 class_sep 1.0 hypercube true shift 0.0 scale 1.0 shuffle true}} ]
  
   (py/call-attr-kw samples-generator "make_classification" [] {:n_samples n_samples :n_features n_features :n_informative n_informative :n_redundant n_redundant :n_repeated n_repeated :n_classes n_classes :n_clusters_per_class n_clusters_per_class :weights weights :flip_y flip_y :class_sep class_sep :hypercube hypercube :shift shift :scale scale :shuffle shuffle :random_state random_state }))

(defn make-friedman1 
  "Generate the \"Friedman #1\" regression problem

    This dataset is described in Friedman [1] and Breiman [2].

    Inputs `X` are independent features uniformly distributed on the interval
    [0, 1]. The output `y` is created according to the formula::

        y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] + noise * N(0, 1).

    Out of the `n_features` features, only 5 are actually used to compute
    `y`. The remaining features are independent of `y`.

    The number of features has to be >= 5.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=10)
        The number of features. Should be at least 5.

    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise applied to the output.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset noise. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The input samples.

    y : array of shape [n_samples]
        The output values.

    References
    ----------
    .. [1] J. Friedman, \"Multivariate adaptive regression splines\", The Annals
           of Statistics 19 (1), pages 1-67, 1991.

    .. [2] L. Breiman, \"Bagging predictors\", Machine Learning 24,
           pages 123-140, 1996.
    "
  [ & {:keys [n_samples n_features noise random_state]
       :or {n_samples 100 n_features 10 noise 0.0}} ]
  
   (py/call-attr-kw samples-generator "make_friedman1" [] {:n_samples n_samples :n_features n_features :noise noise :random_state random_state }))

(defn make-friedman2 
  "Generate the \"Friedman #2\" regression problem

    This dataset is described in Friedman [1] and Breiman [2].

    Inputs `X` are 4 independent features uniformly distributed on the
    intervals::

        0 <= X[:, 0] <= 100,
        40 * pi <= X[:, 1] <= 560 * pi,
        0 <= X[:, 2] <= 1,
        1 <= X[:, 3] <= 11.

    The output `y` is created according to the formula::

        y(X) = (X[:, 0] ** 2 + (X[:, 1] * X[:, 2]  - 1 / (X[:, 1] * X[:, 3])) ** 2) ** 0.5 + noise * N(0, 1).

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise applied to the output.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset noise. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, 4]
        The input samples.

    y : array of shape [n_samples]
        The output values.

    References
    ----------
    .. [1] J. Friedman, \"Multivariate adaptive regression splines\", The Annals
           of Statistics 19 (1), pages 1-67, 1991.

    .. [2] L. Breiman, \"Bagging predictors\", Machine Learning 24,
           pages 123-140, 1996.
    "
  [ & {:keys [n_samples noise random_state]
       :or {n_samples 100 noise 0.0}} ]
  
   (py/call-attr-kw samples-generator "make_friedman2" [] {:n_samples n_samples :noise noise :random_state random_state }))

(defn make-friedman3 
  "Generate the \"Friedman #3\" regression problem

    This dataset is described in Friedman [1] and Breiman [2].

    Inputs `X` are 4 independent features uniformly distributed on the
    intervals::

        0 <= X[:, 0] <= 100,
        40 * pi <= X[:, 1] <= 560 * pi,
        0 <= X[:, 2] <= 1,
        1 <= X[:, 3] <= 11.

    The output `y` is created according to the formula::

        y(X) = arctan((X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) / X[:, 0]) + noise * N(0, 1).

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise applied to the output.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset noise. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, 4]
        The input samples.

    y : array of shape [n_samples]
        The output values.

    References
    ----------
    .. [1] J. Friedman, \"Multivariate adaptive regression splines\", The Annals
           of Statistics 19 (1), pages 1-67, 1991.

    .. [2] L. Breiman, \"Bagging predictors\", Machine Learning 24,
           pages 123-140, 1996.
    "
  [ & {:keys [n_samples noise random_state]
       :or {n_samples 100 noise 0.0}} ]
  
   (py/call-attr-kw samples-generator "make_friedman3" [] {:n_samples n_samples :noise noise :random_state random_state }))

(defn make-gaussian-quantiles 
  "Generate isotropic Gaussian and label samples by quantile

    This classification dataset is constructed by taking a multi-dimensional
    standard normal distribution and defining classes separated by nested
    concentric multi-dimensional spheres such that roughly equal numbers of
    samples are in each class (quantiles of the :math:`\chi^2` distribution).

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    mean : array of shape [n_features], optional (default=None)
        The mean of the multi-dimensional normal distribution.
        If None then use the origin (0, 0, ...).

    cov : float, optional (default=1.)
        The covariance matrix will be this value times the unit matrix. This
        dataset only produces symmetric normal distributions.

    n_samples : int, optional (default=100)
        The total number of points equally divided among classes.

    n_features : int, optional (default=2)
        The number of features for each sample.

    n_classes : int, optional (default=3)
        The number of classes

    shuffle : boolean, optional (default=True)
        Shuffle the samples.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for quantile membership of each sample.

    Notes
    -----
    The dataset is from Zhu et al [1].

    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, \"Multi-class AdaBoost\", 2009.

    "
  [mean & {:keys [cov n_samples n_features n_classes shuffle random_state]
                       :or {cov 1.0 n_samples 100 n_features 2 n_classes 3 shuffle true}} ]
    (py/call-attr-kw samples-generator "make_gaussian_quantiles" [mean] {:cov cov :n_samples n_samples :n_features n_features :n_classes n_classes :shuffle shuffle :random_state random_state }))

(defn make-hastie-10-2 
  "Generates data for binary classification used in
    Hastie et al. 2009, Example 10.2.

    The ten features are standard independent Gaussian and
    the target ``y`` is defined by::

      y[i] = 1 if np.sum(X[i] ** 2) > 9.34 else -1

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=12000)
        The number of samples.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, 10]
        The input samples.

    y : array of shape [n_samples]
        The output values.

    References
    ----------
    .. [1] T. Hastie, R. Tibshirani and J. Friedman, \"Elements of Statistical
           Learning Ed. 2\", Springer, 2009.

    See also
    --------
    make_gaussian_quantiles: a generalization of this dataset approach
    "
  [ & {:keys [n_samples random_state]
       :or {n_samples 12000}} ]
  
   (py/call-attr-kw samples-generator "make_hastie_10_2" [] {:n_samples n_samples :random_state random_state }))

(defn make-low-rank-matrix 
  "Generate a mostly low rank matrix with bell-shaped singular values

    Most of the variance can be explained by a bell-shaped curve of width
    effective_rank: the low rank part of the singular values profile is::

        (1 - tail_strength) * exp(-1.0 * (i / effective_rank) ** 2)

    The remaining singular values' tail is fat, decreasing as::

        tail_strength * exp(-0.1 * i / effective_rank).

    The low rank part of the profile can be considered the structured
    signal part of the data while the tail can be considered the noisy
    part of the data that cannot be summarized by a low number of linear
    components (singular vectors).

    This kind of singular profiles is often seen in practice, for instance:
     - gray level pictures of faces
     - TF-IDF vectors of text documents crawled from the web

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=100)
        The number of features.

    effective_rank : int, optional (default=10)
        The approximate number of singular vectors required to explain most of
        the data by linear combinations.

    tail_strength : float between 0.0 and 1.0, optional (default=0.5)
        The relative importance of the fat noisy tail of the singular values
        profile.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The matrix.
    "
  [ & {:keys [n_samples n_features effective_rank tail_strength random_state]
       :or {n_samples 100 n_features 100 effective_rank 10 tail_strength 0.5}} ]
  
   (py/call-attr-kw samples-generator "make_low_rank_matrix" [] {:n_samples n_samples :n_features n_features :effective_rank effective_rank :tail_strength tail_strength :random_state random_state }))

(defn make-moons 
  "Make two interleaving half circles

    A simple toy dataset to visualize clustering and classification
    algorithms. Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.

    shuffle : bool, optional (default=True)
        Whether to shuffle the samples.

    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    "
  [ & {:keys [n_samples shuffle noise random_state]
       :or {n_samples 100 shuffle true}} ]
  
   (py/call-attr-kw samples-generator "make_moons" [] {:n_samples n_samples :shuffle shuffle :noise noise :random_state random_state }))

(defn make-multilabel-classification 
  "Generate a random multilabel classification problem.

    For each sample, the generative process is:
        - pick the number of labels: n ~ Poisson(n_labels)
        - n times, choose a class c: c ~ Multinomial(theta)
        - pick the document length: k ~ Poisson(length)
        - k times, choose a word: w ~ Multinomial(theta_c)

    In the above process, rejection sampling is used to make sure that
    n is never zero or more than `n_classes`, and that the document length
    is never zero. Likewise, we reject classes which have already been chosen.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=20)
        The total number of features.

    n_classes : int, optional (default=5)
        The number of classes of the classification problem.

    n_labels : int, optional (default=2)
        The average number of labels per instance. More precisely, the number
        of labels per sample is drawn from a Poisson distribution with
        ``n_labels`` as its expected value, but samples are bounded (using
        rejection sampling) by ``n_classes``, and must be nonzero if
        ``allow_unlabeled`` is False.

    length : int, optional (default=50)
        The sum of the features (number of words if documents) is drawn from
        a Poisson distribution with this expected value.

    allow_unlabeled : bool, optional (default=True)
        If ``True``, some instances might not belong to any class.

    sparse : bool, optional (default=False)
        If ``True``, return a sparse feature matrix

        .. versionadded:: 0.17
           parameter to allow *sparse* output.

    return_indicator : 'dense' (default) | 'sparse' | False
        If ``dense`` return ``Y`` in the dense binary indicator format. If
        ``'sparse'`` return ``Y`` in the sparse binary indicator format.
        ``False`` returns a list of lists of labels.

    return_distributions : bool, optional (default=False)
        If ``True``, return the prior class probability and conditional
        probabilities of features given classes, from which the data was
        drawn.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    Y : array or sparse CSR matrix of shape [n_samples, n_classes]
        The label sets.

    p_c : array, shape [n_classes]
        The probability of each class being drawn. Only returned if
        ``return_distributions=True``.

    p_w_c : array, shape [n_features, n_classes]
        The probability of each feature being drawn given each class.
        Only returned if ``return_distributions=True``.

    "
  [ & {:keys [n_samples n_features n_classes n_labels length allow_unlabeled sparse return_indicator return_distributions random_state]
       :or {n_samples 100 n_features 20 n_classes 5 n_labels 2 length 50 allow_unlabeled true sparse false return_indicator "dense" return_distributions false}} ]
  
   (py/call-attr-kw samples-generator "make_multilabel_classification" [] {:n_samples n_samples :n_features n_features :n_classes n_classes :n_labels n_labels :length length :allow_unlabeled allow_unlabeled :sparse sparse :return_indicator return_indicator :return_distributions return_distributions :random_state random_state }))

(defn make-regression 
  "Generate a random regression problem.

    The input set can either be well conditioned (by default) or have a low
    rank-fat tail singular profile. See :func:`make_low_rank_matrix` for
    more details.

    The output is generated by applying a (potentially biased) random linear
    regression model with `n_informative` nonzero regressors to the previously
    generated input and some gaussian centered noise with some adjustable
    scale.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=100)
        The number of features.

    n_informative : int, optional (default=10)
        The number of informative features, i.e., the number of features used
        to build the linear model used to generate the output.

    n_targets : int, optional (default=1)
        The number of regression targets, i.e., the dimension of the y output
        vector associated with a sample. By default, the output is a scalar.

    bias : float, optional (default=0.0)
        The bias term in the underlying linear model.

    effective_rank : int or None, optional (default=None)
        if not None:
            The approximate number of singular vectors required to explain most
            of the input data by linear combinations. Using this kind of
            singular spectrum in the input allows the generator to reproduce
            the correlations often observed in practice.
        if None:
            The input set is well conditioned, centered and gaussian with
            unit variance.

    tail_strength : float between 0.0 and 1.0, optional (default=0.5)
        The relative importance of the fat noisy tail of the singular values
        profile if `effective_rank` is not None.

    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise applied to the output.

    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.

    coef : boolean, optional (default=False)
        If True, the coefficients of the underlying linear model are returned.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The input samples.

    y : array of shape [n_samples] or [n_samples, n_targets]
        The output values.

    coef : array of shape [n_features] or [n_features, n_targets], optional
        The coefficient of the underlying linear model. It is returned only if
        coef is True.
    "
  [ & {:keys [n_samples n_features n_informative n_targets bias effective_rank tail_strength noise shuffle coef random_state]
       :or {n_samples 100 n_features 100 n_informative 10 n_targets 1 bias 0.0 tail_strength 0.5 noise 0.0 shuffle true coef false}} ]
  
   (py/call-attr-kw samples-generator "make_regression" [] {:n_samples n_samples :n_features n_features :n_informative n_informative :n_targets n_targets :bias bias :effective_rank effective_rank :tail_strength tail_strength :noise noise :shuffle shuffle :coef coef :random_state random_state }))

(defn make-s-curve 
  "Generate an S curve dataset.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of sample points on the S curve.

    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, 3]
        The points.

    t : array of shape [n_samples]
        The univariate position of the sample according to the main dimension
        of the points in the manifold.
    "
  [ & {:keys [n_samples noise random_state]
       :or {n_samples 100 noise 0.0}} ]
  
   (py/call-attr-kw samples-generator "make_s_curve" [] {:n_samples n_samples :noise noise :random_state random_state }))

(defn make-sparse-coded-signal 
  "Generate a signal as a sparse combination of dictionary elements.

    Returns a matrix Y = DX, such as D is (n_features, n_components),
    X is (n_components, n_samples) and each column of X has exactly
    n_nonzero_coefs non-zero elements.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int
        number of samples to generate

    n_components :  int,
        number of components in the dictionary

    n_features : int
        number of features of the dataset to generate

    n_nonzero_coefs : int
        number of active (non-zero) coefficients in each sample

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    data : array of shape [n_features, n_samples]
        The encoded signal (Y).

    dictionary : array of shape [n_features, n_components]
        The dictionary with normalized components (D).

    code : array of shape [n_components, n_samples]
        The sparse code such that each column of this matrix has exactly
        n_nonzero_coefs non-zero items (X).

    "
  [ n_samples n_components n_features n_nonzero_coefs random_state ]
  (py/call-attr samples-generator "make_sparse_coded_signal"  n_samples n_components n_features n_nonzero_coefs random_state ))

(defn make-sparse-spd-matrix 
  "Generate a sparse symmetric definite positive matrix.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    dim : integer, optional (default=1)
        The size of the random matrix to generate.

    alpha : float between 0 and 1, optional (default=0.95)
        The probability that a coefficient is zero (see notes). Larger values
        enforce more sparsity.

    norm_diag : boolean, optional (default=False)
        Whether to normalize the output matrix to make the leading diagonal
        elements all 1

    smallest_coef : float between 0 and 1, optional (default=0.1)
        The value of the smallest coefficient.

    largest_coef : float between 0 and 1, optional (default=0.9)
        The value of the largest coefficient.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    prec : sparse matrix of shape (dim, dim)
        The generated matrix.

    Notes
    -----
    The sparsity is actually imposed on the cholesky factor of the matrix.
    Thus alpha does not translate directly into the filling fraction of
    the matrix itself.

    See also
    --------
    make_spd_matrix
    "
  [ & {:keys [dim alpha norm_diag smallest_coef largest_coef random_state]
       :or {dim 1 alpha 0.95 norm_diag false smallest_coef 0.1 largest_coef 0.9}} ]
  
   (py/call-attr-kw samples-generator "make_sparse_spd_matrix" [] {:dim dim :alpha alpha :norm_diag norm_diag :smallest_coef smallest_coef :largest_coef largest_coef :random_state random_state }))

(defn make-sparse-uncorrelated 
  "Generate a random regression problem with sparse uncorrelated design

    This dataset is described in Celeux et al [1]. as::

        X ~ N(0, 1)
        y(X) = X[:, 0] + 2 * X[:, 1] - 2 * X[:, 2] - 1.5 * X[:, 3]

    Only the first 4 features are informative. The remaining features are
    useless.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=10)
        The number of features.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The input samples.

    y : array of shape [n_samples]
        The output values.

    References
    ----------
    .. [1] G. Celeux, M. El Anbari, J.-M. Marin, C. P. Robert,
           \"Regularization in regression: comparing Bayesian and frequentist
           methods in a poorly informative situation\", 2009.
    "
  [ & {:keys [n_samples n_features random_state]
       :or {n_samples 100 n_features 10}} ]
  
   (py/call-attr-kw samples-generator "make_sparse_uncorrelated" [] {:n_samples n_samples :n_features n_features :random_state random_state }))

(defn make-spd-matrix 
  "Generate a random symmetric, positive-definite matrix.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_dim : int
        The matrix dimension.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_dim, n_dim]
        The random symmetric, positive-definite matrix.

    See also
    --------
    make_sparse_spd_matrix
    "
  [ n_dim random_state ]
  (py/call-attr samples-generator "make_spd_matrix"  n_dim random_state ))

(defn make-swiss-roll 
  "Generate a swiss roll dataset.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of sample points on the S curve.

    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, 3]
        The points.

    t : array of shape [n_samples]
        The univariate position of the sample according to the main dimension
        of the points in the manifold.

    Notes
    -----
    The algorithm is from Marsland [1].

    References
    ----------
    .. [1] S. Marsland, \"Machine Learning: An Algorithmic Perspective\",
           Chapter 10, 2009.
           http://seat.massey.ac.nz/personal/s.r.marsland/Code/10/lle.py
    "
  [ & {:keys [n_samples noise random_state]
       :or {n_samples 100 noise 0.0}} ]
  
   (py/call-attr-kw samples-generator "make_swiss_roll" [] {:n_samples n_samples :noise noise :random_state random_state }))

(defn util-shuffle 
  "Shuffle arrays or sparse matrices in a consistent way

    This is a convenience alias to ``resample(*arrays, replace=False)`` to do
    random permutations of the collections.

    Parameters
    ----------
    *arrays : sequence of indexable data-structures
        Indexable data-structures can be arrays, lists, dataframes or scipy
        sparse matrices with consistent first dimension.

    Other Parameters
    ----------------
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    n_samples : int, None by default
        Number of samples to generate. If left to None this is
        automatically set to the first dimension of the arrays.

    Returns
    -------
    shuffled_arrays : sequence of indexable data-structures
        Sequence of shuffled copies of the collections. The original arrays
        are not impacted.

    Examples
    --------
    It is possible to mix sparse and dense arrays in the same run::

      >>> X = np.array([[1., 0.], [2., 1.], [0., 0.]])
      >>> y = np.array([0, 1, 2])

      >>> from scipy.sparse import coo_matrix
      >>> X_sparse = coo_matrix(X)

      >>> from sklearn.utils import shuffle
      >>> X, X_sparse, y = shuffle(X, X_sparse, y, random_state=0)
      >>> X
      array([[0., 0.],
             [2., 1.],
             [1., 0.]])

      >>> X_sparse                   # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
      <3x2 sparse matrix of type '<... 'numpy.float64'>'
          with 3 stored elements in Compressed Sparse Row format>

      >>> X_sparse.toarray()
      array([[0., 0.],
             [2., 1.],
             [1., 0.]])

      >>> y
      array([2, 1, 0])

      >>> shuffle(y, n_samples=2, random_state=0)
      array([0, 1])

    See also
    --------
    :func:`sklearn.utils.resample`
    "
  [  ]
  (py/call-attr samples-generator "util_shuffle"  ))
