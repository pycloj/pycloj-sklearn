(ns sklearn.metrics.cluster.supervised
  "Utilities to evaluate the clustering performance of models.

Functions named as *_score return a scalar value to maximize: the higher the
better.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce supervised (import-module "sklearn.metrics.cluster.supervised"))

(defn adjusted-mutual-info-score 
  "Adjusted Mutual Information between two clusterings.

    Adjusted Mutual Information (AMI) is an adjustment of the Mutual
    Information (MI) score to account for chance. It accounts for the fact that
    the MI is generally higher for two clusterings with a larger number of
    clusters, regardless of whether there is actually more information shared.
    For two clusterings :math:`U` and :math:`V`, the AMI is given as::

        AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.

    Be mindful that this function is an order of magnitude slower than other
    metrics, such as the Adjusted Rand Index.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets.

    labels_pred : array, shape = [n_samples]
        A clustering of the data into disjoint subsets.

    average_method : string, optional (default: 'warn')
        How to compute the normalizer in the denominator. Possible options
        are 'min', 'geometric', 'arithmetic', and 'max'.
        If 'warn', 'max' will be used. The default will change to
        'arithmetic' in version 0.22.

        .. versionadded:: 0.20

    Returns
    -------
    ami: float (upperlimited by 1.0)
       The AMI returns a value of 1 when the two partitions are identical
       (ie perfectly matched). Random partitions (independent labellings) have
       an expected AMI around 0 on average hence can be negative.

    See also
    --------
    adjusted_rand_score: Adjusted Rand Index
    mutual_info_score: Mutual Information (not adjusted for chance)

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import adjusted_mutual_info_score
      >>> adjusted_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
      ... # doctest: +SKIP
      1.0
      >>> adjusted_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
      ... # doctest: +SKIP
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally in-complete, hence the AMI is null::

      >>> adjusted_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
      ... # doctest: +SKIP
      0.0

    References
    ----------
    .. [1] `Vinh, Epps, and Bailey, (2010). Information Theoretic Measures for
       Clusterings Comparison: Variants, Properties, Normalization and
       Correction for Chance, JMLR
       <http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf>`_

    .. [2] `Wikipedia entry for the Adjusted Mutual Information
       <https://en.wikipedia.org/wiki/Adjusted_Mutual_Information>`_

    "
  [labels_true labels_pred & {:keys [average_method]
                       :or {average_method "warn"}} ]
    (py/call-attr-kw supervised "adjusted_mutual_info_score" [labels_true labels_pred] {:average_method average_method }))

(defn adjusted-rand-score 
  "Rand index adjusted for chance.

    The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are
    assigned in the same or different clusters in the predicted and
    true clusterings.

    The raw RI score is then \"adjusted for chance\" into the ARI score
    using the following scheme::

        ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

    The adjusted Rand index is thus ensured to have a value close to
    0.0 for random labeling independently of the number of clusters and
    samples and exactly 1.0 when the clusterings are identical (up to
    a permutation).

    ARI is a symmetric measure::

        adjusted_rand_score(a, b) == adjusted_rand_score(b, a)

    Read more in the :ref:`User Guide <adjusted_rand_score>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    Returns
    -------
    ari : float
       Similarity score between -1.0 and 1.0. Random labelings have an ARI
       close to 0.0. 1.0 stands for perfect match.

    Examples
    --------

    Perfectly matching labelings have a score of 1 even

      >>> from sklearn.metrics.cluster import adjusted_rand_score
      >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Labelings that assign all classes members to the same clusters
    are complete be not always pure, hence penalized::

      >>> adjusted_rand_score([0, 0, 1, 2], [0, 0, 1, 1])  # doctest: +ELLIPSIS
      0.57...

    ARI is symmetric, so labelings that have pure clusters with members
    coming from the same classes but unnecessary splits are penalized::

      >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 2])  # doctest: +ELLIPSIS
      0.57...

    If classes members are completely split across different clusters, the
    assignment is totally incomplete, hence the ARI is very low::

      >>> adjusted_rand_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0

    References
    ----------

    .. [Hubert1985] L. Hubert and P. Arabie, Comparing Partitions,
      Journal of Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075

    .. [wk] https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index

    See also
    --------
    adjusted_mutual_info_score: Adjusted Mutual Information

    "
  [ labels_true labels_pred ]
  (py/call-attr supervised "adjusted_rand_score"  labels_true labels_pred ))

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
    (py/call-attr-kw supervised "check_array" [array] {:accept_sparse accept_sparse :accept_large_sparse accept_large_sparse :dtype dtype :order order :copy copy :force_all_finite force_all_finite :ensure_2d ensure_2d :allow_nd allow_nd :ensure_min_samples ensure_min_samples :ensure_min_features ensure_min_features :warn_on_dtype warn_on_dtype :estimator estimator }))

(defn check-clusterings 
  "Check that the labels arrays are 1D and of same dimension.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        The true labels

    labels_pred : int array, shape = [n_samples]
        The predicted labels
    "
  [ labels_true labels_pred ]
  (py/call-attr supervised "check_clusterings"  labels_true labels_pred ))

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
    (py/call-attr-kw supervised "comb" [N k] {:exact exact :repetition repetition }))

(defn completeness-score 
  "Completeness metric of a cluster labeling given a ground truth.

    A clustering result satisfies completeness if all the data points
    that are members of a given class are elements of the same cluster.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is not symmetric: switching ``label_true`` with ``label_pred``
    will return the :func:`homogeneity_score` which will be different in
    general.

    Read more in the :ref:`User Guide <homogeneity_completeness>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        cluster labels to evaluate

    Returns
    -------
    completeness : float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

    References
    ----------

    .. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
       conditional entropy-based external cluster evaluation measure
       <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

    See also
    --------
    homogeneity_score
    v_measure_score

    Examples
    --------

    Perfect labelings are complete::

      >>> from sklearn.metrics.cluster import completeness_score
      >>> completeness_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Non-perfect labelings that assign all classes members to the same clusters
    are still complete::

      >>> print(completeness_score([0, 0, 1, 1], [0, 0, 0, 0]))
      1.0
      >>> print(completeness_score([0, 1, 2, 3], [0, 0, 1, 1]))
      0.999...

    If classes members are split across different clusters, the
    assignment cannot be complete::

      >>> print(completeness_score([0, 0, 1, 1], [0, 1, 0, 1]))
      0.0
      >>> print(completeness_score([0, 0, 0, 0], [0, 1, 2, 3]))
      0.0

    "
  [ labels_true labels_pred ]
  (py/call-attr supervised "completeness_score"  labels_true labels_pred ))

(defn contingency-matrix 
  "Build a contingency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.

    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.

        .. versionadded:: 0.18

    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    "
  [labels_true labels_pred eps & {:keys [sparse]
                       :or {sparse false}} ]
    (py/call-attr-kw supervised "contingency_matrix" [labels_true labels_pred eps] {:sparse sparse }))

(defn entropy 
  "Calculates the entropy for a labeling.

    Parameters
    ----------
    labels : int array, shape = [n_samples]
        The labels
    "
  [ labels ]
  (py/call-attr supervised "entropy"  labels ))

(defn fowlkes-mallows-score 
  "Measure the similarity of two clusterings of a set of points.

    The Fowlkes-Mallows index (FMI) is defined as the geometric mean between of
    the precision and recall::

        FMI = TP / sqrt((TP + FP) * (TP + FN))

    Where ``TP`` is the number of **True Positive** (i.e. the number of pair of
    points that belongs in the same clusters in both ``labels_true`` and
    ``labels_pred``), ``FP`` is the number of **False Positive** (i.e. the
    number of pair of points that belongs in the same clusters in
    ``labels_true`` and not in ``labels_pred``) and ``FN`` is the number of
    **False Negative** (i.e the number of pair of points that belongs in the
    same clusters in ``labels_pred`` and not in ``labels_True``).

    The score ranges from 0 to 1. A high value indicates a good similarity
    between two clusters.

    Read more in the :ref:`User Guide <fowlkes_mallows_scores>`.

    Parameters
    ----------
    labels_true : int array, shape = (``n_samples``,)
        A clustering of the data into disjoint subsets.

    labels_pred : array, shape = (``n_samples``, )
        A clustering of the data into disjoint subsets.

    sparse : bool
        Compute contingency matrix internally with sparse matrix.

    Returns
    -------
    score : float
       The resulting Fowlkes-Mallows score.

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import fowlkes_mallows_score
      >>> fowlkes_mallows_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> fowlkes_mallows_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally random, hence the FMI is null::

      >>> fowlkes_mallows_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0

    References
    ----------
    .. [1] `E. B. Fowkles and C. L. Mallows, 1983. \"A method for comparing two
       hierarchical clusterings\". Journal of the American Statistical
       Association
       <http://wildfire.stat.ucla.edu/pdflibrary/fowlkes.pdf>`_

    .. [2] `Wikipedia entry for the Fowlkes-Mallows Index
           <https://en.wikipedia.org/wiki/Fowlkes-Mallows_index>`_
    "
  [labels_true labels_pred & {:keys [sparse]
                       :or {sparse false}} ]
    (py/call-attr-kw supervised "fowlkes_mallows_score" [labels_true labels_pred] {:sparse sparse }))

(defn homogeneity-completeness-v-measure 
  "Compute the homogeneity and completeness and V-Measure scores at once.

    Those metrics are based on normalized conditional entropy measures of
    the clustering labeling to evaluate given the knowledge of a Ground
    Truth class labels of the same samples.

    A clustering result satisfies homogeneity if all of its clusters
    contain only data points which are members of a single class.

    A clustering result satisfies completeness if all the data points
    that are members of a given class are elements of the same cluster.

    Both scores have positive values between 0.0 and 1.0, larger values
    being desirable.

    Those 3 metrics are independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score values in any way.

    V-Measure is furthermore symmetric: swapping ``labels_true`` and
    ``label_pred`` will give the same score. This does not hold for
    homogeneity and completeness. V-Measure is identical to
    :func:`normalized_mutual_info_score` with the arithmetic averaging
    method.

    Read more in the :ref:`User Guide <homogeneity_completeness>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        cluster labels to evaluate

    beta : float
        Ratio of weight attributed to ``homogeneity`` vs ``completeness``.
        If ``beta`` is greater than 1, ``completeness`` is weighted more
        strongly in the calculation. If ``beta`` is less than 1,
        ``homogeneity`` is weighted more strongly.

    Returns
    -------
    homogeneity : float
       score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling

    completeness : float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

    v_measure : float
        harmonic mean of the first two

    See also
    --------
    homogeneity_score
    completeness_score
    v_measure_score
    "
  [labels_true labels_pred & {:keys [beta]
                       :or {beta 1.0}} ]
    (py/call-attr-kw supervised "homogeneity_completeness_v_measure" [labels_true labels_pred] {:beta beta }))

(defn homogeneity-score 
  "Homogeneity metric of a cluster labeling given a ground truth.

    A clustering result satisfies homogeneity if all of its clusters
    contain only data points which are members of a single class.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is not symmetric: switching ``label_true`` with ``label_pred``
    will return the :func:`completeness_score` which will be different in
    general.

    Read more in the :ref:`User Guide <homogeneity_completeness>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        cluster labels to evaluate

    Returns
    -------
    homogeneity : float
       score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling

    References
    ----------

    .. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
       conditional entropy-based external cluster evaluation measure
       <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

    See also
    --------
    completeness_score
    v_measure_score

    Examples
    --------

    Perfect labelings are homogeneous::

      >>> from sklearn.metrics.cluster import homogeneity_score
      >>> homogeneity_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Non-perfect labelings that further split classes into more clusters can be
    perfectly homogeneous::

      >>> print(\"%.6f\" % homogeneity_score([0, 0, 1, 1], [0, 0, 1, 2]))
      ...                                                  # doctest: +ELLIPSIS
      1.000000
      >>> print(\"%.6f\" % homogeneity_score([0, 0, 1, 1], [0, 1, 2, 3]))
      ...                                                  # doctest: +ELLIPSIS
      1.000000

    Clusters that include samples from different classes do not make for an
    homogeneous labeling::

      >>> print(\"%.6f\" % homogeneity_score([0, 0, 1, 1], [0, 1, 0, 1]))
      ...                                                  # doctest: +ELLIPSIS
      0.0...
      >>> print(\"%.6f\" % homogeneity_score([0, 0, 1, 1], [0, 0, 0, 0]))
      ...                                                  # doctest: +ELLIPSIS
      0.0...

    "
  [ labels_true labels_pred ]
  (py/call-attr supervised "homogeneity_score"  labels_true labels_pred ))

(defn mutual-info-score 
  "Mutual Information between two clusterings.

    The Mutual Information is a measure of the similarity between two labels of
    the same data. Where :math:`|U_i|` is the number of the samples
    in cluster :math:`U_i` and :math:`|V_j|` is the number of the
    samples in cluster :math:`V_j`, the Mutual Information
    between clusterings :math:`U` and :math:`V` is given as:

    .. math::

        MI(U,V)=\sum_{i=1}^{|U|} \sum_{j=1}^{|V|} \frac{|U_i\cap V_j|}{N}
        \log\frac{N|U_i \cap V_j|}{|U_i||V_j|}

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets.

    labels_pred : array, shape = [n_samples]
        A clustering of the data into disjoint subsets.

    contingency : {None, array, sparse matrix},                   shape = [n_classes_true, n_classes_pred]
        A contingency matrix given by the :func:`contingency_matrix` function.
        If value is ``None``, it will be computed, otherwise the given value is
        used, with ``labels_true`` and ``labels_pred`` ignored.

    Returns
    -------
    mi : float
       Mutual information, a non-negative value

    See also
    --------
    adjusted_mutual_info_score: Adjusted against chance Mutual Information
    normalized_mutual_info_score: Normalized Mutual Information
    "
  [ labels_true labels_pred contingency ]
  (py/call-attr supervised "mutual_info_score"  labels_true labels_pred contingency ))

(defn normalized-mutual-info-score 
  "Normalized Mutual Information between two clusterings.

    Normalized Mutual Information (NMI) is a normalization of the Mutual
    Information (MI) score to scale the results between 0 (no mutual
    information) and 1 (perfect correlation). In this function, mutual
    information is normalized by some generalized mean of ``H(labels_true)``
    and ``H(labels_pred))``, defined by the `average_method`.

    This measure is not adjusted for chance. Therefore
    :func:`adjusted_mutual_info_score` might be preferred.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets.

    labels_pred : array, shape = [n_samples]
        A clustering of the data into disjoint subsets.

    average_method : string, optional (default: 'warn')
        How to compute the normalizer in the denominator. Possible options
        are 'min', 'geometric', 'arithmetic', and 'max'.
        If 'warn', 'geometric' will be used. The default will change to
        'arithmetic' in version 0.22.

        .. versionadded:: 0.20

    Returns
    -------
    nmi : float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

    See also
    --------
    v_measure_score: V-Measure (NMI with arithmetic mean option.)
    adjusted_rand_score: Adjusted Rand Index
    adjusted_mutual_info_score: Adjusted Mutual Information (adjusted
        against chance)

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import normalized_mutual_info_score
      >>> normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
      ... # doctest: +SKIP
      1.0
      >>> normalized_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
      ... # doctest: +SKIP
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally in-complete, hence the NMI is null::

      >>> normalized_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
      ... # doctest: +SKIP
      0.0

    "
  [labels_true labels_pred & {:keys [average_method]
                       :or {average_method "warn"}} ]
    (py/call-attr-kw supervised "normalized_mutual_info_score" [labels_true labels_pred] {:average_method average_method }))

(defn v-measure-score 
  "V-measure cluster labeling given a ground truth.

    This score is identical to :func:`normalized_mutual_info_score` with
    the ``'arithmetic'`` option for averaging.

    The V-measure is the harmonic mean between homogeneity and completeness::

        v = (1 + beta) * homogeneity * completeness
             / (beta * homogeneity + completeness)

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.


    Read more in the :ref:`User Guide <homogeneity_completeness>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        cluster labels to evaluate

    beta : float
        Ratio of weight attributed to ``homogeneity`` vs ``completeness``.
        If ``beta`` is greater than 1, ``completeness`` is weighted more
        strongly in the calculation. If ``beta`` is less than 1,
        ``homogeneity`` is weighted more strongly.

    Returns
    -------
    v_measure : float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

    References
    ----------

    .. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
       conditional entropy-based external cluster evaluation measure
       <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

    See also
    --------
    homogeneity_score
    completeness_score
    normalized_mutual_info_score

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have score 1.0::

      >>> from sklearn.metrics.cluster import v_measure_score
      >>> v_measure_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> v_measure_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Labelings that assign all classes members to the same clusters
    are complete be not homogeneous, hence penalized::

      >>> print(\"%.6f\" % v_measure_score([0, 0, 1, 2], [0, 0, 1, 1]))
      ...                                                  # doctest: +ELLIPSIS
      0.8...
      >>> print(\"%.6f\" % v_measure_score([0, 1, 2, 3], [0, 0, 1, 1]))
      ...                                                  # doctest: +ELLIPSIS
      0.66...

    Labelings that have pure clusters with members coming from the same
    classes are homogeneous but un-necessary splits harms completeness
    and thus penalize V-measure as well::

      >>> print(\"%.6f\" % v_measure_score([0, 0, 1, 1], [0, 0, 1, 2]))
      ...                                                  # doctest: +ELLIPSIS
      0.8...
      >>> print(\"%.6f\" % v_measure_score([0, 0, 1, 1], [0, 1, 2, 3]))
      ...                                                  # doctest: +ELLIPSIS
      0.66...

    If classes members are completely split across different clusters,
    the assignment is totally incomplete, hence the V-Measure is null::

      >>> print(\"%.6f\" % v_measure_score([0, 0, 0, 0], [0, 1, 2, 3]))
      ...                                                  # doctest: +ELLIPSIS
      0.0...

    Clusters that include samples from totally different classes totally
    destroy the homogeneity of the labeling, hence::

      >>> print(\"%.6f\" % v_measure_score([0, 0, 1, 1], [0, 0, 0, 0]))
      ...                                                  # doctest: +ELLIPSIS
      0.0...

    "
  [labels_true labels_pred & {:keys [beta]
                       :or {beta 1.0}} ]
    (py/call-attr-kw supervised "v_measure_score" [labels_true labels_pred] {:beta beta }))
