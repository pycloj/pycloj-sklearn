(ns sklearn.cluster.birch.AgglomerativeClustering
  "
    Agglomerative Clustering

    Recursively merges the pair of clusters that minimally increases
    a given linkage distance.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    n_clusters : int or None, optional (default=2)
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    affinity : string or callable, default: \"euclidean\"
        Metric used to compute the linkage. Can be \"euclidean\", \"l1\", \"l2\",
        \"manhattan\", \"cosine\", or \"precomputed\".
        If linkage is \"ward\", only \"euclidean\" is accepted.
        If \"precomputed\", a distance matrix (instead of a similarity matrix)
        is needed as input for the fit method.

    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    connectivity : array-like or callable, optional
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        kneighbors_graph. Default is None, i.e, the
        hierarchical clustering algorithm is unstructured.

    compute_full_tree : bool or 'auto' (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. This option is
        useful only when specifying a connectivity matrix. Note also that
        when varying the number of clusters and using caching, it may
        be advantageous to compute the full tree. It must be ``True`` if
        ``distance_threshold`` is not ``None``.

    linkage : {\"ward\", \"complete\", \"average\", \"single\"}, optional             (default=\"ward\")
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - ward minimizes the variance of the clusters being merged.
        - average uses the average of the distances of each observation of
          the two sets.
        - complete or maximum linkage uses the maximum distances between
          all observations of the two sets.
        - single uses the minimum of the distances between all observations
          of the two sets.

    pooling_func : callable, default='deprecated'
        Ignored.

        .. deprecated:: 0.20
            ``pooling_func`` has been deprecated in 0.20 and will be removed
            in 0.22.

    distance_threshold : float, optional (default=None)
        The linkage distance threshold above which, clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

        .. versionadded:: 0.21

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.

    labels_ : array [n_samples]
        cluster labels for each point

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        The estimated number of connected components in the graph.

    children_ : array-like, shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`

    Examples
    --------
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AgglomerativeClustering().fit(X)
    >>> clustering # doctest: +NORMALIZE_WHITESPACE
    AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                            connectivity=None, distance_threshold=None,
                            linkage='ward', memory=None, n_clusters=2,
                            pooling_func='deprecated')
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce birch (import-module "sklearn.cluster.birch"))

(defn AgglomerativeClustering 
  "
    Agglomerative Clustering

    Recursively merges the pair of clusters that minimally increases
    a given linkage distance.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    n_clusters : int or None, optional (default=2)
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    affinity : string or callable, default: \"euclidean\"
        Metric used to compute the linkage. Can be \"euclidean\", \"l1\", \"l2\",
        \"manhattan\", \"cosine\", or \"precomputed\".
        If linkage is \"ward\", only \"euclidean\" is accepted.
        If \"precomputed\", a distance matrix (instead of a similarity matrix)
        is needed as input for the fit method.

    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    connectivity : array-like or callable, optional
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        kneighbors_graph. Default is None, i.e, the
        hierarchical clustering algorithm is unstructured.

    compute_full_tree : bool or 'auto' (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. This option is
        useful only when specifying a connectivity matrix. Note also that
        when varying the number of clusters and using caching, it may
        be advantageous to compute the full tree. It must be ``True`` if
        ``distance_threshold`` is not ``None``.

    linkage : {\"ward\", \"complete\", \"average\", \"single\"}, optional             (default=\"ward\")
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - ward minimizes the variance of the clusters being merged.
        - average uses the average of the distances of each observation of
          the two sets.
        - complete or maximum linkage uses the maximum distances between
          all observations of the two sets.
        - single uses the minimum of the distances between all observations
          of the two sets.

    pooling_func : callable, default='deprecated'
        Ignored.

        .. deprecated:: 0.20
            ``pooling_func`` has been deprecated in 0.20 and will be removed
            in 0.22.

    distance_threshold : float, optional (default=None)
        The linkage distance threshold above which, clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

        .. versionadded:: 0.21

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.

    labels_ : array [n_samples]
        cluster labels for each point

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        The estimated number of connected components in the graph.

    children_ : array-like, shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`

    Examples
    --------
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AgglomerativeClustering().fit(X)
    >>> clustering # doctest: +NORMALIZE_WHITESPACE
    AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                            connectivity=None, distance_threshold=None,
                            linkage='ward', memory=None, n_clusters=2,
                            pooling_func='deprecated')
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])

    "
  [ & {:keys [n_clusters affinity memory connectivity compute_full_tree linkage pooling_func distance_threshold]
       :or {n_clusters 2 affinity "euclidean" compute_full_tree "auto" linkage "ward" pooling_func "deprecated"}} ]
  
   (py/call-attr-kw birch "AgglomerativeClustering" [] {:n_clusters n_clusters :affinity affinity :memory memory :connectivity connectivity :compute_full_tree compute_full_tree :linkage linkage :pooling_func pooling_func :distance_threshold distance_threshold }))

(defn fit 
  "Fit the hierarchical clustering on the data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data. Shape [n_samples, n_features], or [n_samples,
            n_samples] if affinity=='precomputed'.

        y : Ignored

        Returns
        -------
        self
        "
  [ self X y ]
  (py/call-attr self "fit"  self X y ))

(defn fit-predict 
  "Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : Ignored
            not used, present for API consistency by convention.

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            cluster labels
        "
  [ self X y ]
  (py/call-attr self "fit_predict"  self X y ))

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

(defn n-components- 
  ""
  [ self ]
    (py/call-attr self "n_components_"))

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
