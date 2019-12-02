(ns sklearn.neighbors.base.KDTree
  "KDTree for fast generalized N-point problems

KDTree(X, leaf_size=40, metric='minkowski', \**kwargs)

Parameters
----------
X : array-like, shape = [n_samples, n_features]
    n_samples is the number of points in the data set, and
    n_features is the dimension of the parameter space.
    Note: if X is a C-contiguous array of doubles then data will
    not be copied. Otherwise, an internal copy will be made.

leaf_size : positive integer (default = 40)
    Number of points at which to switch to brute-force. Changing
    leaf_size will not affect the results of a query, but can
    significantly impact the speed of a query and the memory required
    to store the constructed tree.  The amount of memory needed to
    store the tree scales as approximately n_samples / leaf_size.
    For a specified ``leaf_size``, a leaf node is guaranteed to
    satisfy ``leaf_size <= n_points <= 2 * leaf_size``, except in
    the case that ``n_samples < leaf_size``.

metric : string or DistanceMetric object
    the distance metric to use for the tree.  Default='minkowski'
    with p=2 (that is, a euclidean metric). See the documentation
    of the DistanceMetric class for a list of available metrics.
    kd_tree.valid_metrics gives a list of the metrics which
    are valid for KDTree.

Additional keywords are passed to the distance metric class.

Attributes
----------
data : memory view
    The training data

Examples
--------
Query for k-nearest neighbors

    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
    >>> tree = KDTree(X, leaf_size=2)              # doctest: +SKIP
    >>> dist, ind = tree.query(X[:1], k=3)                # doctest: +SKIP
    >>> print(ind)  # indices of 3 closest neighbors
    [0 3 1]
    >>> print(dist)  # distances to 3 closest neighbors
    [ 0.          0.19662693  0.29473397]

Pickle and Unpickle a tree.  Note that the state of the tree is saved in the
pickle operation: the tree needs not be rebuilt upon unpickling.

    >>> import numpy as np
    >>> import pickle
    >>> rng = np.random.RandomState(0)
    >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
    >>> tree = KDTree(X, leaf_size=2)        # doctest: +SKIP
    >>> s = pickle.dumps(tree)                     # doctest: +SKIP
    >>> tree_copy = pickle.loads(s)                # doctest: +SKIP
    >>> dist, ind = tree_copy.query(X[:1], k=3)     # doctest: +SKIP
    >>> print(ind)  # indices of 3 closest neighbors
    [0 3 1]
    >>> print(dist)  # distances to 3 closest neighbors
    [ 0.          0.19662693  0.29473397]

Query for neighbors within a given radius

    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
    >>> tree = KDTree(X, leaf_size=2)     # doctest: +SKIP
    >>> print(tree.query_radius(X[:1], r=0.3, count_only=True))
    3
    >>> ind = tree.query_radius(X[:1], r=0.3)  # doctest: +SKIP
    >>> print(ind)  # indices of neighbors within distance 0.3
    [3 0 1]


Compute a gaussian kernel density estimate:

    >>> import numpy as np
    >>> rng = np.random.RandomState(42)
    >>> X = rng.random_sample((100, 3))
    >>> tree = KDTree(X)                # doctest: +SKIP
    >>> tree.kernel_density(X[:3], h=0.1, kernel='gaussian')
    array([ 6.94114649,  7.83281226,  7.2071716 ])

Compute a two-point auto-correlation function

    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> X = rng.random_sample((30, 3))
    >>> r = np.linspace(0, 1, 5)
    >>> tree = KDTree(X)                # doctest: +SKIP
    >>> tree.two_point_correlation(X, r)
    array([ 30,  62, 278, 580, 820])

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.neighbors.base"))
