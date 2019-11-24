(ns sklearn.base.BiclusterMixin
  "Mixin class for all bicluster estimators in scikit-learn"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.base"))

(defn BiclusterMixin 
  "Mixin class for all bicluster estimators in scikit-learn"
  [  ]
  (py/call-attr base "BiclusterMixin"   ))

(defn biclusters- 
  "Convenient way to get row and column indicators together.

        Returns the ``rows_`` and ``columns_`` members.
        "
  [ self ]
    (py/call-attr base "biclusters_"  self))

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
  [self  & {:keys [i]} ]
    (py/call-attr-kw base "get_indices" [self] {:i i }))

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
  [self  & {:keys [i]} ]
    (py/call-attr-kw base "get_shape" [self] {:i i }))

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
  [self  & {:keys [i data]} ]
    (py/call-attr-kw base "get_submatrix" [self] {:i i :data data }))
