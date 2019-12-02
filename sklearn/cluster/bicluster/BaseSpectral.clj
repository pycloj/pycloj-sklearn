(ns sklearn.cluster.bicluster.BaseSpectral
  "Base class for spectral biclustering."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce bicluster (import-module "sklearn.cluster.bicluster"))

(defn BaseSpectral 
  "Base class for spectral biclustering."
  [ & {:keys [n_clusters svd_method n_svd_vecs mini_batch init n_init n_jobs random_state]
       :or {n_clusters 3 svd_method "randomized" mini_batch false init "k-means++" n_init 10}} ]
  
   (py/call-attr-kw bicluster "BaseSpectral" [] {:n_clusters n_clusters :svd_method svd_method :n_svd_vecs n_svd_vecs :mini_batch mini_batch :init init :n_init n_init :n_jobs n_jobs :random_state random_state }))

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
