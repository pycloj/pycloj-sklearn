(ns sklearn.utils.sparsefuncs
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce sparsefuncs (import-module "sklearn.utils.sparsefuncs"))

(defn count-nonzero 
  "A variant of X.getnnz() with extension to weighting on axis 0

    Useful in efficiently calculating multilabel metrics.

    Parameters
    ----------
    X : CSR sparse matrix, shape = (n_samples, n_labels)
        Input data.

    axis : None, 0 or 1
        The axis on which the data is aggregated.

    sample_weight : array, shape = (n_samples,), optional
        Weight for each row of X.
    "
  [ X axis sample_weight ]
  (py/call-attr sparsefuncs "count_nonzero"  X axis sample_weight ))

(defn csc-median-axis-0 
  "Find the median across axis 0 of a CSC matrix.
    It is equivalent to doing np.median(X, axis=0).

    Parameters
    ----------
    X : CSC sparse matrix, shape (n_samples, n_features)
        Input data.

    Returns
    -------
    median : ndarray, shape (n_features,)
        Median.

    "
  [ X ]
  (py/call-attr sparsefuncs "csc_median_axis_0"  X ))

(defn incr-mean-variance-axis 
  "Compute incremental mean and variance along an axix on a CSR or
    CSC matrix.

    last_mean, last_var are the statistics computed at the last step by this
    function. Both must be initialized to 0-arrays of the proper size, i.e.
    the number of features in X. last_n is the number of samples encountered
    until now.

    Parameters
    ----------
    X : CSR or CSC sparse matrix, shape (n_samples, n_features)
        Input data.

    axis : int (either 0 or 1)
        Axis along which the axis should be computed.

    last_mean : float array with shape (n_features,)
        Array of feature-wise means to update with the new data X.

    last_var : float array with shape (n_features,)
        Array of feature-wise var to update with the new data X.

    last_n : int with shape (n_features,)
        Number of samples seen so far, excluded X.

    Returns
    -------

    means : float array with shape (n_features,)
        Updated feature-wise means.

    variances : float array with shape (n_features,)
        Updated feature-wise variances.

    n : int with shape (n_features,)
        Updated number of seen samples.

    Notes
    -----
    NaNs are ignored in the algorithm.

    "
  [ X axis last_mean last_var last_n ]
  (py/call-attr sparsefuncs "incr_mean_variance_axis"  X axis last_mean last_var last_n ))

(defn inplace-column-scale 
  "Inplace column scaling of a CSC/CSR matrix.

    Scale each feature of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : CSC or CSR matrix with shape (n_samples, n_features)
        Matrix to normalize using the variance of the features.

    scale : float array with shape (n_features,)
        Array of precomputed feature-wise values to use for scaling.
    "
  [ X scale ]
  (py/call-attr sparsefuncs "inplace_column_scale"  X scale ))

(defn inplace-csr-column-scale 
  "Inplace column scaling of a CSR matrix.

    Scale each feature of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : CSR matrix with shape (n_samples, n_features)
        Matrix to normalize using the variance of the features.

    scale : float array with shape (n_features,)
        Array of precomputed feature-wise values to use for scaling.
    "
  [ X scale ]
  (py/call-attr sparsefuncs "inplace_csr_column_scale"  X scale ))

(defn inplace-csr-row-scale 
  " Inplace row scaling of a CSR matrix.

    Scale each sample of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : CSR sparse matrix, shape (n_samples, n_features)
        Matrix to be scaled.

    scale : float array with shape (n_samples,)
        Array of precomputed sample-wise values to use for scaling.
    "
  [ X scale ]
  (py/call-attr sparsefuncs "inplace_csr_row_scale"  X scale ))

(defn inplace-row-scale 
  " Inplace row scaling of a CSR or CSC matrix.

    Scale each row of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : CSR or CSC sparse matrix, shape (n_samples, n_features)
        Matrix to be scaled.

    scale : float array with shape (n_features,)
        Array of precomputed sample-wise values to use for scaling.
    "
  [ X scale ]
  (py/call-attr sparsefuncs "inplace_row_scale"  X scale ))

(defn inplace-swap-column 
  "
    Swaps two columns of a CSC/CSR matrix in-place.

    Parameters
    ----------
    X : CSR or CSC sparse matrix, shape=(n_samples, n_features)
        Matrix whose two columns are to be swapped.

    m : int
        Index of the column of X to be swapped.

    n : int
        Index of the column of X to be swapped.
    "
  [ X m n ]
  (py/call-attr sparsefuncs "inplace_swap_column"  X m n ))

(defn inplace-swap-row 
  "
    Swaps two rows of a CSC/CSR matrix in-place.

    Parameters
    ----------
    X : CSR or CSC sparse matrix, shape=(n_samples, n_features)
        Matrix whose two rows are to be swapped.

    m : int
        Index of the row of X to be swapped.

    n : int
        Index of the row of X to be swapped.
    "
  [ X m n ]
  (py/call-attr sparsefuncs "inplace_swap_row"  X m n ))

(defn inplace-swap-row-csc 
  "
    Swaps two rows of a CSC matrix in-place.

    Parameters
    ----------
    X : scipy.sparse.csc_matrix, shape=(n_samples, n_features)
        Matrix whose two rows are to be swapped.

    m : int
        Index of the row of X to be swapped.

    n : int
        Index of the row of X to be swapped.
    "
  [ X m n ]
  (py/call-attr sparsefuncs "inplace_swap_row_csc"  X m n ))

(defn inplace-swap-row-csr 
  "
    Swaps two rows of a CSR matrix in-place.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix, shape=(n_samples, n_features)
        Matrix whose two rows are to be swapped.

    m : int
        Index of the row of X to be swapped.

    n : int
        Index of the row of X to be swapped.
    "
  [ X m n ]
  (py/call-attr sparsefuncs "inplace_swap_row_csr"  X m n ))

(defn mean-variance-axis 
  "Compute mean and variance along an axix on a CSR or CSC matrix

    Parameters
    ----------
    X : CSR or CSC sparse matrix, shape (n_samples, n_features)
        Input data.

    axis : int (either 0 or 1)
        Axis along which the axis should be computed.

    Returns
    -------

    means : float array with shape (n_features,)
        Feature-wise means

    variances : float array with shape (n_features,)
        Feature-wise variances

    "
  [ X axis ]
  (py/call-attr sparsefuncs "mean_variance_axis"  X axis ))

(defn min-max-axis 
  "Compute minimum and maximum along an axis on a CSR or CSC matrix and
    optionally ignore NaN values.

    Parameters
    ----------
    X : CSR or CSC sparse matrix, shape (n_samples, n_features)
        Input data.

    axis : int (either 0 or 1)
        Axis along which the axis should be computed.

    ignore_nan : bool, default is False
        Ignore or passing through NaN values.

        .. versionadded:: 0.20

    Returns
    -------

    mins : float array with shape (n_features,)
        Feature-wise minima

    maxs : float array with shape (n_features,)
        Feature-wise maxima
    "
  [X axis & {:keys [ignore_nan]
                       :or {ignore_nan false}} ]
    (py/call-attr-kw sparsefuncs "min_max_axis" [X axis] {:ignore_nan ignore_nan }))
