(ns sklearn.preprocessing
  "
The :mod:`sklearn.preprocessing` module includes scaling, centering,
normalization, binarization and imputation methods.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce preprocessing (import-module "sklearn.preprocessing"))

(defn add-dummy-feature 
  "Augment dataset with an additional dummy feature.

    This is useful for fitting an intercept term with implementations which
    cannot otherwise fit it directly.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Data.

    value : float
        Value to use for the dummy feature.

    Returns
    -------

    X : {array, sparse matrix}, shape [n_samples, n_features + 1]
        Same data with dummy feature added as first column.

    Examples
    --------

    >>> from sklearn.preprocessing import add_dummy_feature
    >>> add_dummy_feature([[0, 1], [1, 0]])
    array([[1., 0., 1.],
           [1., 1., 0.]])
    "
  [X & {:keys [value]
                       :or {value 1.0}} ]
    (py/call-attr-kw preprocessing "add_dummy_feature" [X] {:value value }))

(defn binarize 
  "Boolean thresholding of array-like or scipy.sparse matrix

    Read more in the :ref:`User Guide <preprocessing_binarization>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        The data to binarize, element by element.
        scipy.sparse matrices should be in CSR or CSC format to avoid an
        un-necessary copy.

    threshold : float, optional (0.0 by default)
        Feature values below or equal to this are replaced by 0, above it by 1.
        Threshold may not be less than 0 for operations on sparse matrices.

    copy : boolean, optional, default True
        set to False to perform inplace binarization and avoid a copy
        (if the input is already a numpy array or a scipy.sparse CSR / CSC
        matrix and if axis is 1).

    See also
    --------
    Binarizer: Performs binarization using the ``Transformer`` API
        (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).
    "
  [X & {:keys [threshold copy]
                       :or {threshold 0.0 copy true}} ]
    (py/call-attr-kw preprocessing "binarize" [X] {:threshold threshold :copy copy }))

(defn label-binarize 
  "Binarize labels in a one-vs-all fashion

    Several regression and binary classification algorithms are
    available in scikit-learn. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    This function makes it possible to compute this transformation for a
    fixed set of class labels known ahead of time.

    Parameters
    ----------
    y : array-like
        Sequence of integer labels or multilabel data to encode.

    classes : array-like of shape [n_classes]
        Uniquely holds the label for each class.

    neg_label : int (default: 0)
        Value with which negative labels must be encoded.

    pos_label : int (default: 1)
        Value with which positive labels must be encoded.

    sparse_output : boolean (default: False),
        Set to true if output binary array is desired in CSR sparse format

    Returns
    -------
    Y : numpy array or CSR matrix of shape [n_samples, n_classes]
        Shape will be [n_samples, 1] for binary problems.

    Examples
    --------
    >>> from sklearn.preprocessing import label_binarize
    >>> label_binarize([1, 6], classes=[1, 2, 4, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    The class ordering is preserved:

    >>> label_binarize([1, 6], classes=[1, 6, 4, 2])
    array([[1, 0, 0, 0],
           [0, 1, 0, 0]])

    Binary targets transform to a column vector

    >>> label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])
    array([[1],
           [0],
           [0],
           [1]])

    See also
    --------
    LabelBinarizer : class used to wrap the functionality of label_binarize and
        allow for fitting to classes independently of the transform operation
    "
  [y classes & {:keys [neg_label pos_label sparse_output]
                       :or {neg_label 0 pos_label 1 sparse_output false}} ]
    (py/call-attr-kw preprocessing "label_binarize" [y classes] {:neg_label neg_label :pos_label pos_label :sparse_output sparse_output }))

(defn maxabs-scale 
  "Scale each feature to the [-1, 1] range without breaking the sparsity.

    This estimator scales each feature individually such
    that the maximal absolute value of each feature in the
    training set will be 1.0.

    This scaler can also be applied to sparse CSR or CSC matrices.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data.

    axis : int (0 by default)
        axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.

    copy : boolean, optional, default is True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a numpy array).

    See also
    --------
    MaxAbsScaler: Performs scaling to the [-1, 1] range using the``Transformer`` API
        (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).

    Notes
    -----
    NaNs are treated as missing values: disregarded to compute the statistics,
    and maintained during the data transformation.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    "
  [X & {:keys [axis copy]
                       :or {axis 0 copy true}} ]
    (py/call-attr-kw preprocessing "maxabs_scale" [X] {:axis axis :copy copy }))

(defn minmax-scale 
  "Transforms features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, i.e. between
    zero and one.

    The transformation is given by (when ``axis=0``)::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    The transformation is calculated as (when ``axis=0``)::

       X_scaled = scale * X + min - X.min(axis=0) * scale
       where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    .. versionadded:: 0.17
       *minmax_scale* function interface
       to :class:`sklearn.preprocessing.MinMaxScaler`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data.

    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    axis : int (0 by default)
        axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.

    copy : boolean, optional, default is True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a numpy array).

    See also
    --------
    MinMaxScaler: Performs scaling to a given range using the``Transformer`` API
        (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).

    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    "
  [X & {:keys [feature_range axis copy]
                       :or {feature_range (0, 1) axis 0 copy true}} ]
    (py/call-attr-kw preprocessing "minmax_scale" [X] {:feature_range feature_range :axis axis :copy copy }))

(defn normalize 
  "Scale input vectors individually to unit norm (vector length).

    Read more in the :ref:`User Guide <preprocessing_normalization>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        The data to normalize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.

    norm : 'l1', 'l2', or 'max', optional ('l2' by default)
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).

    axis : 0 or 1, optional (1 by default)
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.

    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).

    return_norm : boolean, default False
        whether to return the computed norms

    Returns
    -------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Normalized input X.

    norms : array, shape [n_samples] if axis=1 else [n_features]
        An array of norms along given axis for X.
        When X is sparse, a NotImplementedError will be raised
        for norm 'l1' or 'l2'.

    See also
    --------
    Normalizer: Performs normalization using the ``Transformer`` API
        (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).

    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    "
  [X & {:keys [norm axis copy return_norm]
                       :or {norm "l2" axis 1 copy true return_norm false}} ]
    (py/call-attr-kw preprocessing "normalize" [X] {:norm norm :axis axis :copy copy :return_norm return_norm }))

(defn power-transform 
  "
    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.

    Currently, power_transform supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.

    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to be transformed using a power transformation.

    method : str
        The power transform method. Available methods are:

        - 'yeo-johnson' [1]_, works with positive and negative values
        - 'box-cox' [2]_, only works with strictly positive values

        The default method will be changed from 'box-cox' to 'yeo-johnson'
        in version 0.23. To suppress the FutureWarning, explicitly set the
        parameter.

    standardize : boolean, default=True
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.

    copy : boolean, optional, default=True
        Set to False to perform inplace computation during transformation.

    Returns
    -------
    X_trans : array-like, shape (n_samples, n_features)
        The transformed data.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import power_transform
    >>> data = [[1, 2], [3, 2], [4, 5]]
    >>> print(power_transform(data, method='box-cox'))  # doctest: +ELLIPSIS
    [[-1.332... -0.707...]
     [ 0.256... -0.707...]
     [ 1.076...  1.414...]]

    See also
    --------
    PowerTransformer : Equivalent transformation with the
        ``Transformer`` API (e.g. as part of a preprocessing
        :class:`sklearn.pipeline.Pipeline`).

    quantile_transform : Maps data to a standard normal distribution with
        the parameter `output_distribution='normal'`.

    Notes
    -----
    NaNs are treated as missing values: disregarded in ``fit``, and maintained
    in ``transform``.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    References
    ----------

    .. [1] I.K. Yeo and R.A. Johnson, \"A new family of power transformations to
           improve normality or symmetry.\" Biometrika, 87(4), pp.954-959,
           (2000).

    .. [2] G.E.P. Box and D.R. Cox, \"An Analysis of Transformations\", Journal
           of the Royal Statistical Society B, 26, 211-252 (1964).
    "
  [X & {:keys [method standardize copy]
                       :or {method "warn" standardize true copy true}} ]
    (py/call-attr-kw preprocessing "power_transform" [X] {:method method :standardize standardize :copy copy }))

(defn quantile-transform 
  "Transform features using quantiles information.

    This method transforms the features to follow a uniform or a normal
    distribution. Therefore, for a given feature, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.

    The transformation is applied on each feature independently. First an
    estimate of the cumulative distribution function of a feature is
    used to map the original values to a uniform distribution. The obtained
    values are then mapped to the desired output distribution using the
    associated quantile function. Features values of new/unseen data that fall
    below or above the fitted range will be mapped to the bounds of the output
    distribution. Note that this transform is non-linear. It may distort linear
    correlations between variables measured at the same scale but renders
    variables measured at different scales more directly comparable.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    Parameters
    ----------
    X : array-like, sparse matrix
        The data to transform.

    axis : int, (default=0)
        Axis used to compute the means and standard deviations along. If 0,
        transform each feature, otherwise (if 1) transform each sample.

    n_quantiles : int, optional (default=1000 or n_samples)
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.
        If n_quantiles is larger than the number of samples, n_quantiles is set
        to the number of samples as a larger number of quantiles does not give
        a better approximation of the cumulative distribution function
        estimator.

    output_distribution : str, optional (default='uniform')
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.

    ignore_implicit_zeros : bool, optional (default=False)
        Only applies to sparse matrices. If True, the sparse entries of the
        matrix are discarded to compute the quantile statistics. If False,
        these entries are treated as zeros.

    subsample : int, optional (default=1e5)
        Maximum number of samples used to estimate the quantiles for
        computational efficiency. Note that the subsampling procedure may
        differ for value-identical sparse and dense matrices.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random. Note that this is used by subsampling and smoothing
        noise.

    copy : boolean, optional, (default=\"warn\")
        Set to False to perform inplace transformation and avoid a copy (if the
        input is already a numpy array). If True, a copy of `X` is transformed,
        leaving the original `X` unchanged

        .. deprecated:: 0.21
            The default value of parameter `copy` will be changed from False
            to True in 0.23. The current default of False is being changed to
            make it more consistent with the default `copy` values of other
            functions in :mod:`sklearn.preprocessing.data`. Furthermore, the
            current default of False may have unexpected side effects by
            modifying the value of `X` inplace

    Returns
    -------
    Xt : ndarray or sparse matrix, shape (n_samples, n_features)
        The transformed data.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import quantile_transform
    >>> rng = np.random.RandomState(0)
    >>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
    >>> quantile_transform(X, n_quantiles=10, random_state=0, copy=True)
    ... # doctest: +ELLIPSIS
    array([...])

    See also
    --------
    QuantileTransformer : Performs quantile-based scaling using the
        ``Transformer`` API (e.g. as part of a preprocessing
        :class:`sklearn.pipeline.Pipeline`).
    power_transform : Maps data to a normal distribution using a
        power transformation.
    scale : Performs standardization that is faster, but less robust
        to outliers.
    robust_scale : Performs robust standardization that removes the influence
        of outliers but does not put outliers and inliers on the same scale.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    "
  [X & {:keys [axis n_quantiles output_distribution ignore_implicit_zeros subsample random_state copy]
                       :or {axis 0 n_quantiles 1000 output_distribution "uniform" ignore_implicit_zeros false subsample 100000 copy "warn"}} ]
    (py/call-attr-kw preprocessing "quantile_transform" [X] {:axis axis :n_quantiles n_quantiles :output_distribution output_distribution :ignore_implicit_zeros ignore_implicit_zeros :subsample subsample :random_state random_state :copy copy }))

(defn robust-scale 
  "Standardize a dataset along any axis

    Center to the median and component wise scale
    according to the interquartile range.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    X : array-like
        The data to center and scale.

    axis : int (0 by default)
        axis used to compute the medians and IQR along. If 0,
        independently scale each feature, otherwise (if 1) scale
        each sample.

    with_centering : boolean, True by default
        If True, center the data before scaling.

    with_scaling : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
        Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR
        Quantile range used to calculate ``scale_``.

        .. versionadded:: 0.18

    copy : boolean, optional, default is True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).

    Notes
    -----
    This implementation will refuse to center scipy.sparse matrices
    since it would make them non-sparse and would potentially crash the
    program with memory exhaustion problems.

    Instead the caller is expected to either set explicitly
    `with_centering=False` (in that case, only variance scaling will be
    performed on the features of the CSR matrix) or to call `X.toarray()`
    if he/she expects the materialized dense array to fit in memory.

    To avoid memory copy the caller should pass a CSR matrix.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    See also
    --------
    RobustScaler: Performs centering and scaling using the ``Transformer`` API
        (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).
    "
  [X & {:keys [axis with_centering with_scaling quantile_range copy]
                       :or {axis 0 with_centering true with_scaling true quantile_range (25.0, 75.0) copy true}} ]
    (py/call-attr-kw preprocessing "robust_scale" [X] {:axis axis :with_centering with_centering :with_scaling with_scaling :quantile_range quantile_range :copy copy }))

(defn scale 
  "Standardize a dataset along any axis

    Center to the mean and component wise scale to unit variance.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        The data to center and scale.

    axis : int (0 by default)
        axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.

    with_mean : boolean, True by default
        If True, center the data before scaling.

    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSC matrix and if axis is 1).

    Notes
    -----
    This implementation will refuse to center scipy.sparse matrices
    since it would make them non-sparse and would potentially crash the
    program with memory exhaustion problems.

    Instead the caller is expected to either set explicitly
    `with_mean=False` (in that case, only variance scaling will be
    performed on the features of the CSC matrix) or to call `X.toarray()`
    if he/she expects the materialized dense array to fit in memory.

    To avoid memory copy the caller should pass a CSC matrix.

    NaNs are treated as missing values: disregarded to compute the statistics,
    and maintained during the data transformation.

    We use a biased estimator for the standard deviation, equivalent to
    `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
    affect model performance.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    See also
    --------
    StandardScaler: Performs scaling to unit variance using the``Transformer`` API
        (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).

    "
  [X & {:keys [axis with_mean with_std copy]
                       :or {axis 0 with_mean true with_std true copy true}} ]
    (py/call-attr-kw preprocessing "scale" [X] {:axis axis :with_mean with_mean :with_std with_std :copy copy }))
