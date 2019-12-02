(ns sklearn.preprocessing.PowerTransformer
  "Apply a power transform featurewise to make data more Gaussian-like.

    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.

    Currently, PowerTransformer supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.

    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    Parameters
    ----------
    method : str, (default='yeo-johnson')
        The power transform method. Available methods are:

        - 'yeo-johnson' [1]_, works with positive and negative values
        - 'box-cox' [2]_, only works with strictly positive values

    standardize : boolean, default=True
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.

    copy : boolean, optional, default=True
        Set to False to perform inplace computation during transformation.

    Attributes
    ----------
    lambdas_ : array of float, shape (n_features,)
        The parameters of the power transformation for the selected features.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import PowerTransformer
    >>> pt = PowerTransformer()
    >>> data = [[1, 2], [3, 2], [4, 5]]
    >>> print(pt.fit(data))
    PowerTransformer(copy=True, method='yeo-johnson', standardize=True)
    >>> print(pt.lambdas_)
    [ 1.386... -3.100...]
    >>> print(pt.transform(data))
    [[-1.316... -0.707...]
     [ 0.209... -0.707...]
     [ 1.106...  1.414...]]

    See also
    --------
    power_transform : Equivalent function without the estimator API.

    QuantileTransformer : Maps data to a standard normal distribution with
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
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce preprocessing (import-module "sklearn.preprocessing"))

(defn PowerTransformer 
  "Apply a power transform featurewise to make data more Gaussian-like.

    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.

    Currently, PowerTransformer supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.

    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    Parameters
    ----------
    method : str, (default='yeo-johnson')
        The power transform method. Available methods are:

        - 'yeo-johnson' [1]_, works with positive and negative values
        - 'box-cox' [2]_, only works with strictly positive values

    standardize : boolean, default=True
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.

    copy : boolean, optional, default=True
        Set to False to perform inplace computation during transformation.

    Attributes
    ----------
    lambdas_ : array of float, shape (n_features,)
        The parameters of the power transformation for the selected features.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import PowerTransformer
    >>> pt = PowerTransformer()
    >>> data = [[1, 2], [3, 2], [4, 5]]
    >>> print(pt.fit(data))
    PowerTransformer(copy=True, method='yeo-johnson', standardize=True)
    >>> print(pt.lambdas_)
    [ 1.386... -3.100...]
    >>> print(pt.transform(data))
    [[-1.316... -0.707...]
     [ 0.209... -0.707...]
     [ 1.106...  1.414...]]

    See also
    --------
    power_transform : Equivalent function without the estimator API.

    QuantileTransformer : Maps data to a standard normal distribution with
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
  [ & {:keys [method standardize copy]
       :or {method "yeo-johnson" standardize true copy true}} ]
  
   (py/call-attr-kw preprocessing "PowerTransformer" [] {:method method :standardize standardize :copy copy }))

(defn fit 
  "Estimate the optimal parameter lambda for each feature.

        The optimal lambda parameter for minimizing skewness is estimated on
        each feature independently using maximum likelihood.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data used to estimate the optimal transformation parameters.

        y : Ignored

        Returns
        -------
        self : object
        "
  [ self X y ]
  (py/call-attr self "fit"  self X y ))

(defn fit-transform 
  ""
  [ self X y ]
  (py/call-attr self "fit_transform"  self X y ))

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

(defn inverse-transform 
  "Apply the inverse power transformation using the fitted lambdas.

        The inverse of the Box-Cox transformation is given by::

            if lambda == 0:
                X = exp(X_trans)
            else:
                X = (X_trans * lambda + 1) ** (1 / lambda)

        The inverse of the Yeo-Johnson transformation is given by::

            if X >= 0 and lambda == 0:
                X = exp(X_trans) - 1
            elif X >= 0 and lambda != 0:
                X = (X_trans * lambda + 1) ** (1 / lambda) - 1
            elif X < 0 and lambda != 2:
                X = 1 - (-(2 - lambda) * X_trans + 1) ** (1 / (2 - lambda))
            elif X < 0 and lambda == 2:
                X = 1 - exp(-X_trans)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The transformed data.

        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            The original data
        "
  [ self X ]
  (py/call-attr self "inverse_transform"  self X ))

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

(defn transform 
  "Apply the power transform to each feature using the fitted lambdas.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to be transformed using a power transformation.

        Returns
        -------
        X_trans : array-like, shape (n_samples, n_features)
            The transformed data.
        "
  [ self X ]
  (py/call-attr self "transform"  self X ))
