(ns sklearn.impute.SimpleImputer
  "Imputation transformer for completing missing values.

    Read more in the :ref:`User Guide <impute>`.

    Parameters
    ----------
    missing_values : number, string, np.nan (default) or None
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.

    strategy : string, optional (default=\"mean\")
        The imputation strategy.

        - If \"mean\", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If \"median\", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If \"most_frequent\", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
        - If \"constant\", then replace missing values with fill_value. Can be
          used with strings or numeric data.

        .. versionadded:: 0.20
           strategy=\"constant\" for fixed value imputation.

    fill_value : string or numerical value, optional (default=None)
        When strategy == \"constant\", fill_value is used to replace all
        occurrences of missing_values.
        If left to the default, fill_value will be 0 when imputing numerical
        data and \"missing_value\" for strings or object data types.

    verbose : integer, optional (default=0)
        Controls the verbosity of the imputer.

    copy : boolean, optional (default=True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If X is not an array of floating values;
        - If X is encoded as a CSR matrix;
        - If add_indicator=True.

    add_indicator : boolean, optional (default=False)
        If True, a `MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/test time.

    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature.

    indicator_ : :class:`sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        ``None`` if add_indicator is False.

    See also
    --------
    IterativeImputer : Multivariate imputation of missing values.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import SimpleImputer
    >>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    ... # doctest: +NORMALIZE_WHITESPACE
    SimpleImputer(add_indicator=False, copy=True, fill_value=None,
            missing_values=nan, strategy='mean', verbose=0)
    >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> print(imp_mean.transform(X))
    ... # doctest: +NORMALIZE_WHITESPACE
    [[ 7.   2.   3. ]
     [ 4.   3.5  6. ]
     [10.   3.5  9. ]]

    Notes
    -----
    Columns which only contained missing values at `fit` are discarded upon
    `transform` if strategy is not \"constant\".

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce impute (import-module "sklearn.impute"))

(defn SimpleImputer 
  "Imputation transformer for completing missing values.

    Read more in the :ref:`User Guide <impute>`.

    Parameters
    ----------
    missing_values : number, string, np.nan (default) or None
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.

    strategy : string, optional (default=\"mean\")
        The imputation strategy.

        - If \"mean\", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If \"median\", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If \"most_frequent\", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
        - If \"constant\", then replace missing values with fill_value. Can be
          used with strings or numeric data.

        .. versionadded:: 0.20
           strategy=\"constant\" for fixed value imputation.

    fill_value : string or numerical value, optional (default=None)
        When strategy == \"constant\", fill_value is used to replace all
        occurrences of missing_values.
        If left to the default, fill_value will be 0 when imputing numerical
        data and \"missing_value\" for strings or object data types.

    verbose : integer, optional (default=0)
        Controls the verbosity of the imputer.

    copy : boolean, optional (default=True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If X is not an array of floating values;
        - If X is encoded as a CSR matrix;
        - If add_indicator=True.

    add_indicator : boolean, optional (default=False)
        If True, a `MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/test time.

    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature.

    indicator_ : :class:`sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        ``None`` if add_indicator is False.

    See also
    --------
    IterativeImputer : Multivariate imputation of missing values.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import SimpleImputer
    >>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    ... # doctest: +NORMALIZE_WHITESPACE
    SimpleImputer(add_indicator=False, copy=True, fill_value=None,
            missing_values=nan, strategy='mean', verbose=0)
    >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> print(imp_mean.transform(X))
    ... # doctest: +NORMALIZE_WHITESPACE
    [[ 7.   2.   3. ]
     [ 4.   3.5  6. ]
     [10.   3.5  9. ]]

    Notes
    -----
    Columns which only contained missing values at `fit` are discarded upon
    `transform` if strategy is not \"constant\".

    "
  [ & {:keys [missing_values strategy fill_value verbose copy add_indicator]
       :or {missing_values nan strategy "mean" verbose 0 copy true add_indicator false}} ]
  
   (py/call-attr-kw impute "SimpleImputer" [] {:missing_values missing_values :strategy strategy :fill_value fill_value :verbose verbose :copy copy :add_indicator add_indicator }))

(defn fit 
  "Fit the imputer on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : SimpleImputer
        "
  [ self X y ]
  (py/call-attr self "fit"  self X y ))

(defn fit-transform 
  "Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        "
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
  "Impute all missing values in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.
        "
  [ self X ]
  (py/call-attr self "transform"  self X ))
