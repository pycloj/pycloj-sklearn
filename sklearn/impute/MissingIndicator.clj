(ns sklearn.impute.MissingIndicator
  "Binary indicators for missing values.

    Note that this component typically should not be used in a vanilla
    :class:`Pipeline` consisting of transformers and a classifier, but rather
    could be added using a :class:`FeatureUnion` or :class:`ColumnTransformer`.

    Read more in the :ref:`User Guide <impute>`.

    Parameters
    ----------
    missing_values : number, string, np.nan (default) or None
        The placeholder for the missing values. All occurrences of
        `missing_values` will be indicated (True in the output array), the
        other values will be marked as False.

    features : str, optional
        Whether the imputer mask should represent all or a subset of
        features.

        - If \"missing-only\" (default), the imputer mask will only represent
          features containing missing values during fit time.
        - If \"all\", the imputer mask will represent all features.

    sparse : boolean or \"auto\", optional
        Whether the imputer mask format should be sparse or dense.

        - If \"auto\" (default), the imputer mask will be of same type as
          input.
        - If True, the imputer mask will be a sparse matrix.
        - If False, the imputer mask will be a numpy array.

    error_on_new : boolean, optional
        If True (default), transform will raise an error when there are
        features with missing values in transform that have no missing values
        in fit. This is applicable only when ``features=\"missing-only\"``.

    Attributes
    ----------
    features_ : ndarray, shape (n_missing_features,) or (n_features,)
        The features indices which will be returned when calling ``transform``.
        They are computed during ``fit``. For ``features='all'``, it is
        to ``range(n_features)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import MissingIndicator
    >>> X1 = np.array([[np.nan, 1, 3],
    ...                [4, 0, np.nan],
    ...                [8, 1, 0]])
    >>> X2 = np.array([[5, 1, np.nan],
    ...                [np.nan, 2, 3],
    ...                [2, 4, 0]])
    >>> indicator = MissingIndicator()
    >>> indicator.fit(X1)  # doctest: +NORMALIZE_WHITESPACE
    MissingIndicator(error_on_new=True, features='missing-only',
             missing_values=nan, sparse='auto')
    >>> X2_tr = indicator.transform(X2)
    >>> X2_tr
    array([[False,  True],
           [ True, False],
           [False, False]])

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

(defn MissingIndicator 
  "Binary indicators for missing values.

    Note that this component typically should not be used in a vanilla
    :class:`Pipeline` consisting of transformers and a classifier, but rather
    could be added using a :class:`FeatureUnion` or :class:`ColumnTransformer`.

    Read more in the :ref:`User Guide <impute>`.

    Parameters
    ----------
    missing_values : number, string, np.nan (default) or None
        The placeholder for the missing values. All occurrences of
        `missing_values` will be indicated (True in the output array), the
        other values will be marked as False.

    features : str, optional
        Whether the imputer mask should represent all or a subset of
        features.

        - If \"missing-only\" (default), the imputer mask will only represent
          features containing missing values during fit time.
        - If \"all\", the imputer mask will represent all features.

    sparse : boolean or \"auto\", optional
        Whether the imputer mask format should be sparse or dense.

        - If \"auto\" (default), the imputer mask will be of same type as
          input.
        - If True, the imputer mask will be a sparse matrix.
        - If False, the imputer mask will be a numpy array.

    error_on_new : boolean, optional
        If True (default), transform will raise an error when there are
        features with missing values in transform that have no missing values
        in fit. This is applicable only when ``features=\"missing-only\"``.

    Attributes
    ----------
    features_ : ndarray, shape (n_missing_features,) or (n_features,)
        The features indices which will be returned when calling ``transform``.
        They are computed during ``fit``. For ``features='all'``, it is
        to ``range(n_features)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import MissingIndicator
    >>> X1 = np.array([[np.nan, 1, 3],
    ...                [4, 0, np.nan],
    ...                [8, 1, 0]])
    >>> X2 = np.array([[5, 1, np.nan],
    ...                [np.nan, 2, 3],
    ...                [2, 4, 0]])
    >>> indicator = MissingIndicator()
    >>> indicator.fit(X1)  # doctest: +NORMALIZE_WHITESPACE
    MissingIndicator(error_on_new=True, features='missing-only',
             missing_values=nan, sparse='auto')
    >>> X2_tr = indicator.transform(X2)
    >>> X2_tr
    array([[False,  True],
           [ True, False],
           [False, False]])

    "
  [ & {:keys [missing_values features sparse error_on_new]
       :or {missing_values nan features "missing-only" sparse "auto" error_on_new true}} ]
  
   (py/call-attr-kw impute "MissingIndicator" [] {:missing_values missing_values :features features :sparse sparse :error_on_new error_on_new }))

(defn fit 
  "Fit the transformer on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns self.
        "
  [ self X y ]
  (py/call-attr self "fit"  self X y ))

(defn fit-transform 
  "Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.

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
  "Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.

        "
  [ self X ]
  (py/call-attr self "transform"  self X ))
