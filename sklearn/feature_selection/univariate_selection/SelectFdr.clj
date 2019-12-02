(ns sklearn.feature-selection.univariate-selection.SelectFdr
  "Filter: Select the p-values for an estimated false discovery rate

    This uses the Benjamini-Hochberg procedure. ``alpha`` is an upper bound
    on the expected false discovery rate.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues).
        Default is f_classif (see below \"See also\"). The default function only
        works with classification tasks.

    alpha : float, optional
        The highest uncorrected p-value for features to keep.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import SelectFdr, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> X_new = SelectFdr(chi2, alpha=0.01).fit_transform(X, y)
    >>> X_new.shape
    (569, 16)

    Attributes
    ----------
    scores_ : array-like, shape=(n_features,)
        Scores of features.

    pvalues_ : array-like, shape=(n_features,)
        p-values of feature scores.

    References
    ----------
    https://en.wikipedia.org/wiki/False_discovery_rate

    See also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    mutual_info_regression: Mutual information for a contnuous target.
    SelectPercentile: Select features based on percentile of the highest scores.
    SelectKBest: Select features based on the k highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFwe: Select features based on family-wise error rate.
    GenericUnivariateSelect: Univariate feature selector with configurable mode.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce univariate-selection (import-module "sklearn.feature_selection.univariate_selection"))

(defn SelectFdr 
  "Filter: Select the p-values for an estimated false discovery rate

    This uses the Benjamini-Hochberg procedure. ``alpha`` is an upper bound
    on the expected false discovery rate.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues).
        Default is f_classif (see below \"See also\"). The default function only
        works with classification tasks.

    alpha : float, optional
        The highest uncorrected p-value for features to keep.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import SelectFdr, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> X_new = SelectFdr(chi2, alpha=0.01).fit_transform(X, y)
    >>> X_new.shape
    (569, 16)

    Attributes
    ----------
    scores_ : array-like, shape=(n_features,)
        Scores of features.

    pvalues_ : array-like, shape=(n_features,)
        p-values of feature scores.

    References
    ----------
    https://en.wikipedia.org/wiki/False_discovery_rate

    See also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    mutual_info_regression: Mutual information for a contnuous target.
    SelectPercentile: Select features based on percentile of the highest scores.
    SelectKBest: Select features based on the k highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFwe: Select features based on family-wise error rate.
    GenericUnivariateSelect: Univariate feature selector with configurable mode.
    "
  [ & {:keys [score_func alpha]
       :or {score_func <function f_classif at 0x1a2252ab90> alpha 0.05}} ]
  
   (py/call-attr-kw univariate-selection "SelectFdr" [] {:score_func score_func :alpha alpha }))

(defn fit 
  "Run score function on (X, y) and get the appropriate features.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
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

(defn get-support 
  "
        Get a mask, or integer index, of the features selected

        Parameters
        ----------
        indices : boolean (default False)
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        "
  [self  & {:keys [indices]
                       :or {indices false}} ]
    (py/call-attr-kw self "get_support" [] {:indices indices }))

(defn inverse-transform 
  "
        Reverse the transformation operation

        Parameters
        ----------
        X : array of shape [n_samples, n_selected_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_original_features]
            `X` with columns of zeros inserted where features would have
            been removed by `transform`.
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
  "Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        "
  [ self X ]
  (py/call-attr self "transform"  self X ))
