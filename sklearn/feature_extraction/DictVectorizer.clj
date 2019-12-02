(ns sklearn.feature-extraction.DictVectorizer
  "Transforms lists of feature-value mappings to vectors.

    This transformer turns lists of mappings (dict-like objects) of feature
    names to feature values into Numpy arrays or scipy.sparse matrices for use
    with scikit-learn estimators.

    When feature values are strings, this transformer will do a binary one-hot
    (aka one-of-K) coding: one boolean-valued feature is constructed for each
    of the possible string values that the feature can take on. For instance,
    a feature \"f\" that can take on the values \"ham\" and \"spam\" will become two
    features in the output, one signifying \"f=ham\", the other \"f=spam\".

    However, note that this transformer will only do a binary one-hot encoding
    when feature values are of type string. If categorical features are
    represented as numeric values such as int, the DictVectorizer can be
    followed by :class:`sklearn.preprocessing.OneHotEncoder` to complete
    binary one-hot encoding.

    Features that do not occur in a sample (mapping) will have a zero value
    in the resulting array/matrix.

    Read more in the :ref:`User Guide <dict_feature_extraction>`.

    Parameters
    ----------
    dtype : callable, optional
        The type of feature values. Passed to Numpy array/scipy.sparse matrix
        constructors as the dtype argument.
    separator : string, optional
        Separator string used when constructing new features for one-hot
        coding.
    sparse : boolean, optional.
        Whether transform should produce scipy.sparse matrices.
        True by default.
    sort : boolean, optional.
        Whether ``feature_names_`` and ``vocabulary_`` should be
        sorted when fitting. True by default.

    Attributes
    ----------
    vocabulary_ : dict
        A dictionary mapping feature names to feature indices.

    feature_names_ : list
        A list of length n_features containing the feature names (e.g., \"f=ham\"
        and \"f=spam\").

    Examples
    --------
    >>> from sklearn.feature_extraction import DictVectorizer
    >>> v = DictVectorizer(sparse=False)
    >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    >>> X = v.fit_transform(D)
    >>> X
    array([[2., 0., 1.],
           [0., 1., 3.]])
    >>> v.inverse_transform(X) ==         [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]
    True
    >>> v.transform({'foo': 4, 'unseen_feature': 3})
    array([[0., 0., 4.]])

    See also
    --------
    FeatureHasher : performs vectorization using only a hash function.
    sklearn.preprocessing.OrdinalEncoder : handles nominal/categorical
      features encoded as columns of arbitrary data types.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce feature-extraction (import-module "sklearn.feature_extraction"))

(defn DictVectorizer 
  "Transforms lists of feature-value mappings to vectors.

    This transformer turns lists of mappings (dict-like objects) of feature
    names to feature values into Numpy arrays or scipy.sparse matrices for use
    with scikit-learn estimators.

    When feature values are strings, this transformer will do a binary one-hot
    (aka one-of-K) coding: one boolean-valued feature is constructed for each
    of the possible string values that the feature can take on. For instance,
    a feature \"f\" that can take on the values \"ham\" and \"spam\" will become two
    features in the output, one signifying \"f=ham\", the other \"f=spam\".

    However, note that this transformer will only do a binary one-hot encoding
    when feature values are of type string. If categorical features are
    represented as numeric values such as int, the DictVectorizer can be
    followed by :class:`sklearn.preprocessing.OneHotEncoder` to complete
    binary one-hot encoding.

    Features that do not occur in a sample (mapping) will have a zero value
    in the resulting array/matrix.

    Read more in the :ref:`User Guide <dict_feature_extraction>`.

    Parameters
    ----------
    dtype : callable, optional
        The type of feature values. Passed to Numpy array/scipy.sparse matrix
        constructors as the dtype argument.
    separator : string, optional
        Separator string used when constructing new features for one-hot
        coding.
    sparse : boolean, optional.
        Whether transform should produce scipy.sparse matrices.
        True by default.
    sort : boolean, optional.
        Whether ``feature_names_`` and ``vocabulary_`` should be
        sorted when fitting. True by default.

    Attributes
    ----------
    vocabulary_ : dict
        A dictionary mapping feature names to feature indices.

    feature_names_ : list
        A list of length n_features containing the feature names (e.g., \"f=ham\"
        and \"f=spam\").

    Examples
    --------
    >>> from sklearn.feature_extraction import DictVectorizer
    >>> v = DictVectorizer(sparse=False)
    >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    >>> X = v.fit_transform(D)
    >>> X
    array([[2., 0., 1.],
           [0., 1., 3.]])
    >>> v.inverse_transform(X) ==         [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]
    True
    >>> v.transform({'foo': 4, 'unseen_feature': 3})
    array([[0., 0., 4.]])

    See also
    --------
    FeatureHasher : performs vectorization using only a hash function.
    sklearn.preprocessing.OrdinalEncoder : handles nominal/categorical
      features encoded as columns of arbitrary data types.
    "
  [ & {:keys [dtype separator sparse sort]
       :or {dtype <class 'numpy.float64'> separator "=" sparse true sort true}} ]
  
   (py/call-attr-kw feature-extraction "DictVectorizer" [] {:dtype dtype :separator separator :sparse sparse :sort sort }))

(defn fit 
  "Learn a list of feature name -> indices mappings.

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        y : (ignored)

        Returns
        -------
        self
        "
  [ self X y ]
  (py/call-attr self "fit"  self X y ))

(defn fit-transform 
  "Learn a list of feature name -> indices mappings and transform X.

        Like fit(X) followed by transform(X), but does not require
        materializing X in memory.

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        y : (ignored)

        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.
        "
  [ self X y ]
  (py/call-attr self "fit_transform"  self X y ))

(defn get-feature-names 
  "Returns a list of feature names, ordered by their indices.

        If one-of-K coding is applied to categorical features, this will
        include the constructed feature names but not the original ones.
        "
  [ self  ]
  (py/call-attr self "get_feature_names"  self  ))

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
  "Transform array or sparse matrix X back to feature mappings.

        X must have been produced by this DictVectorizer's transform or
        fit_transform method; it may only have passed through transformers
        that preserve the number of features and their order.

        In the case of one-hot/one-of-K coding, the constructed feature
        names and values are returned rather than the original ones.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Sample matrix.
        dict_type : callable, optional
            Constructor for feature mappings. Must conform to the
            collections.Mapping API.

        Returns
        -------
        D : list of dict_type objects, length = n_samples
            Feature mappings for the samples in X.
        "
  [self X & {:keys [dict_type]
                       :or {dict_type <class 'dict'>}} ]
    (py/call-attr-kw self "inverse_transform" [X] {:dict_type dict_type }))

(defn restrict 
  "Restrict the features to those in support using feature selection.

        This function modifies the estimator in-place.

        Parameters
        ----------
        support : array-like
            Boolean mask or list of indices (as returned by the get_support
            member of feature selectors).
        indices : boolean, optional
            Whether support is a list of indices.

        Returns
        -------
        self

        Examples
        --------
        >>> from sklearn.feature_extraction import DictVectorizer
        >>> from sklearn.feature_selection import SelectKBest, chi2
        >>> v = DictVectorizer()
        >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
        >>> X = v.fit_transform(D)
        >>> support = SelectKBest(chi2, k=2).fit(X, [0, 1])
        >>> v.get_feature_names()
        ['bar', 'baz', 'foo']
        >>> v.restrict(support.get_support()) # doctest: +ELLIPSIS
        ...                                   # doctest: +NORMALIZE_WHITESPACE
        DictVectorizer(dtype=..., separator='=', sort=True,
                sparse=True)
        >>> v.get_feature_names()
        ['bar', 'foo']
        "
  [self support & {:keys [indices]
                       :or {indices false}} ]
    (py/call-attr-kw self "restrict" [support] {:indices indices }))

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
  "Transform feature->value dicts to array or sparse matrix.

        Named features not encountered during fit or fit_transform will be
        silently ignored.

        Parameters
        ----------
        X : Mapping or iterable over Mappings, length = n_samples
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).

        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.
        "
  [ self X ]
  (py/call-attr self "transform"  self X ))
