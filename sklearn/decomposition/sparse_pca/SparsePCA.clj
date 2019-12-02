(ns sklearn.decomposition.sparse-pca.SparsePCA
  "Sparse Principal Components Analysis (SparsePCA)

    Finds the set of sparse components that can optimally reconstruct
    the data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha.

    Read more in the :ref:`User Guide <SparsePCA>`.

    Parameters
    ----------
    n_components : int,
        Number of sparse atoms to extract.

    alpha : float,
        Sparsity controlling parameter. Higher values lead to sparser
        components.

    ridge_alpha : float,
        Amount of ridge shrinkage to apply in order to improve
        conditioning when calling the transform method.

    max_iter : int,
        Maximum number of iterations to perform.

    tol : float,
        Tolerance for the stopping condition.

    method : {'lars', 'cd'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    n_jobs : int or None, optional (default=None)
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    U_init : array of shape (n_samples, n_components),
        Initial values for the loadings for warm restart scenarios.

    V_init : array of shape (n_components, n_features),
        Initial values for the components for warm restart scenarios.

    verbose : int
        Controls the verbosity; the higher, the more messages. Defaults to 0.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    normalize_components : boolean, optional (default=False)
        - if False, use a version of Sparse PCA without components
          normalization and without data centering. This is likely a bug and
          even though it's the default for backward compatibility,
          this should not be used.
        - if True, use a version of Sparse PCA with components normalization
          and data centering.

        .. versionadded:: 0.20

        .. deprecated:: 0.22
           ``normalize_components`` was added and set to ``False`` for
           backward compatibility. It would be set to ``True`` from 0.22
           onwards.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Sparse components extracted from the data.

    error_ : array
        Vector of errors at each iteration.

    n_iter_ : int
        Number of iterations run.

    mean_ : array, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
        Equal to ``X.mean(axis=0)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.decomposition import SparsePCA
    >>> X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
    >>> transformer = SparsePCA(n_components=5,
    ...         normalize_components=True,
    ...         random_state=0)
    >>> transformer.fit(X) # doctest: +ELLIPSIS
    SparsePCA(...)
    >>> X_transformed = transformer.transform(X)
    >>> X_transformed.shape
    (200, 5)
    >>> # most values in the components_ are zero (sparsity)
    >>> np.mean(transformer.components_ == 0) # doctest: +ELLIPSIS
    0.9666...

    See also
    --------
    PCA
    MiniBatchSparsePCA
    DictionaryLearning
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce sparse-pca (import-module "sklearn.decomposition.sparse_pca"))

(defn SparsePCA 
  "Sparse Principal Components Analysis (SparsePCA)

    Finds the set of sparse components that can optimally reconstruct
    the data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha.

    Read more in the :ref:`User Guide <SparsePCA>`.

    Parameters
    ----------
    n_components : int,
        Number of sparse atoms to extract.

    alpha : float,
        Sparsity controlling parameter. Higher values lead to sparser
        components.

    ridge_alpha : float,
        Amount of ridge shrinkage to apply in order to improve
        conditioning when calling the transform method.

    max_iter : int,
        Maximum number of iterations to perform.

    tol : float,
        Tolerance for the stopping condition.

    method : {'lars', 'cd'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    n_jobs : int or None, optional (default=None)
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    U_init : array of shape (n_samples, n_components),
        Initial values for the loadings for warm restart scenarios.

    V_init : array of shape (n_components, n_features),
        Initial values for the components for warm restart scenarios.

    verbose : int
        Controls the verbosity; the higher, the more messages. Defaults to 0.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    normalize_components : boolean, optional (default=False)
        - if False, use a version of Sparse PCA without components
          normalization and without data centering. This is likely a bug and
          even though it's the default for backward compatibility,
          this should not be used.
        - if True, use a version of Sparse PCA with components normalization
          and data centering.

        .. versionadded:: 0.20

        .. deprecated:: 0.22
           ``normalize_components`` was added and set to ``False`` for
           backward compatibility. It would be set to ``True`` from 0.22
           onwards.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Sparse components extracted from the data.

    error_ : array
        Vector of errors at each iteration.

    n_iter_ : int
        Number of iterations run.

    mean_ : array, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
        Equal to ``X.mean(axis=0)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.decomposition import SparsePCA
    >>> X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
    >>> transformer = SparsePCA(n_components=5,
    ...         normalize_components=True,
    ...         random_state=0)
    >>> transformer.fit(X) # doctest: +ELLIPSIS
    SparsePCA(...)
    >>> X_transformed = transformer.transform(X)
    >>> X_transformed.shape
    (200, 5)
    >>> # most values in the components_ are zero (sparsity)
    >>> np.mean(transformer.components_ == 0) # doctest: +ELLIPSIS
    0.9666...

    See also
    --------
    PCA
    MiniBatchSparsePCA
    DictionaryLearning
    "
  [n_components & {:keys [alpha ridge_alpha max_iter tol method n_jobs U_init V_init verbose random_state normalize_components]
                       :or {alpha 1 ridge_alpha 0.01 max_iter 1000 tol 1e-08 method "lars" verbose false normalize_components false}} ]
    (py/call-attr-kw sparse-pca "SparsePCA" [n_components] {:alpha alpha :ridge_alpha ridge_alpha :max_iter max_iter :tol tol :method method :n_jobs n_jobs :U_init U_init :V_init V_init :verbose verbose :random_state random_state :normalize_components normalize_components }))

(defn fit 
  "Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
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
  "Least Squares projection of the data onto the sparse components.

        To avoid instability issues in case the system is under-determined,
        regularization can be applied (Ridge regression) via the
        `ridge_alpha` parameter.

        Note that Sparse PCA components orthogonality is not enforced as in PCA
        hence one cannot use a simple linear projection.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Test data to be transformed, must have the same number of
            features as the data used to train the model.

        Returns
        -------
        X_new array, shape (n_samples, n_components)
            Transformed data.
        "
  [ self X ]
  (py/call-attr self "transform"  self X ))
