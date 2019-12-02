(ns sklearn.cross-decomposition.pls-.PLSSVD
  "Partial Least Square SVD

    Simply perform a svd on the crosscovariance matrix: X'Y
    There are no iterative deflation here.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    Parameters
    ----------
    n_components : int, default 2
        Number of components to keep.

    scale : boolean, default True
        Whether to scale X and Y.

    copy : boolean, default True
        Whether to copy X and Y, or perform in-place computations.

    Attributes
    ----------
    x_weights_ : array, [p, n_components]
        X block weights vectors.

    y_weights_ : array, [q, n_components]
        Y block weights vectors.

    x_scores_ : array, [n_samples, n_components]
        X scores.

    y_scores_ : array, [n_samples, n_components]
        Y scores.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cross_decomposition import PLSSVD
    >>> X = np.array([[0., 0., 1.],
    ...     [1.,0.,0.],
    ...     [2.,2.,2.],
    ...     [2.,5.,4.]])
    >>> Y = np.array([[0.1, -0.2],
    ...     [0.9, 1.1],
    ...     [6.2, 5.9],
    ...     [11.9, 12.3]])
    >>> plsca = PLSSVD(n_components=2)
    >>> plsca.fit(X, Y)
    PLSSVD(copy=True, n_components=2, scale=True)
    >>> X_c, Y_c = plsca.transform(X, Y)
    >>> X_c.shape, Y_c.shape
    ((4, 2), (4, 2))

    See also
    --------
    PLSCanonical
    CCA
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce pls- (import-module "sklearn.cross_decomposition.pls_"))

(defn PLSSVD 
  "Partial Least Square SVD

    Simply perform a svd on the crosscovariance matrix: X'Y
    There are no iterative deflation here.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    Parameters
    ----------
    n_components : int, default 2
        Number of components to keep.

    scale : boolean, default True
        Whether to scale X and Y.

    copy : boolean, default True
        Whether to copy X and Y, or perform in-place computations.

    Attributes
    ----------
    x_weights_ : array, [p, n_components]
        X block weights vectors.

    y_weights_ : array, [q, n_components]
        Y block weights vectors.

    x_scores_ : array, [n_samples, n_components]
        X scores.

    y_scores_ : array, [n_samples, n_components]
        Y scores.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cross_decomposition import PLSSVD
    >>> X = np.array([[0., 0., 1.],
    ...     [1.,0.,0.],
    ...     [2.,2.,2.],
    ...     [2.,5.,4.]])
    >>> Y = np.array([[0.1, -0.2],
    ...     [0.9, 1.1],
    ...     [6.2, 5.9],
    ...     [11.9, 12.3]])
    >>> plsca = PLSSVD(n_components=2)
    >>> plsca.fit(X, Y)
    PLSSVD(copy=True, n_components=2, scale=True)
    >>> X_c, Y_c = plsca.transform(X, Y)
    >>> X_c.shape, Y_c.shape
    ((4, 2), (4, 2))

    See also
    --------
    PLSCanonical
    CCA
    "
  [ & {:keys [n_components scale copy]
       :or {n_components 2 scale true copy true}} ]
  
   (py/call-attr-kw pls- "PLSSVD" [] {:n_components n_components :scale scale :copy copy }))

(defn fit 
  "Fit model to data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like, shape = [n_samples, n_targets]
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        "
  [ self X Y ]
  (py/call-attr self "fit"  self X Y ))

(defn fit-transform 
  "Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        y : array-like, shape = [n_samples, n_targets]
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
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
  "
        Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like, shape = [n_samples, n_targets]
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        "
  [ self X Y ]
  (py/call-attr self "transform"  self X Y ))
