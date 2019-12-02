(ns sklearn.covariance.LedoitWolf
  "LedoitWolf Estimator

    Ledoit-Wolf is a particular form of shrinkage, where the shrinkage
    coefficient is computed using O. Ledoit and M. Wolf's formula as
    described in \"A Well-Conditioned Estimator for Large-Dimensional
    Covariance Matrices\", Ledoit and Wolf, Journal of Multivariate
    Analysis, Volume 88, Issue 2, February 2004, pages 365-411.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data will be centered before computation.

    block_size : int, default=1000
        Size of the blocks into which the covariance matrix will be split
        during its Ledoit-Wolf estimation. This is purely a memory
        optimization and does not affect results.

    Attributes
    ----------
    location_ : array-like, shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    shrinkage_ : float, 0 <= shrinkage <= 1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import LedoitWolf
    >>> real_cov = np.array([[.4, .2],
    ...                      [.2, .8]])
    >>> np.random.seed(0)
    >>> X = np.random.multivariate_normal(mean=[0, 0],
    ...                                   cov=real_cov,
    ...                                   size=50)
    >>> cov = LedoitWolf().fit(X)
    >>> cov.covariance_ # doctest: +ELLIPSIS
    array([[0.4406..., 0.1616...],
           [0.1616..., 0.8022...]])
    >>> cov.location_
    array([ 0.0595... , -0.0075...])

    Notes
    -----
    The regularised covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features
    and shrinkage is given by the Ledoit and Wolf formula (see References)

    References
    ----------
    \"A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices\",
    Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2,
    February 2004, pages 365-411.

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce covariance (import-module "sklearn.covariance"))

(defn LedoitWolf 
  "LedoitWolf Estimator

    Ledoit-Wolf is a particular form of shrinkage, where the shrinkage
    coefficient is computed using O. Ledoit and M. Wolf's formula as
    described in \"A Well-Conditioned Estimator for Large-Dimensional
    Covariance Matrices\", Ledoit and Wolf, Journal of Multivariate
    Analysis, Volume 88, Issue 2, February 2004, pages 365-411.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data will be centered before computation.

    block_size : int, default=1000
        Size of the blocks into which the covariance matrix will be split
        during its Ledoit-Wolf estimation. This is purely a memory
        optimization and does not affect results.

    Attributes
    ----------
    location_ : array-like, shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    shrinkage_ : float, 0 <= shrinkage <= 1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import LedoitWolf
    >>> real_cov = np.array([[.4, .2],
    ...                      [.2, .8]])
    >>> np.random.seed(0)
    >>> X = np.random.multivariate_normal(mean=[0, 0],
    ...                                   cov=real_cov,
    ...                                   size=50)
    >>> cov = LedoitWolf().fit(X)
    >>> cov.covariance_ # doctest: +ELLIPSIS
    array([[0.4406..., 0.1616...],
           [0.1616..., 0.8022...]])
    >>> cov.location_
    array([ 0.0595... , -0.0075...])

    Notes
    -----
    The regularised covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features
    and shrinkage is given by the Ledoit and Wolf formula (see References)

    References
    ----------
    \"A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices\",
    Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2,
    February 2004, pages 365-411.

    "
  [ & {:keys [store_precision assume_centered block_size]
       :or {store_precision true assume_centered false block_size 1000}} ]
  
   (py/call-attr-kw covariance "LedoitWolf" [] {:store_precision store_precision :assume_centered assume_centered :block_size block_size }))

(defn error-norm 
  "Computes the Mean Squared Error between two covariance estimators.
        (In the sense of the Frobenius norm).

        Parameters
        ----------
        comp_cov : array-like, shape = [n_features, n_features]
            The covariance to compare with.

        norm : str
            The type of norm used to compute the error. Available error types:
            - 'frobenius' (default): sqrt(tr(A^t.A))
            - 'spectral': sqrt(max(eigenvalues(A^t.A))
            where A is the error ``(comp_cov - self.covariance_)``.

        scaling : bool
            If True (default), the squared error norm is divided by n_features.
            If False, the squared error norm is not rescaled.

        squared : bool
            Whether to compute the squared error norm or the error norm.
            If True (default), the squared error norm is returned.
            If False, the error norm is returned.

        Returns
        -------
        The Mean Squared Error (in the sense of the Frobenius norm) between
        `self` and `comp_cov` covariance estimators.

        "
  [self comp_cov & {:keys [norm scaling squared]
                       :or {norm "frobenius" scaling true squared true}} ]
    (py/call-attr-kw self "error_norm" [comp_cov] {:norm norm :scaling scaling :squared squared }))

(defn fit 
  " Fits the Ledoit-Wolf shrunk covariance model
        according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y
            not used, present for API consistence purpose.

        Returns
        -------
        self : object

        "
  [ self X y ]
  (py/call-attr self "fit"  self X y ))

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

(defn get-precision 
  "Getter for the precision matrix.

        Returns
        -------
        precision_ : array-like
            The precision matrix associated to the current covariance object.

        "
  [ self  ]
  (py/call-attr self "get_precision"  self  ))

(defn mahalanobis 
  "Computes the squared Mahalanobis distances of given observations.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The observations, the Mahalanobis distances of the which we
            compute. Observations are assumed to be drawn from the same
            distribution than the data used in fit.

        Returns
        -------
        dist : array, shape = [n_samples,]
            Squared Mahalanobis distances of the observations.

        "
  [ self X ]
  (py/call-attr self "mahalanobis"  self X ))

(defn score 
  "Computes the log-likelihood of a Gaussian data set with
        `self.covariance_` as an estimator of its covariance matrix.

        Parameters
        ----------
        X_test : array-like, shape = [n_samples, n_features]
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X_test is assumed to be drawn from the same distribution than
            the data used in fit (including centering).

        y
            not used, present for API consistence purpose.

        Returns
        -------
        res : float
            The likelihood of the data set with `self.covariance_` as an
            estimator of its covariance matrix.

        "
  [ self X_test y ]
  (py/call-attr self "score"  self X_test y ))

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
