(ns sklearn.gaussian-process.kernels.DotProduct
  "Dot-Product kernel.

    The DotProduct kernel is non-stationary and can be obtained from linear
    regression by putting N(0, 1) priors on the coefficients of x_d (d = 1, . .
    . , D) and a prior of N(0, \sigma_0^2) on the bias. The DotProduct kernel
    is invariant to a rotation of the coordinates about the origin, but not
    translations. It is parameterized by a parameter sigma_0^2. For
    sigma_0^2 =0, the kernel is called the homogeneous linear kernel, otherwise
    it is inhomogeneous. The kernel is given by

    k(x_i, x_j) = sigma_0 ^ 2 + x_i \cdot x_j

    The DotProduct kernel is commonly combined with exponentiation.

    .. versionadded:: 0.18

    Parameters
    ----------
    sigma_0 : float >= 0, default: 1.0
        Parameter controlling the inhomogenity of the kernel. If sigma_0=0,
        the kernel is homogenous.

    sigma_0_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on l

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce kernels (import-module "sklearn.gaussian_process.kernels"))

(defn DotProduct 
  "Dot-Product kernel.

    The DotProduct kernel is non-stationary and can be obtained from linear
    regression by putting N(0, 1) priors on the coefficients of x_d (d = 1, . .
    . , D) and a prior of N(0, \sigma_0^2) on the bias. The DotProduct kernel
    is invariant to a rotation of the coordinates about the origin, but not
    translations. It is parameterized by a parameter sigma_0^2. For
    sigma_0^2 =0, the kernel is called the homogeneous linear kernel, otherwise
    it is inhomogeneous. The kernel is given by

    k(x_i, x_j) = sigma_0 ^ 2 + x_i \cdot x_j

    The DotProduct kernel is commonly combined with exponentiation.

    .. versionadded:: 0.18

    Parameters
    ----------
    sigma_0 : float >= 0, default: 1.0
        Parameter controlling the inhomogenity of the kernel. If sigma_0=0,
        the kernel is homogenous.

    sigma_0_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on l

    "
  [ & {:keys [sigma_0 sigma_0_bounds]
       :or {sigma_0 1.0 sigma_0_bounds (1e-05, 100000.0)}} ]
  
   (py/call-attr-kw kernels "DotProduct" [] {:sigma_0 sigma_0 :sigma_0_bounds sigma_0_bounds }))

(defn bounds 
  "Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : array, shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        "
  [ self ]
    (py/call-attr self "bounds"))

(defn clone-with-theta 
  "Returns a clone of self with given hyperparameters theta.

        Parameters
        ----------
        theta : array, shape (n_dims,)
            The hyperparameters
        "
  [ self theta ]
  (py/call-attr self "clone_with_theta"  self theta ))

(defn diag 
  "Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        "
  [ self X ]
  (py/call-attr self "diag"  self X ))

(defn get-params 
  "Get parameters of this kernel.

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

(defn hyperparameter-sigma-0 
  ""
  [ self ]
    (py/call-attr self "hyperparameter_sigma_0"))

(defn hyperparameters 
  "Returns a list of all hyperparameter specifications."
  [ self ]
    (py/call-attr self "hyperparameters"))

(defn is-stationary 
  "Returns whether the kernel is stationary. "
  [ self  ]
  (py/call-attr self "is_stationary"  self  ))

(defn n-dims 
  "Returns the number of non-fixed hyperparameters of the kernel."
  [ self ]
    (py/call-attr self "n_dims"))

(defn set-params 
  "Set the parameters of this kernel.

        The method works on simple kernels as well as on nested kernels.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it's possible to update each component of a nested object.

        Returns
        -------
        self
        "
  [ self  ]
  (py/call-attr self "set_params"  self  ))

(defn theta 
  "Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        "
  [ self ]
    (py/call-attr self "theta"))
