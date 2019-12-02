(ns sklearn.gaussian-process.kernels.RationalQuadratic
  "Rational Quadratic kernel.

    The RationalQuadratic kernel can be seen as a scale mixture (an infinite
    sum) of RBF kernels with different characteristic length-scales. It is
    parameterized by a length-scale parameter length_scale>0 and a scale
    mixture parameter alpha>0. Only the isotropic variant where length_scale is
    a scalar is supported at the moment. The kernel given by:

    k(x_i, x_j) = (1 + d(x_i, x_j)^2 / (2*alpha * length_scale^2))^-alpha

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float > 0, default: 1.0
        The length scale of the kernel.

    alpha : float > 0, default: 1.0
        Scale mixture parameter

    length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale

    alpha_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on alpha

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

(defn RationalQuadratic 
  "Rational Quadratic kernel.

    The RationalQuadratic kernel can be seen as a scale mixture (an infinite
    sum) of RBF kernels with different characteristic length-scales. It is
    parameterized by a length-scale parameter length_scale>0 and a scale
    mixture parameter alpha>0. Only the isotropic variant where length_scale is
    a scalar is supported at the moment. The kernel given by:

    k(x_i, x_j) = (1 + d(x_i, x_j)^2 / (2*alpha * length_scale^2))^-alpha

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float > 0, default: 1.0
        The length scale of the kernel.

    alpha : float > 0, default: 1.0
        Scale mixture parameter

    length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale

    alpha_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on alpha

    "
  [ & {:keys [length_scale alpha length_scale_bounds alpha_bounds]
       :or {length_scale 1.0 alpha 1.0 length_scale_bounds (1e-05, 100000.0) alpha_bounds (1e-05, 100000.0)}} ]
  
   (py/call-attr-kw kernels "RationalQuadratic" [] {:length_scale length_scale :alpha alpha :length_scale_bounds length_scale_bounds :alpha_bounds alpha_bounds }))

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

(defn hyperparameter-alpha 
  ""
  [ self ]
    (py/call-attr self "hyperparameter_alpha"))

(defn hyperparameter-length-scale 
  ""
  [ self ]
    (py/call-attr self "hyperparameter_length_scale"))

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
