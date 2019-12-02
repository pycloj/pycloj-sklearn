(ns sklearn.gaussian-process.correlation-models
  "
The built-in correlation models submodule for the gaussian_process module.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce correlation-models (import-module "sklearn.gaussian_process.correlation_models"))

(defn absolute-exponential 
  "DEPRECATED: The function absolute_exponential of correlation_models is deprecated in version 0.19.1 and will be removed in 0.22.


    Absolute exponential autocorrelation model.
    (Ornstein-Uhlenbeck stochastic process)::

                                          n
        theta, d --> r(theta, d) = exp(  sum  - theta_i * |d_i| )
                                        i = 1

    Parameters
    ----------
    theta : array_like
        An array with shape 1 (isotropic) or n (anisotropic) giving the
        autocorrelation parameter(s).

    d : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.

    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) containing the values of the
        autocorrelation model.
    "
  [ theta d ]
  (py/call-attr correlation-models "absolute_exponential"  theta d ))

(defn cubic 
  "DEPRECATED: The function cubic of correlation_models is deprecated in version 0.19.1 and will be removed in 0.22.


    Cubic correlation model::

        theta, d --> r(theta, d) =
          n
         prod max(0, 1 - 3(theta_j*d_ij)^2 + 2(theta_j*d_ij)^3) ,  i = 1,...,m
        j = 1

    Parameters
    ----------
    theta : array_like
        An array with shape 1 (isotropic) or n (anisotropic) giving the
        autocorrelation parameter(s).

    d : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.

    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) with the values of the autocorrelation
        model.
    "
  [ theta d ]
  (py/call-attr correlation-models "cubic"  theta d ))

(defn generalized-exponential 
  "DEPRECATED: The function generalized_exponential of correlation_models is deprecated in version 0.19.1 and will be removed in 0.22.


    Generalized exponential correlation model.
    (Useful when one does not know the smoothness of the function to be
    predicted.)::

                                          n
        theta, d --> r(theta, d) = exp(  sum  - theta_i * |d_i|^p )
                                        i = 1

    Parameters
    ----------
    theta : array_like
        An array with shape 1+1 (isotropic) or n+1 (anisotropic) giving the
        autocorrelation parameter(s) (theta, p).

    d : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.

    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) with the values of the autocorrelation
        model.
    "
  [ theta d ]
  (py/call-attr correlation-models "generalized_exponential"  theta d ))

(defn linear 
  "DEPRECATED: The function linear of correlation_models is deprecated in version 0.19.1 and will be removed in 0.22.


    Linear correlation model::

        theta, d --> r(theta, d) =
              n
            prod max(0, 1 - theta_j*d_ij) ,  i = 1,...,m
            j = 1

    Parameters
    ----------
    theta : array_like
        An array with shape 1 (isotropic) or n (anisotropic) giving the
        autocorrelation parameter(s).

    d : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.

    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) with the values of the autocorrelation
        model.
    "
  [ theta d ]
  (py/call-attr correlation-models "linear"  theta d ))

(defn pure-nugget 
  "DEPRECATED: The function pure_nugget of correlation_models is deprecated in version 0.19.1 and will be removed in 0.22.


    Spatial independence correlation model (pure nugget).
    (Useful when one wants to solve an ordinary least squares problem!)::

                                           n
        theta, d --> r(theta, d) = 1 if   sum |d_i| == 0
                                         i = 1
                                   0 otherwise

    Parameters
    ----------
    theta : array_like
        None.

    d : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.

    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) with the values of the autocorrelation
        model.
    "
  [ theta d ]
  (py/call-attr correlation-models "pure_nugget"  theta d ))

(defn squared-exponential 
  "DEPRECATED: The function squared_exponential of correlation_models is deprecated in version 0.19.1 and will be removed in 0.22.


    Squared exponential correlation model (Radial Basis Function).
    (Infinitely differentiable stochastic process, very smooth)::

                                          n
        theta, d --> r(theta, d) = exp(  sum  - theta_i * (d_i)^2 )
                                        i = 1

    Parameters
    ----------
    theta : array_like
        An array with shape 1 (isotropic) or n (anisotropic) giving the
        autocorrelation parameter(s).

    d : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.

    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) containing the values of the
        autocorrelation model.
    "
  [ theta d ]
  (py/call-attr correlation-models "squared_exponential"  theta d ))
