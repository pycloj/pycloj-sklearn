(ns sklearn.gaussian-process.regression-models
  "
The built-in regression models submodule for the gaussian_process module.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce regression-models (import-module "sklearn.gaussian_process.regression_models"))

(defn constant 
  "DEPRECATED: The function constant of regression_models is deprecated in version 0.19.1 and will be removed in 0.22.


    Zero order polynomial (constant, p = 1) regression model.

    x --> f(x) = 1

    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    "
  [ x ]
  (py/call-attr regression-models "constant"  x ))

(defn linear 
  "DEPRECATED: The function linear of regression_models is deprecated in version 0.19.1 and will be removed in 0.22.


    First order polynomial (linear, p = n+1) regression model.

    x --> f(x) = [ 1, x_1, ..., x_n ].T

    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    "
  [ x ]
  (py/call-attr regression-models "linear"  x ))

(defn quadratic 
  "DEPRECATED: The function quadratic of regression_models is deprecated in version 0.19.1 and will be removed in 0.22.


    Second order polynomial (quadratic, p = n*(n-1)/2+n+1) regression model.

    x --> f(x) = [ 1, { x_i, i = 1,...,n }, { x_i * x_j,  (i,j) = 1,...,n } ].T
                                                          i > j

    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    "
  [ x ]
  (py/call-attr regression-models "quadratic"  x ))
