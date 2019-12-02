(ns sklearn.gaussian-process.kernels.StationaryKernelMixin
  "Mixin for kernels which are stationary: k(X, Y)= f(X-Y).

    .. versionadded:: 0.18
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

(defn StationaryKernelMixin 
  "Mixin for kernels which are stationary: k(X, Y)= f(X-Y).

    .. versionadded:: 0.18
    "
  [  ]
  (py/call-attr kernels "StationaryKernelMixin"  ))

(defn is-stationary 
  "Returns whether the kernel is stationary. "
  [ self  ]
  (py/call-attr self "is_stationary"  self  ))
