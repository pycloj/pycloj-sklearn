(ns sklearn.neighbors.dist-metrics.ChebyshevDistance
  "Chebyshev/Infinity Distance

    .. math::
       D(x, y) = max_i (|x_i - y_i|)
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce dist-metrics (import-module "sklearn.neighbors.dist_metrics"))
