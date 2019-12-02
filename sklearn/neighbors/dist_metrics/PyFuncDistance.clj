(ns sklearn.neighbors.dist-metrics.PyFuncDistance
  "PyFunc Distance

    A user-defined distance

    Parameters
    ----------
    func : function
        func should take two numpy arrays as input, and return a distance.
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
