(ns sklearn.linear-model.theil-sen.ConvergenceWarning
  "Custom warning to capture convergence problems

    .. versionchanged:: 0.18
       Moved from sklearn.utils.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce theil-sen (import-module "sklearn.linear_model.theil_sen"))
