(ns sklearn.metrics.ranking.UndefinedMetricWarning
  "Warning used when the metric is invalid

    .. versionchanged:: 0.18
       Moved from sklearn.base.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce ranking (import-module "sklearn.metrics.ranking"))
