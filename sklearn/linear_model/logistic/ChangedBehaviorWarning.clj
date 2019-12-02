(ns sklearn.linear-model.logistic.ChangedBehaviorWarning
  "Warning class used to notify the user of any change in the behavior.

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
(defonce logistic (import-module "sklearn.linear_model.logistic"))
