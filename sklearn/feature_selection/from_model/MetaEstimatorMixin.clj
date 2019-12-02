(ns sklearn.feature-selection.from-model.MetaEstimatorMixin
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce from-model (import-module "sklearn.feature_selection.from_model"))

(defn MetaEstimatorMixin 
  ""
  [  ]
  (py/call-attr from-model "MetaEstimatorMixin"  ))
