(ns sklearn.feature-selection.rfe.MetaEstimatorMixin
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce rfe (import-module "sklearn.feature_selection.rfe"))

(defn MetaEstimatorMixin 
  ""
  [  ]
  (py/call-attr rfe "MetaEstimatorMixin"  ))
