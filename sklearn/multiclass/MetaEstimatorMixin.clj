(ns sklearn.multiclass.MetaEstimatorMixin
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce multiclass (import-module "sklearn.multiclass"))

(defn MetaEstimatorMixin 
  ""
  [  ]
  (py/call-attr multiclass "MetaEstimatorMixin"  ))
