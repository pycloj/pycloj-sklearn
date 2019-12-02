(ns sklearn.linear-model.ransac.MetaEstimatorMixin
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce ransac (import-module "sklearn.linear_model.ransac"))

(defn MetaEstimatorMixin 
  ""
  [  ]
  (py/call-attr ransac "MetaEstimatorMixin"  ))
