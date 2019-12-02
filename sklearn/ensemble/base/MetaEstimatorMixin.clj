(ns sklearn.ensemble.base.MetaEstimatorMixin
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.ensemble.base"))

(defn MetaEstimatorMixin 
  ""
  [  ]
  (py/call-attr base "MetaEstimatorMixin"  ))
