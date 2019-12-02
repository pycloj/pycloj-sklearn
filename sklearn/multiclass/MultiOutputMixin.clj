(ns sklearn.multiclass.MultiOutputMixin
  "Mixin to mark estimators that support multioutput."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce multiclass (import-module "sklearn.multiclass"))

(defn MultiOutputMixin 
  "Mixin to mark estimators that support multioutput."
  [  ]
  (py/call-attr multiclass "MultiOutputMixin"  ))
