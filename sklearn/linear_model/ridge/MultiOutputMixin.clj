(ns sklearn.linear-model.ridge.MultiOutputMixin
  "Mixin to mark estimators that support multioutput."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce ridge (import-module "sklearn.linear_model.ridge"))

(defn MultiOutputMixin 
  "Mixin to mark estimators that support multioutput."
  [  ]
  (py/call-attr ridge "MultiOutputMixin"  ))
