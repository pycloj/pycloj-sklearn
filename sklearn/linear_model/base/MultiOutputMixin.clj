(ns sklearn.linear-model.base.MultiOutputMixin
  "Mixin to mark estimators that support multioutput."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.linear_model.base"))

(defn MultiOutputMixin 
  "Mixin to mark estimators that support multioutput."
  [  ]
  (py/call-attr base "MultiOutputMixin"  ))
