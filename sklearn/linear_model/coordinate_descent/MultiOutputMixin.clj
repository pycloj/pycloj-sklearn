(ns sklearn.linear-model.coordinate-descent.MultiOutputMixin
  "Mixin to mark estimators that support multioutput."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce coordinate-descent (import-module "sklearn.linear_model.coordinate_descent"))

(defn MultiOutputMixin 
  "Mixin to mark estimators that support multioutput."
  [  ]
  (py/call-attr coordinate-descent "MultiOutputMixin"  ))
