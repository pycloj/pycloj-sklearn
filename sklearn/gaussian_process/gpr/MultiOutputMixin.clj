(ns sklearn.gaussian-process.gpr.MultiOutputMixin
  "Mixin to mark estimators that support multioutput."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce gpr (import-module "sklearn.gaussian_process.gpr"))

(defn MultiOutputMixin 
  "Mixin to mark estimators that support multioutput."
  [  ]
  (py/call-attr gpr "MultiOutputMixin"  ))
