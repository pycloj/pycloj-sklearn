(ns sklearn.linear-model.stochastic-gradient.Log
  "Logistic regression loss for binary classification with y in {-1, 1}"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce stochastic-gradient (import-module "sklearn.linear_model.stochastic_gradient"))
