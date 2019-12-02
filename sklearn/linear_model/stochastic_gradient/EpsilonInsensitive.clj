(ns sklearn.linear-model.stochastic-gradient.EpsilonInsensitive
  "Epsilon-Insensitive loss (used by SVR).

    loss = max(0, |y - p| - epsilon)
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce stochastic-gradient (import-module "sklearn.linear_model.stochastic_gradient"))
