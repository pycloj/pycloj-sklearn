(ns sklearn.linear-model.sgd-fast.EpsilonInsensitive
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
(defonce sgd-fast (import-module "sklearn.linear_model.sgd_fast"))
