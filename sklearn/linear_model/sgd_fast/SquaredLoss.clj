(ns sklearn.linear-model.sgd-fast.SquaredLoss
  "Squared loss traditional used in linear regression."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce sgd-fast (import-module "sklearn.linear_model.sgd_fast"))
