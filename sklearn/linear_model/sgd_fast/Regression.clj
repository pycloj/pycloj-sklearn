(ns sklearn.linear-model.sgd-fast.Regression
  "Base class for loss functions for regression"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce sgd-fast (import-module "sklearn.linear_model.sgd_fast"))
