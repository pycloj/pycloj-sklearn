(ns sklearn.linear-model.sag-fast.MultinomialLogLoss32
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce sag-fast (import-module "sklearn.linear_model.sag_fast"))
