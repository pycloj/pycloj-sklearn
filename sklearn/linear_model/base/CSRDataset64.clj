(ns sklearn.linear-model.base.CSRDataset64
  "A ``SequentialDataset`` backed by a scipy sparse CSR matrix. "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.linear_model.base"))
