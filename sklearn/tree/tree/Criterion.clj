(ns sklearn.tree.tree.Criterion
  "Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is using
    different metrics.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tree (import-module "sklearn.tree.tree"))
