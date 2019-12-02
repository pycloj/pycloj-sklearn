(ns sklearn.neighbors.ball-tree.BinaryTree
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce ball-tree (import-module "sklearn.neighbors.ball_tree"))
