(ns sklearn.tree.export.Sentinel
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce export (import-module "sklearn.tree.export"))

(defn Sentinel 
  ""
  [  ]
  (py/call-attr export "Sentinel"  ))
