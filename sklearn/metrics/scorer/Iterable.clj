(ns sklearn.metrics.scorer.Iterable
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce scorer (import-module "sklearn.metrics.scorer"))

(defn Iterable 
  ""
  [  ]
  (py/call-attr scorer "Iterable"  ))
