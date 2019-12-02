(ns sklearn.datasets.samples-generator.Iterable
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce samples-generator (import-module "sklearn.datasets.samples_generator"))

(defn Iterable 
  ""
  [  ]
  (py/call-attr samples-generator "Iterable"  ))
