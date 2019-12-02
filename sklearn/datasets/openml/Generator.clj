(ns sklearn.datasets.openml.Generator
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce openml (import-module "sklearn.datasets.openml"))

(defn Generator 
  ""
  [  ]
  (py/call-attr openml "Generator"  ))

(defn close 
  "Raise GeneratorExit inside generator.
        "
  [ self  ]
  (py/call-attr self "close"  self  ))

(defn send 
  "Send a value into the generator.
        Return next yielded value or raise StopIteration.
        "
  [ self value ]
  (py/call-attr self "send"  self value ))

(defn throw 
  "Raise an exception in the generator.
        Return next yielded value or raise StopIteration.
        "
  [ self typ val tb ]
  (py/call-attr self "throw"  self typ val tb ))
