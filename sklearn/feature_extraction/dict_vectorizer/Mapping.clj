(ns sklearn.feature-extraction.dict-vectorizer.Mapping
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce dict-vectorizer (import-module "sklearn.feature_extraction.dict_vectorizer"))

(defn Mapping 
  ""
  [  ]
  (py/call-attr dict-vectorizer "Mapping"  ))

(defn get 
  "D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None."
  [ self key default ]
  (py/call-attr self "get"  self key default ))

(defn items 
  "D.items() -> a set-like object providing a view on D's items"
  [ self  ]
  (py/call-attr self "items"  self  ))

(defn keys 
  "D.keys() -> a set-like object providing a view on D's keys"
  [ self  ]
  (py/call-attr self "keys"  self  ))

(defn values 
  "D.values() -> an object providing a view on D's values"
  [ self  ]
  (py/call-attr self "values"  self  ))
