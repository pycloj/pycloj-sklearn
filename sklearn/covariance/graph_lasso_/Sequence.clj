(ns sklearn.covariance.graph-lasso-.Sequence
  "All the operations on a read-only sequence.

    Concrete subclasses must override __new__ or __init__,
    __getitem__, and __len__.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce graph-lasso- (import-module "sklearn.covariance.graph_lasso_"))

(defn Sequence 
  "All the operations on a read-only sequence.

    Concrete subclasses must override __new__ or __init__,
    __getitem__, and __len__.
    "
  [  ]
  (py/call-attr graph-lasso- "Sequence"  ))

(defn count 
  "S.count(value) -> integer -- return number of occurrences of value"
  [ self value ]
  (py/call-attr self "count"  self value ))

(defn index 
  "S.index(value, [start, [stop]]) -> integer -- return first index of value.
           Raises ValueError if the value is not present.

           Supporting start and stop arguments is optional, but
           recommended.
        "
  [self value & {:keys [start stop]
                       :or {start 0}} ]
    (py/call-attr-kw self "index" [value] {:start start :stop stop }))
