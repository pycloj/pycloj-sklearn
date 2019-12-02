(ns sklearn.utils.graph-shortest-path.ITYPE
  "Signed integer type, compatible with C ``int``.
    Character code: ``'i'``.
    Canonical name: ``np.intc``.
    Alias *on this platform*: ``np.int32``: 32-bit signed integer (-2147483648 to 2147483647)."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce graph-shortest-path (import-module "sklearn.utils.graph_shortest_path"))
