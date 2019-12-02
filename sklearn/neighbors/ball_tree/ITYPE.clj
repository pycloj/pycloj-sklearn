(ns sklearn.neighbors.ball-tree.ITYPE
  "Signed integer type, compatible with Python `int` anc C ``long``.
    Character code: ``'l'``.
    Canonical name: ``np.int_``.
    Alias *on this platform*: ``np.int64``: 64-bit signed integer (-9223372036854775808 to 9223372036854775807).
    Alias *on this platform*: ``np.intp``: Signed integer large enough to fit pointer, compatible with C ``intptr_t``."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce ball-tree (import-module "sklearn.neighbors.ball_tree"))
