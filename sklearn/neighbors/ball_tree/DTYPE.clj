(ns sklearn.neighbors.ball-tree.DTYPE
  "Double-precision floating-point number type, compatible with Python `float`
    and C ``double``.
    Character code: ``'d'``.
    Canonical name: ``np.double``.
    Alias: ``np.float_``.
    Alias *on this platform*: ``np.float64``: 64-bit precision floating-point number type: sign bit, 11 bits exponent, 52 bits mantissa."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce ball-tree (import-module "sklearn.neighbors.ball_tree"))

(defn DTYPE 
  "Double-precision floating-point number type, compatible with Python `float`
    and C ``double``.
    Character code: ``'d'``.
    Canonical name: ``np.double``.
    Alias: ``np.float_``.
    Alias *on this platform*: ``np.float64``: 64-bit precision floating-point number type: sign bit, 11 bits exponent, 52 bits mantissa."
  [ & {:keys [x]
       :or {x 0}} ]
  
   (py/call-attr-kw ball-tree "DTYPE" [] {:x x }))
