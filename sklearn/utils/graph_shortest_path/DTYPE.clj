(ns sklearn.utils.graph-shortest-path.DTYPE
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
(defonce graph-shortest-path (import-module "sklearn.utils.graph_shortest_path"))

(defn DTYPE 
  "Double-precision floating-point number type, compatible with Python `float`
    and C ``double``.
    Character code: ``'d'``.
    Canonical name: ``np.double``.
    Alias: ``np.float_``.
    Alias *on this platform*: ``np.float64``: 64-bit precision floating-point number type: sign bit, 11 bits exponent, 52 bits mantissa."
  [ & {:keys [x]
       :or {x 0}} ]
  
   (py/call-attr-kw graph-shortest-path "DTYPE" [] {:x x }))