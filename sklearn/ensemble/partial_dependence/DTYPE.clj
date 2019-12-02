(ns sklearn.ensemble.partial-dependence.DTYPE
  "Single-precision floating-point number type, compatible with C ``float``.
    Character code: ``'f'``.
    Canonical name: ``np.single``.
    Alias *on this platform*: ``np.float32``: 32-bit-precision floating-point number type: sign bit, 8 bits exponent, 23 bits mantissa."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce partial-dependence (import-module "sklearn.ensemble.partial_dependence"))
