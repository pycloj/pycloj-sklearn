(ns sklearn.tree.export.Integral
  "Integral adds a conversion to int and the bit-string operations."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce export (import-module "sklearn.tree.export"))

(defn Integral 
  "Integral adds a conversion to int and the bit-string operations."
  [  ]
  (py/call-attr export "Integral"  ))

(defn conjugate 
  "Conjugate is a no-op for Reals."
  [ self  ]
  (py/call-attr self "conjugate"  self  ))

(defn denominator 
  "Integers have a denominator of 1."
  [ self ]
    (py/call-attr self "denominator"))

(defn imag 
  "Real numbers have no imaginary component."
  [ self ]
    (py/call-attr self "imag"))

(defn numerator 
  "Integers are their own numerators."
  [ self ]
    (py/call-attr self "numerator"))

(defn real 
  "Real numbers are their real component."
  [ self ]
    (py/call-attr self "real"))
