(ns sklearn.preprocessing.data.combinations-w-r
  "combinations_with_replacement(iterable, r) --> combinations_with_replacement object

Return successive r-length combinations of elements in the iterable
allowing individual elements to have successive repeats.
combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data (import-module "sklearn.preprocessing.data"))
