(ns sklearn.ensemble.partial-dependence.count
  "count(start=0, step=1) --> count object

Return a count object whose .__next__() method returns consecutive values.
Equivalent to:

    def count(firstval=0, step=1):
        x = firstval
        while 1:
            yield x
            x += step
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce partial-dependence (import-module "sklearn.ensemble.partial_dependence"))
