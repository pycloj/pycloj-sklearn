(ns sklearn.metrics.ranking.partial
  "partial(func, *args, **keywords) - new function with partial application
    of the given arguments and keywords.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce ranking (import-module "sklearn.metrics.ranking"))
