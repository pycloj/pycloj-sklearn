(ns sklearn.utils.fast-dict.IntFloatDict
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce fast-dict (import-module "sklearn.utils.fast_dict"))
