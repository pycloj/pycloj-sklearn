(ns sklearn.tree.export.StringIO
  "Text I/O implementation using an in-memory buffer.

The initial_value argument sets the value of object.  The newline
argument is like the one of TextIOWrapper's constructor."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce export (import-module "sklearn.tree.export"))

(defn StringIO 
  "Text I/O implementation using an in-memory buffer.

The initial_value argument sets the value of object.  The newline
argument is like the one of TextIOWrapper's constructor."
  [ & {:keys [initial_value newline]
       :or {initial_value "" newline "
"}} ]
  
   (py/call-attr-kw export "StringIO" [] {:initial_value initial_value :newline newline }))
