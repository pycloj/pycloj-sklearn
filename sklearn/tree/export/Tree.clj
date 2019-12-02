(ns sklearn.tree.export.Tree
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce export (import-module "sklearn.tree.export"))

(defn Tree 
  ""
  [ & {:keys [label node_id]
       :or {label "" node_id -1}} ]
  
   (py/call-attr-kw export "Tree" [] {:label label :node_id node_id }))
