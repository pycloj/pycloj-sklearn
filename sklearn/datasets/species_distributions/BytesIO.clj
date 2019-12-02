(ns sklearn.datasets.species-distributions.BytesIO
  "Buffered I/O implementation using an in-memory bytes buffer."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce species-distributions (import-module "sklearn.datasets.species_distributions"))

(defn BytesIO 
  "Buffered I/O implementation using an in-memory bytes buffer."
  [ & {:keys [initial_bytes]
       :or {initial_bytes b''}} ]
  
   (py/call-attr-kw species-distributions "BytesIO" [] {:initial_bytes initial_bytes }))
