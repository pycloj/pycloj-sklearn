(ns sklearn.datasets.covtype.RemoteFileMetadata
  "RemoteFileMetadata(filename, url, checksum)"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce covtype (import-module "sklearn.datasets.covtype"))

(defn RemoteFileMetadata 
  "RemoteFileMetadata(filename, url, checksum)"
  [ filename url checksum ]
  (py/call-attr covtype "RemoteFileMetadata"  filename url checksum ))

(defn checksum 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "checksum"))

(defn filename 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "filename"))

(defn url 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "url"))
