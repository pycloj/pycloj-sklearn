(ns sklearn.datasets.openml.Request
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce openml (import-module "sklearn.datasets.openml"))

(defn Request 
  ""
  [url data & {:keys [headers origin_req_host unverifiable method]
                       :or {headers {} unverifiable false}} ]
    (py/call-attr-kw openml "Request" [url data] {:headers headers :origin_req_host origin_req_host :unverifiable unverifiable :method method }))

(defn add-header 
  ""
  [ self key val ]
  (py/call-attr self "add_header"  self key val ))

(defn add-unredirected-header 
  ""
  [ self key val ]
  (py/call-attr self "add_unredirected_header"  self key val ))

(defn data 
  ""
  [ self ]
    (py/call-attr self "data"))

(defn full-url 
  ""
  [ self ]
    (py/call-attr self "full_url"))

(defn get-full-url 
  ""
  [ self  ]
  (py/call-attr self "get_full_url"  self  ))

(defn get-header 
  ""
  [ self header_name default ]
  (py/call-attr self "get_header"  self header_name default ))

(defn get-method 
  "Return a string indicating the HTTP request method."
  [ self  ]
  (py/call-attr self "get_method"  self  ))

(defn has-header 
  ""
  [ self header_name ]
  (py/call-attr self "has_header"  self header_name ))

(defn has-proxy 
  ""
  [ self  ]
  (py/call-attr self "has_proxy"  self  ))

(defn header-items 
  ""
  [ self  ]
  (py/call-attr self "header_items"  self  ))

(defn remove-header 
  ""
  [ self header_name ]
  (py/call-attr self "remove_header"  self header_name ))

(defn set-proxy 
  ""
  [ self host type ]
  (py/call-attr self "set_proxy"  self host type ))
