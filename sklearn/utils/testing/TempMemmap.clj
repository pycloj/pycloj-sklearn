(ns sklearn.utils.testing.TempMemmap
  "
    Parameters
    ----------
    data
    mmap_mode
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce testing (import-module "sklearn.utils.testing"))

(defn TempMemmap 
  "
    Parameters
    ----------
    data
    mmap_mode
    "
  [data & {:keys [mmap_mode]
                       :or {mmap_mode "r"}} ]
    (py/call-attr-kw testing "TempMemmap" [data] {:mmap_mode mmap_mode }))
