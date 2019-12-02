(ns sklearn.utils.testing.TimeoutExpired
  "This exception is raised when the timeout expires while waiting for a
    child process.

    Attributes:
        cmd, output, stdout, stderr, timeout
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

(defn TimeoutExpired 
  "This exception is raised when the timeout expires while waiting for a
    child process.

    Attributes:
        cmd, output, stdout, stderr, timeout
    "
  [ cmd timeout output stderr ]
  (py/call-attr testing "TimeoutExpired"  cmd timeout output stderr ))

(defn stdout 
  ""
  [ self ]
    (py/call-attr self "stdout"))
