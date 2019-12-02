(ns sklearn.utils.testing.CalledProcessError
  "Raised when run() is called with check=True and the process
    returns a non-zero exit status.

    Attributes:
      cmd, returncode, stdout, stderr, output
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

(defn CalledProcessError 
  "Raised when run() is called with check=True and the process
    returns a non-zero exit status.

    Attributes:
      cmd, returncode, stdout, stderr, output
    "
  [ returncode cmd output stderr ]
  (py/call-attr testing "CalledProcessError"  returncode cmd output stderr ))

(defn stdout 
  "Alias for output attribute, to match stderr"
  [ self ]
    (py/call-attr self "stdout"))
