(ns sklearn.linear-model.base.ArrayDataset64
  "Dataset backed by a two-dimensional numpy array.

    The dtype of the numpy array is expected to be ``np.float64`` (double)
    and C-style memory layout.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.linear_model.base"))
