(ns sklearn.utils.seq-dataset.ArrayDataset64
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
(defonce seq-dataset (import-module "sklearn.utils.seq_dataset"))
