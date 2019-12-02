(ns sklearn.utils.seq-dataset.SequentialDataset32
  "Base class for datasets with sequential data access.

    SequentialDataset is used to iterate over the rows of a matrix X and
    corresponding target values y, i.e. to iterate over samples.
    There are two methods to get the next sample:
        - next : Iterate sequentially (optionally randomized)
        - random : Iterate randomly (with replacement)

    Attributes
    ----------
    index : np.ndarray
        Index array for fast shuffling.

    index_data_ptr : int
        Pointer to the index array.

    current_index : int
        Index of current sample in ``index``.
        The index of current sample in the data is given by
        index_data_ptr[current_index].

    n_samples : Py_ssize_t
        Number of samples in the dataset.

    seed : np.uint32_t
        Seed used for random sampling.

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
