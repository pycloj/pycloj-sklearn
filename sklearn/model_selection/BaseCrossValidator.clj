(ns sklearn.model-selection.BaseCrossValidator
  "Base class for all cross-validators

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce model-selection (import-module "sklearn.model_selection"))

(defn BaseCrossValidator 
  "Base class for all cross-validators

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    "
  [  ]
  (py/call-attr model-selection "BaseCrossValidator"  ))

(defn get-n-splits 
  "Returns the number of splitting iterations in the cross-validator"
  [ self X y groups ]
  (py/call-attr self "get_n_splits"  self X y groups ))

(defn split 
  "Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, of length n_samples
            The target variable for supervised learning problems.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        "
  [ self X y groups ]
  (py/call-attr self "split"  self X y groups ))
