(ns sklearn.utils.class-weight
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce class-weight (import-module "sklearn.utils.class_weight"))

(defn compute-class-weight 
  "Estimate class weights for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, 'balanced' or None
        If 'balanced', class weights will be given by
        ``n_samples / (n_classes * np.bincount(y))``.
        If a dictionary is given, keys are classes and values
        are corresponding class weights.
        If None is given, the class weights will be uniform.

    classes : ndarray
        Array of the classes occurring in the data, as given by
        ``np.unique(y_org)`` with ``y_org`` the original class labels.

    y : array-like, shape (n_samples,)
        Array of original class labels per sample;

    Returns
    -------
    class_weight_vect : ndarray, shape (n_classes,)
        Array with class_weight_vect[i] the weight for i-th class

    References
    ----------
    The \"balanced\" heuristic is inspired by
    Logistic Regression in Rare Events Data, King, Zen, 2001.
    "
  [ class_weight classes y ]
  (py/call-attr class-weight "compute_class_weight"  class_weight classes y ))

(defn compute-sample-weight 
  "Estimate sample weights by class for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, list of dicts, \"balanced\", or None, optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The \"balanced\" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data:
        ``n_samples / (n_classes * np.bincount(y))``.

        For multi-output, the weights of each column of y will be multiplied.

    y : array-like, shape = [n_samples] or [n_samples, n_outputs]
        Array of original class labels per sample.

    indices : array-like, shape (n_subsample,), or None
        Array of indices to be used in a subsample. Can be of length less than
        n_samples in the case of a subsample, or equal to n_samples in the
        case of a bootstrap subsample with repeated indices. If None, the
        sample weight will be calculated over the full sample. Only \"balanced\"
        is supported for class_weight if this is provided.

    Returns
    -------
    sample_weight_vect : ndarray, shape (n_samples,)
        Array with sample weights as applied to the original y
    "
  [ class_weight y indices ]
  (py/call-attr class-weight "compute_sample_weight"  class_weight y indices ))
