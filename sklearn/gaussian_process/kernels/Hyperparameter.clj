(ns sklearn.gaussian-process.kernels.Hyperparameter
  "A kernel hyperparameter's specification in form of a namedtuple.

    .. versionadded:: 0.18

    Attributes
    ----------
    name : string
        The name of the hyperparameter. Note that a kernel using a
        hyperparameter with name \"x\" must have the attributes self.x and
        self.x_bounds

    value_type : string
        The type of the hyperparameter. Currently, only \"numeric\"
        hyperparameters are supported.

    bounds : pair of floats >= 0 or \"fixed\"
        The lower and upper bound on the parameter. If n_elements>1, a pair
        of 1d array with n_elements each may be given alternatively. If
        the string \"fixed\" is passed as bounds, the hyperparameter's value
        cannot be changed.

    n_elements : int, default=1
        The number of elements of the hyperparameter value. Defaults to 1,
        which corresponds to a scalar hyperparameter. n_elements > 1
        corresponds to a hyperparameter which is vector-valued,
        such as, e.g., anisotropic length-scales.

    fixed : bool, default: None
        Whether the value of this hyperparameter is fixed, i.e., cannot be
        changed during hyperparameter tuning. If None is passed, the \"fixed\" is
        derived based on the given bounds.

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce kernels (import-module "sklearn.gaussian_process.kernels"))

(defn Hyperparameter 
  "A kernel hyperparameter's specification in form of a namedtuple.

    .. versionadded:: 0.18

    Attributes
    ----------
    name : string
        The name of the hyperparameter. Note that a kernel using a
        hyperparameter with name \"x\" must have the attributes self.x and
        self.x_bounds

    value_type : string
        The type of the hyperparameter. Currently, only \"numeric\"
        hyperparameters are supported.

    bounds : pair of floats >= 0 or \"fixed\"
        The lower and upper bound on the parameter. If n_elements>1, a pair
        of 1d array with n_elements each may be given alternatively. If
        the string \"fixed\" is passed as bounds, the hyperparameter's value
        cannot be changed.

    n_elements : int, default=1
        The number of elements of the hyperparameter value. Defaults to 1,
        which corresponds to a scalar hyperparameter. n_elements > 1
        corresponds to a hyperparameter which is vector-valued,
        such as, e.g., anisotropic length-scales.

    fixed : bool, default: None
        Whether the value of this hyperparameter is fixed, i.e., cannot be
        changed during hyperparameter tuning. If None is passed, the \"fixed\" is
        derived based on the given bounds.

    "
  [name value_type bounds & {:keys [n_elements fixed]
                       :or {n_elements 1}} ]
    (py/call-attr-kw kernels "Hyperparameter" [name value_type bounds] {:n_elements n_elements :fixed fixed }))

(defn bounds 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "bounds"))

(defn fixed 
  "Alias for field number 4"
  [ self ]
    (py/call-attr self "fixed"))

(defn n-elements 
  "Alias for field number 3"
  [ self ]
    (py/call-attr self "n_elements"))

(defn name 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "name"))

(defn value-type 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "value_type"))
