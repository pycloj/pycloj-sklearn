(ns sklearn.utils.stats
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce stats (import-module "sklearn.utils.stats"))

(defn stable-cumsum 
  "Use high precision for cumsum and check that final value matches sum

    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    axis : int, optional
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    "
  [arr axis & {:keys [rtol atol]
                       :or {rtol 1e-05 atol 1e-08}} ]
    (py/call-attr-kw stats "stable_cumsum" [arr axis] {:rtol rtol :atol atol }))
