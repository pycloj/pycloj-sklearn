(ns sklearn.neighbors.dist-metrics.WMinkowskiDistance
  "Weighted Minkowski Distance

    .. math::
       D(x, y) = [\sum_i |w_i * (x_i - y_i)|^p] ^ (1/p)

    Weighted Minkowski Distance requires p >= 1 and finite.

    Parameters
    ----------
    p : int
        The order of the norm of the difference :math:`{||u-v||}_p`.
    w : (N,) array_like
        The weight vector.

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce dist-metrics (import-module "sklearn.neighbors.dist_metrics"))
