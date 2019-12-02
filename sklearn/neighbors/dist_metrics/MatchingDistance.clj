(ns sklearn.neighbors.dist-metrics.MatchingDistance
  "Matching Distance

    Matching Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

    .. math::
       D(x, y) = \frac{N_{TF} + N_{FT}}{N}
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
