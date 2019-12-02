(ns sklearn.neighbors.dist-metrics.HaversineDistance
  "Haversine (Spherical) Distance

    The Haversine distance is the angular distance between two points on
    the surface of a sphere.  The first distance of each point is assumed
    to be the latitude, the second is the longitude, given in radians.
    The dimension of the points must be 2:

    .. math::
       D(x, y) = 2\arcsin[\sqrt{\sin^2((x1 - y1) / 2)
                                + \cos(x1)\cos(y1)\sin^2((x2 - y2) / 2)}]
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
