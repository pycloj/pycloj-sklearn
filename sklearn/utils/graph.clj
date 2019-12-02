(ns sklearn.utils.graph
  "
Graph utilities and algorithms

Graphs are represented with their adjacency matrices, preferably using
sparse matrices.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce graph (import-module "sklearn.utils.graph"))

(defn single-source-shortest-path-length 
  "Return the shortest path length from source to all reachable nodes.

    Returns a dictionary of shortest path lengths keyed by target.

    Parameters
    ----------
    graph : sparse matrix or 2D array (preferably LIL matrix)
        Adjacency matrix of the graph
    source : integer
       Starting node for path
    cutoff : integer, optional
        Depth to stop the search - only
        paths of length <= cutoff are returned.

    Examples
    --------
    >>> from sklearn.utils.graph import single_source_shortest_path_length
    >>> import numpy as np
    >>> graph = np.array([[ 0, 1, 0, 0],
    ...                   [ 1, 0, 1, 0],
    ...                   [ 0, 1, 0, 1],
    ...                   [ 0, 0, 1, 0]])
    >>> list(sorted(single_source_shortest_path_length(graph, 0).items()))
    [(0, 0), (1, 1), (2, 2), (3, 3)]
    >>> graph = np.ones((6, 6))
    >>> list(sorted(single_source_shortest_path_length(graph, 2).items()))
    [(0, 1), (1, 1), (2, 0), (3, 1), (4, 1), (5, 1)]
    "
  [ graph source cutoff ]
  (py/call-attr graph "single_source_shortest_path_length"  graph source cutoff ))
