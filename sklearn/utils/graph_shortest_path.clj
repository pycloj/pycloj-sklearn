(ns sklearn.utils.graph-shortest-path
  "
Routines for performing shortest-path graph searches

The main interface is in the function `graph_shortest_path`.  This
calls cython routines that compute the shortest path using either
the Floyd-Warshall algorithm, or Dykstra's algorithm with Fibonacci Heaps.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce graph-shortest-path (import-module "sklearn.utils.graph_shortest_path"))

(defn isspmatrix 
  "Is x of a sparse matrix type?

    Parameters
    ----------
    x
        object to check for being a sparse matrix

    Returns
    -------
    bool
        True if x is a sparse matrix, False otherwise

    Notes
    -----
    issparse and isspmatrix are aliases for the same function.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix, isspmatrix
    >>> isspmatrix(csr_matrix([[5]]))
    True

    >>> from scipy.sparse import isspmatrix
    >>> isspmatrix(5)
    False
    "
  [ x ]
  (py/call-attr graph-shortest-path "isspmatrix"  x ))

(defn isspmatrix-csr 
  "Is x of csr_matrix type?

    Parameters
    ----------
    x
        object to check for being a csr matrix

    Returns
    -------
    bool
        True if x is a csr matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import csr_matrix, isspmatrix_csr
    >>> isspmatrix_csr(csr_matrix([[5]]))
    True

    >>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc
    >>> isspmatrix_csr(csc_matrix([[5]]))
    False
    "
  [ x ]
  (py/call-attr graph-shortest-path "isspmatrix_csr"  x ))
