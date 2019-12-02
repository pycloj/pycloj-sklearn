(ns sklearn.neighbors.ball-tree.NodeHeap
  "NodeHeap

    This is a min-heap implementation for keeping track of nodes
    during a breadth-first search.  Unlike the NeighborsHeap above,
    the NodeHeap does not have a fixed size and must be able to grow
    as elements are added.

    Internally, the data is stored in a simple binary heap which meets
    the min heap condition:

        heap[i].val < min(heap[2 * i + 1].val, heap[2 * i + 2].val)
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce ball-tree (import-module "sklearn.neighbors.ball_tree"))
