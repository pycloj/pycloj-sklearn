(ns sklearn.utils.Parallel
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce utils (import-module "sklearn.utils"))

(defn Parallel 
  ""
  [  ]
  (py/call-attr utils "Parallel"  ))

(defn debug 
  ""
  [ self msg ]
  (py/call-attr self "debug"  self msg ))

(defn dispatch-next 
  "Dispatch more data for parallel processing

        This method is meant to be called concurrently by the multiprocessing
        callback. We rely on the thread-safety of dispatch_one_batch to protect
        against concurrent consumption of the unprotected iterator.

        "
  [ self  ]
  (py/call-attr self "dispatch_next"  self  ))

(defn dispatch-one-batch 
  "Prefetch the tasks for the next batch and dispatch them.

        The effective size of the batch is computed here.
        If there are no more jobs to dispatch, return False, else return True.

        The iterator consumption and dispatching is protected by the same
        lock so calling this function should be thread safe.

        "
  [ self iterator ]
  (py/call-attr self "dispatch_one_batch"  self iterator ))

(defn format 
  "Return the formatted representation of the object."
  [self obj & {:keys [indent]
                       :or {indent 0}} ]
    (py/call-attr-kw self "format" [obj] {:indent indent }))

(defn print-progress 
  "Display the process of the parallel execution only a fraction
           of time, controlled by self.verbose.
        "
  [ self  ]
  (py/call-attr self "print_progress"  self  ))

(defn retrieve 
  ""
  [ self  ]
  (py/call-attr self "retrieve"  self  ))

(defn warn 
  ""
  [ self msg ]
  (py/call-attr self "warn"  self msg ))
