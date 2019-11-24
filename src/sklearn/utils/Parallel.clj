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
  (py/call-attr utils "Parallel"   ))

(defn debug 
  ""
  [self  & {:keys [msg]} ]
    (py/call-attr-kw utils "debug" [self] {:msg msg }))

(defn dispatch-next 
  "Dispatch more data for parallel processing

        This method is meant to be called concurrently by the multiprocessing
        callback. We rely on the thread-safety of dispatch_one_batch to protect
        against concurrent consumption of the unprotected iterator.

        "
  [ self ]
  (py/call-attr utils "dispatch_next"  self ))

(defn dispatch-one-batch 
  "Prefetch the tasks for the next batch and dispatch them.

        The effective size of the batch is computed here.
        If there are no more jobs to dispatch, return False, else return True.

        The iterator consumption and dispatching is protected by the same
        lock so calling this function should be thread safe.

        "
  [self  & {:keys [iterator]} ]
    (py/call-attr-kw utils "dispatch_one_batch" [self] {:iterator iterator }))

(defn format 
  "Return the formatted representation of the object."
  [self & {:keys [obj indent]
                       :or {indent 0}} ]
    (py/call-attr-kw utils "format" [] {:obj obj :indent indent }))

(defn print-progress 
  "Display the process of the parallel execution only a fraction
           of time, controlled by self.verbose.
        "
  [ self ]
  (py/call-attr utils "print_progress"  self ))

(defn retrieve 
  ""
  [ self ]
  (py/call-attr utils "retrieve"  self ))

(defn warn 
  ""
  [self  & {:keys [msg]} ]
    (py/call-attr-kw utils "warn" [self] {:msg msg }))
