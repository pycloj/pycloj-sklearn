(ns sklearn.ensemble.gradient-boosting.VerboseReporter
  "Reports verbose output to stdout.

    Parameters
    ----------
    verbose : int
        Verbosity level. If ``verbose==1`` output is printed once in a while
        (when iteration mod verbose_mod is zero).; if larger than 1 then output
        is printed for each update.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce gradient-boosting (import-module "sklearn.ensemble.gradient_boosting"))

(defn VerboseReporter 
  "Reports verbose output to stdout.

    Parameters
    ----------
    verbose : int
        Verbosity level. If ``verbose==1`` output is printed once in a while
        (when iteration mod verbose_mod is zero).; if larger than 1 then output
        is printed for each update.
    "
  [ verbose ]
  (py/call-attr gradient-boosting "VerboseReporter"  verbose ))

(defn init 
  "Initialize reporter

        Parameters
        ----------
        est : Estimator
            The estimator

        begin_at_stage : int
            stage at which to begin reporting
        "
  [self est & {:keys [begin_at_stage]
                       :or {begin_at_stage 0}} ]
    (py/call-attr-kw self "init" [est] {:begin_at_stage begin_at_stage }))

(defn update 
  "Update reporter with new iteration.

        Parameters
        ----------
        j : int
            The new iteration
        est : Estimator
            The estimator
        "
  [ self j est ]
  (py/call-attr self "update"  self j est ))
