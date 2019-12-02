(ns sklearn.neighbors.base.NeighborsBase
  "Base class for nearest neighbors estimators."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.neighbors.base"))

(defn NeighborsBase 
  "Base class for nearest neighbors estimators."
  [n_neighbors radius & {:keys [algorithm leaf_size metric p metric_params n_jobs]
                       :or {algorithm "auto" leaf_size 30 metric "minkowski" p 2}} ]
    (py/call-attr-kw base "NeighborsBase" [n_neighbors radius] {:algorithm algorithm :leaf_size leaf_size :metric metric :p p :metric_params metric_params :n_jobs n_jobs }))

(defn get-params 
  "Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        "
  [self  & {:keys [deep]
                       :or {deep true}} ]
    (py/call-attr-kw self "get_params" [] {:deep deep }))

(defn set-params 
  "Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        "
  [ self  ]
  (py/call-attr self "set_params"  self  ))
