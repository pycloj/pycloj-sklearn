(ns sklearn.ensemble.forest.BaseForest
  "Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce forest (import-module "sklearn.ensemble.forest"))

(defn BaseForest 
  "Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    "
  [base_estimator & {:keys [n_estimators estimator_params bootstrap oob_score n_jobs random_state verbose warm_start class_weight]
                       :or {n_estimators 100 estimator_params () bootstrap false oob_score false verbose 0 warm_start false}} ]
    (py/call-attr-kw forest "BaseForest" [base_estimator] {:n_estimators n_estimators :estimator_params estimator_params :bootstrap bootstrap :oob_score oob_score :n_jobs n_jobs :random_state random_state :verbose verbose :warm_start warm_start :class_weight class_weight }))

(defn apply 
  "Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples, n_estimators]
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        "
  [ self X ]
  (py/call-attr self "apply"  self X ))

(defn decision-path 
  "Return the decision path in the forest

        .. versionadded:: 0.18

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        indicator : sparse csr array, shape = [n_samples, n_nodes]
            Return a node indicator matrix where non zero elements
            indicates that the samples goes through the nodes.

        n_nodes_ptr : array of size (n_estimators + 1, )
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.

        "
  [ self X ]
  (py/call-attr self "decision_path"  self X ))

(defn feature-importances- 
  "Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        "
  [ self ]
    (py/call-attr self "feature_importances_"))

(defn fit 
  "Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
        "
  [ self X y sample_weight ]
  (py/call-attr self "fit"  self X y sample_weight ))

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
