(ns sklearn.tree.tree.BaseDecisionTree
  "Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tree (import-module "sklearn.tree.tree"))

(defn BaseDecisionTree 
  "Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    "
  [criterion splitter max_depth min_samples_split min_samples_leaf min_weight_fraction_leaf max_features max_leaf_nodes random_state min_impurity_decrease min_impurity_split class_weight & {:keys [presort]
                       :or {presort false}} ]
    (py/call-attr-kw tree "BaseDecisionTree" [criterion splitter max_depth min_samples_split min_samples_leaf min_weight_fraction_leaf max_features max_leaf_nodes random_state min_impurity_decrease min_impurity_split class_weight] {:presort presort }))

(defn apply 
  "
        Returns the index of the leaf that each sample is predicted as.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples,]
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        "
  [self X & {:keys [check_input]
                       :or {check_input true}} ]
    (py/call-attr-kw self "apply" [X] {:check_input check_input }))

(defn decision-path 
  "Return the decision path in the tree

        .. versionadded:: 0.18

        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        indicator : sparse csr array, shape = [n_samples, n_nodes]
            Return a node indicator matrix where non zero elements
            indicates that the samples goes through the nodes.

        "
  [self X & {:keys [check_input]
                       :or {check_input true}} ]
    (py/call-attr-kw self "decision_path" [X] {:check_input check_input }))

(defn feature-importances- 
  "Return the feature importances.

        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        "
  [ self ]
    (py/call-attr self "feature_importances_"))

(defn fit 
  ""
  [self X y sample_weight & {:keys [check_input X_idx_sorted]
                       :or {check_input true}} ]
    (py/call-attr-kw self "fit" [X y sample_weight] {:check_input check_input :X_idx_sorted X_idx_sorted }))

(defn get-depth 
  "Returns the depth of the decision tree.

        The depth of a tree is the maximum distance between the root
        and any leaf.
        "
  [ self  ]
  (py/call-attr self "get_depth"  self  ))

(defn get-n-leaves 
  "Returns the number of leaves of the decision tree.
        "
  [ self  ]
  (py/call-attr self "get_n_leaves"  self  ))

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

(defn predict 
  "Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        "
  [self X & {:keys [check_input]
                       :or {check_input true}} ]
    (py/call-attr-kw self "predict" [X] {:check_input check_input }))

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
