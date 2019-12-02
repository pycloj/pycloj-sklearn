(ns sklearn.neural-network.multilayer-perceptron.BaseMultilayerPerceptron
  "Base class for MLP classification and regression.

    Warning: This class should not be used directly.
    Use derived classes instead.

    .. versionadded:: 0.18
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce multilayer-perceptron (import-module "sklearn.neural_network.multilayer_perceptron"))

(defn BaseMultilayerPerceptron 
  "Base class for MLP classification and regression.

    Warning: This class should not be used directly.
    Use derived classes instead.

    .. versionadded:: 0.18
    "
  [ hidden_layer_sizes activation solver alpha batch_size learning_rate learning_rate_init power_t max_iter loss shuffle random_state tol verbose warm_start momentum nesterovs_momentum early_stopping validation_fraction beta_1 beta_2 epsilon n_iter_no_change ]
  (py/call-attr multilayer-perceptron "BaseMultilayerPerceptron"  hidden_layer_sizes activation solver alpha batch_size learning_rate learning_rate_init power_t max_iter loss shuffle random_state tol verbose warm_start momentum nesterovs_momentum early_stopping validation_fraction beta_1 beta_2 epsilon n_iter_no_change ))

(defn fit 
  "Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : returns a trained MLP model.
        "
  [ self X y ]
  (py/call-attr self "fit"  self X y ))

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

(defn partial-fit 
  "Update the model with a single iteration over the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        self : returns a trained MLP model.
        "
  [ self ]
    (py/call-attr self "partial_fit"))

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
