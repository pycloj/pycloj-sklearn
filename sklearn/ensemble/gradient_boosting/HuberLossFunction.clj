(ns sklearn.ensemble.gradient-boosting.HuberLossFunction
  "Huber loss function for robust regression.

    M-Regression proposed in Friedman 2001.

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    Parameters
    ----------
    n_classes : int
        Number of classes

    alpha : float
        Percentile at which to extract score
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

(defn HuberLossFunction 
  "Huber loss function for robust regression.

    M-Regression proposed in Friedman 2001.

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    Parameters
    ----------
    n_classes : int
        Number of classes

    alpha : float
        Percentile at which to extract score
    "
  [  ]
  (py/call-attr gradient-boosting "HuberLossFunction"  ))

(defn init-estimator 
  ""
  [ self  ]
  (py/call-attr self "init_estimator"  self  ))

(defn negative-gradient 
  "Compute the negative gradient.

        Parameters
        ----------
        y : array, shape (n_samples,)
            The target labels.

        pred : array, shape (n_samples,)
            The predictions.

        sample_weight : array-like, shape (n_samples,), optional
            Sample weights.
        "
  [ self y pred sample_weight ]
  (py/call-attr self "negative_gradient"  self y pred sample_weight ))

(defn update-terminal-regions 
  "Update the terminal regions (=leaves) of the given tree and
        updates the current predictions of the model. Traverses tree
        and invokes template method `_update_terminal_region`.

        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : array, shape (n, m)
            The data array.
        y : array, shape (n,)
            The target labels.
        residual : array, shape (n,)
            The residuals (usually the negative gradient).
        y_pred : array, shape (n,)
            The predictions.
        sample_weight : array, shape (n,)
            The weight of each sample.
        sample_mask : array, shape (n,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default 0
            The index of the estimator being updated.

        "
  [self tree X y residual y_pred sample_weight sample_mask & {:keys [learning_rate k]
                       :or {learning_rate 0.1 k 0}} ]
    (py/call-attr-kw self "update_terminal_regions" [tree X y residual y_pred sample_weight sample_mask] {:learning_rate learning_rate :k k }))
