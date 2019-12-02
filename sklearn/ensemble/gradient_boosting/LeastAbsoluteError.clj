(ns sklearn.ensemble.gradient-boosting.LeastAbsoluteError
  "Loss function for least absolute deviation (LAD) regression.

    Parameters
    ----------
    n_classes : int
        Number of classes
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

(defn LeastAbsoluteError 
  "Loss function for least absolute deviation (LAD) regression.

    Parameters
    ----------
    n_classes : int
        Number of classes
    "
  [  ]
  (py/call-attr gradient-boosting "LeastAbsoluteError"  ))

(defn init-estimator 
  ""
  [ self  ]
  (py/call-attr self "init_estimator"  self  ))

(defn negative-gradient 
  "Compute the negative gradient.

        1.0 if y - pred > 0.0 else -1.0

        Parameters
        ----------
        y : array, shape (n_samples,)
            The target labels.

        pred : array, shape (n_samples,)
            The predictions.
        "
  [ self y pred ]
  (py/call-attr self "negative_gradient"  self y pred ))

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
