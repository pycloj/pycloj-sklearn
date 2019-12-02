(ns sklearn.ensemble.gradient-boosting.LeastSquaresError
  "Loss function for least squares (LS) estimation.
    Terminal regions need not to be updated for least squares.

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

(defn LeastSquaresError 
  "Loss function for least squares (LS) estimation.
    Terminal regions need not to be updated for least squares.

    Parameters
    ----------
    n_classes : int
        Number of classes
    "
  [  ]
  (py/call-attr gradient-boosting "LeastSquaresError"  ))

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
        "
  [ self y pred ]
  (py/call-attr self "negative_gradient"  self y pred ))

(defn update-terminal-regions 
  "Least squares does not need to update terminal regions.

        But it has to update the predictions.

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
