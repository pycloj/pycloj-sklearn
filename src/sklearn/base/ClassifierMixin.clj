(ns sklearn.base.ClassifierMixin
  "Mixin class for all classifiers in scikit-learn."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "sklearn.base"))

(defn ClassifierMixin 
  "Mixin class for all classifiers in scikit-learn."
  [  ]
  (py/call-attr base "ClassifierMixin"  ))
(defn score 
  "Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        "
  [self X y  & {:keys [sample_weight]} ]
    (py/call-attr-kw self "score" [X y] {:sample_weight sample_weight }))
