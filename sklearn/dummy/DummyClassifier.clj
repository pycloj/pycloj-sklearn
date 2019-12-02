(ns sklearn.dummy.DummyClassifier
  "
    DummyClassifier is a classifier that makes predictions using simple rules.

    This classifier is useful as a simple baseline to compare with other
    (real) classifiers. Do not use it for real problems.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    Parameters
    ----------
    strategy : str, default=\"stratified\"
        Strategy to use to generate predictions.

        * \"stratified\": generates predictions by respecting the training
          set's class distribution.
        * \"most_frequent\": always predicts the most frequent label in the
          training set.
        * \"prior\": always predicts the class that maximizes the class prior
          (like \"most_frequent\") and ``predict_proba`` returns the class prior.
        * \"uniform\": generates predictions uniformly at random.
        * \"constant\": always predicts a constant label that is provided by
          the user. This is useful for metrics that evaluate a non-majority
          class

          .. versionadded:: 0.17
             Dummy Classifier now supports prior fitting strategy using
             parameter *prior*.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    constant : int or str or array of shape = [n_outputs]
        The explicit constant as predicted by the \"constant\" strategy. This
        parameter is useful only for the \"constant\" strategy.

    Attributes
    ----------
    classes_ : array or list of array of shape = [n_classes]
        Class labels for each output.

    n_classes_ : array or list of array of shape = [n_classes]
        Number of label for each output.

    class_prior_ : array or list of array of shape = [n_classes]
        Probability of each class for each output.

    n_outputs_ : int,
        Number of outputs.

    sparse_output_ : bool,
        True if the array returned from predict is to be in sparse CSC format.
        Is automatically set to True if the input y is passed in sparse format.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce dummy (import-module "sklearn.dummy"))

(defn DummyClassifier 
  "
    DummyClassifier is a classifier that makes predictions using simple rules.

    This classifier is useful as a simple baseline to compare with other
    (real) classifiers. Do not use it for real problems.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    Parameters
    ----------
    strategy : str, default=\"stratified\"
        Strategy to use to generate predictions.

        * \"stratified\": generates predictions by respecting the training
          set's class distribution.
        * \"most_frequent\": always predicts the most frequent label in the
          training set.
        * \"prior\": always predicts the class that maximizes the class prior
          (like \"most_frequent\") and ``predict_proba`` returns the class prior.
        * \"uniform\": generates predictions uniformly at random.
        * \"constant\": always predicts a constant label that is provided by
          the user. This is useful for metrics that evaluate a non-majority
          class

          .. versionadded:: 0.17
             Dummy Classifier now supports prior fitting strategy using
             parameter *prior*.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    constant : int or str or array of shape = [n_outputs]
        The explicit constant as predicted by the \"constant\" strategy. This
        parameter is useful only for the \"constant\" strategy.

    Attributes
    ----------
    classes_ : array or list of array of shape = [n_classes]
        Class labels for each output.

    n_classes_ : array or list of array of shape = [n_classes]
        Number of label for each output.

    class_prior_ : array or list of array of shape = [n_classes]
        Probability of each class for each output.

    n_outputs_ : int,
        Number of outputs.

    sparse_output_ : bool,
        True if the array returned from predict is to be in sparse CSC format.
        Is automatically set to True if the input y is passed in sparse format.
    "
  [ & {:keys [strategy random_state constant]
       :or {strategy "stratified"}} ]
  
   (py/call-attr-kw dummy "DummyClassifier" [] {:strategy strategy :random_state random_state :constant constant }))

(defn fit 
  "Fit the random classifier.

        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data, requires length = n_samples

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target values.

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights.

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

(defn predict 
  "Perform classification on test vectors X.

        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data, requires length = n_samples

        Returns
        -------
        y : array, shape = [n_samples] or [n_samples, n_outputs]
            Predicted target values for X.
        "
  [ self X ]
  (py/call-attr self "predict"  self X ))

(defn predict-log-proba 
  "
        Return log probability estimates for the test vectors X.

        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data, requires length = n_samples

        Returns
        -------
        P : array-like or list of array-like of shape = [n_samples, n_classes]
            Returns the log probability of the sample for each class in
            the model, where classes are ordered arithmetically for each
            output.
        "
  [ self X ]
  (py/call-attr self "predict_log_proba"  self X ))

(defn predict-proba 
  "
        Return probability estimates for the test vectors X.

        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data, requires length = n_samples

        Returns
        -------
        P : array-like or list of array-lke of shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model, where classes are ordered arithmetically, for each
            output.
        "
  [ self X ]
  (py/call-attr self "predict_proba"  self X ))

(defn score 
  "Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : {array-like, None}
            Test samples with shape = (n_samples, n_features) or
            None. Passing None as test samples gives the same result
            as passing real test samples, since DummyClassifier
            operates independently of the sampled observations.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        "
  [ self X y sample_weight ]
  (py/call-attr self "score"  self X y sample_weight ))

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
