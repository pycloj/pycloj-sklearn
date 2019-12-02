(ns sklearn.dummy.DummyRegressor
  "
    DummyRegressor is a regressor that makes predictions using
    simple rules.

    This regressor is useful as a simple baseline to compare with other
    (real) regressors. Do not use it for real problems.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    Parameters
    ----------
    strategy : str
        Strategy to use to generate predictions.

        * \"mean\": always predicts the mean of the training set
        * \"median\": always predicts the median of the training set
        * \"quantile\": always predicts a specified quantile of the training set,
          provided with the quantile parameter.
        * \"constant\": always predicts a constant value that is provided by
          the user.

    constant : int or float or array of shape = [n_outputs]
        The explicit constant as predicted by the \"constant\" strategy. This
        parameter is useful only for the \"constant\" strategy.

    quantile : float in [0.0, 1.0]
        The quantile to predict using the \"quantile\" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.

    Attributes
    ----------
    constant_ : float or array of shape [n_outputs]
        Mean or median or quantile of the training targets or constant value
        given by the user.

    n_outputs_ : int,
        Number of outputs.
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

(defn DummyRegressor 
  "
    DummyRegressor is a regressor that makes predictions using
    simple rules.

    This regressor is useful as a simple baseline to compare with other
    (real) regressors. Do not use it for real problems.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    Parameters
    ----------
    strategy : str
        Strategy to use to generate predictions.

        * \"mean\": always predicts the mean of the training set
        * \"median\": always predicts the median of the training set
        * \"quantile\": always predicts a specified quantile of the training set,
          provided with the quantile parameter.
        * \"constant\": always predicts a constant value that is provided by
          the user.

    constant : int or float or array of shape = [n_outputs]
        The explicit constant as predicted by the \"constant\" strategy. This
        parameter is useful only for the \"constant\" strategy.

    quantile : float in [0.0, 1.0]
        The quantile to predict using the \"quantile\" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.

    Attributes
    ----------
    constant_ : float or array of shape [n_outputs]
        Mean or median or quantile of the training targets or constant value
        given by the user.

    n_outputs_ : int,
        Number of outputs.
    "
  [ & {:keys [strategy constant quantile]
       :or {strategy "mean"}} ]
  
   (py/call-attr-kw dummy "DummyRegressor" [] {:strategy strategy :constant constant :quantile quantile }))

(defn fit 
  "Fit the random regressor.

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
  "
        Perform classification on test vectors X.

        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data, requires length = n_samples

        return_std : boolean, optional
            Whether to return the standard deviation of posterior prediction.
            All zeros in this case.

        Returns
        -------
        y : array, shape = [n_samples] or [n_samples, n_outputs]
            Predicted target values for X.

        y_std : array, shape = [n_samples] or [n_samples, n_outputs]
            Standard deviation of predictive distribution of query points.
        "
  [self X & {:keys [return_std]
                       :or {return_std false}} ]
    (py/call-attr-kw self "predict" [X] {:return_std return_std }))

(defn score 
  "Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : {array-like, None}
            Test samples with shape = (n_samples, n_features) or None.
            For some estimators this may be a
            precomputed kernel matrix instead, shape = (n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for the estimator.
            Passing None as test samples gives the same result
            as passing real test samples, since DummyRegressor
            operates independently of the sampled observations.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
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
