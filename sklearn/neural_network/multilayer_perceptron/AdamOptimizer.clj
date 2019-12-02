(ns sklearn.neural-network.multilayer-perceptron.AdamOptimizer
  "Stochastic gradient descent optimizer with Adam

    Note: All default values are from the original Adam paper

    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params

    learning_rate_init : float, optional, default 0.1
        The initial learning rate used. It controls the step-size in updating
        the weights

    beta_1 : float, optional, default 0.9
        Exponential decay rate for estimates of first moment vector, should be
        in [0, 1)

    beta_2 : float, optional, default 0.999
        Exponential decay rate for estimates of second moment vector, should be
        in [0, 1)

    epsilon : float, optional, default 1e-8
        Value for numerical stability

    Attributes
    ----------
    learning_rate : float
        The current learning rate

    t : int
        Timestep

    ms : list, length = len(params)
        First moment vectors

    vs : list, length = len(params)
        Second moment vectors

    References
    ----------
    Kingma, Diederik, and Jimmy Ba.
    \"Adam: A method for stochastic optimization.\"
    arXiv preprint arXiv:1412.6980 (2014).
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

(defn AdamOptimizer 
  "Stochastic gradient descent optimizer with Adam

    Note: All default values are from the original Adam paper

    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params

    learning_rate_init : float, optional, default 0.1
        The initial learning rate used. It controls the step-size in updating
        the weights

    beta_1 : float, optional, default 0.9
        Exponential decay rate for estimates of first moment vector, should be
        in [0, 1)

    beta_2 : float, optional, default 0.999
        Exponential decay rate for estimates of second moment vector, should be
        in [0, 1)

    epsilon : float, optional, default 1e-8
        Value for numerical stability

    Attributes
    ----------
    learning_rate : float
        The current learning rate

    t : int
        Timestep

    ms : list, length = len(params)
        First moment vectors

    vs : list, length = len(params)
        Second moment vectors

    References
    ----------
    Kingma, Diederik, and Jimmy Ba.
    \"Adam: A method for stochastic optimization.\"
    arXiv preprint arXiv:1412.6980 (2014).
    "
  [params & {:keys [learning_rate_init beta_1 beta_2 epsilon]
                       :or {learning_rate_init 0.001 beta_1 0.9 beta_2 0.999 epsilon 1e-08}} ]
    (py/call-attr-kw multilayer-perceptron "AdamOptimizer" [params] {:learning_rate_init learning_rate_init :beta_1 beta_1 :beta_2 beta_2 :epsilon epsilon }))

(defn iteration-ends 
  "Perform update to learning rate and potentially other states at the
        end of an iteration
        "
  [ self time_step ]
  (py/call-attr self "iteration_ends"  self time_step ))

(defn trigger-stopping 
  "Decides whether it is time to stop training

        Parameters
        ----------
        msg : str
            Message passed in for verbose output

        verbose : bool
            Print message to stdin if True

        Returns
        -------
        is_stopping : bool
            True if training needs to stop
        "
  [ self msg verbose ]
  (py/call-attr self "trigger_stopping"  self msg verbose ))

(defn update-params 
  "Update parameters with given gradients

        Parameters
        ----------
        grads : list, length = len(params)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        "
  [ self grads ]
  (py/call-attr self "update_params"  self grads ))
