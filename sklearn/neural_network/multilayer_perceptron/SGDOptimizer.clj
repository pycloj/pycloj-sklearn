(ns sklearn.neural-network.multilayer-perceptron.SGDOptimizer
  "Stochastic gradient descent optimizer with momentum

    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params

    learning_rate_init : float, optional, default 0.1
        The initial learning rate used. It controls the step-size in updating
        the weights

    lr_schedule : {'constant', 'adaptive', 'invscaling'}, default 'constant'
        Learning rate schedule for weight updates.

        -'constant', is a constant learning rate given by
         'learning_rate_init'.

        -'invscaling' gradually decreases the learning rate 'learning_rate_' at
          each time step 't' using an inverse scaling exponent of 'power_t'.
          learning_rate_ = learning_rate_init / pow(t, power_t)

        -'adaptive', keeps the learning rate constant to
         'learning_rate_init' as long as the training keeps decreasing.
         Each time 2 consecutive epochs fail to decrease the training loss by
         tol, or fail to increase validation score by tol if 'early_stopping'
         is on, the current learning rate is divided by 5.

    momentum : float, optional, default 0.9
        Value of momentum used, must be larger than or equal to 0

    nesterov : bool, optional, default True
        Whether to use nesterov's momentum or not. Use nesterov's if True

    Attributes
    ----------
    learning_rate : float
        the current learning rate

    velocities : list, length = len(params)
        velocities that are used to update params
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

(defn SGDOptimizer 
  "Stochastic gradient descent optimizer with momentum

    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params

    learning_rate_init : float, optional, default 0.1
        The initial learning rate used. It controls the step-size in updating
        the weights

    lr_schedule : {'constant', 'adaptive', 'invscaling'}, default 'constant'
        Learning rate schedule for weight updates.

        -'constant', is a constant learning rate given by
         'learning_rate_init'.

        -'invscaling' gradually decreases the learning rate 'learning_rate_' at
          each time step 't' using an inverse scaling exponent of 'power_t'.
          learning_rate_ = learning_rate_init / pow(t, power_t)

        -'adaptive', keeps the learning rate constant to
         'learning_rate_init' as long as the training keeps decreasing.
         Each time 2 consecutive epochs fail to decrease the training loss by
         tol, or fail to increase validation score by tol if 'early_stopping'
         is on, the current learning rate is divided by 5.

    momentum : float, optional, default 0.9
        Value of momentum used, must be larger than or equal to 0

    nesterov : bool, optional, default True
        Whether to use nesterov's momentum or not. Use nesterov's if True

    Attributes
    ----------
    learning_rate : float
        the current learning rate

    velocities : list, length = len(params)
        velocities that are used to update params
    "
  [params & {:keys [learning_rate_init lr_schedule momentum nesterov power_t]
                       :or {learning_rate_init 0.1 lr_schedule "constant" momentum 0.9 nesterov true power_t 0.5}} ]
    (py/call-attr-kw multilayer-perceptron "SGDOptimizer" [params] {:learning_rate_init learning_rate_init :lr_schedule lr_schedule :momentum momentum :nesterov nesterov :power_t power_t }))

(defn iteration-ends 
  "Perform updates to learning rate and potential other states at the
        end of an iteration

        Parameters
        ----------
        time_step : int
            number of training samples trained on so far, used to update
            learning rate for 'invscaling'
        "
  [ self time_step ]
  (py/call-attr self "iteration_ends"  self time_step ))

(defn trigger-stopping 
  ""
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
