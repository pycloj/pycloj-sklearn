(ns sklearn.neighbors.KernelDensity
  "Kernel Density Estimation

    Read more in the :ref:`User Guide <kernel_density>`.

    Parameters
    ----------
    bandwidth : float
        The bandwidth of the kernel.

    algorithm : string
        The tree algorithm to use.  Valid options are
        ['kd_tree'|'ball_tree'|'auto'].  Default is 'auto'.

    kernel : string
        The kernel to use.  Valid kernels are
        ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine']
        Default is 'gaussian'.

    metric : string
        The distance metric to use.  Note that not all metrics are
        valid with all algorithms.  Refer to the documentation of
        :class:`BallTree` and :class:`KDTree` for a description of
        available algorithms.  Note that the normalization of the density
        output is correct only for the Euclidean distance metric. Default
        is 'euclidean'.

    atol : float
        The desired absolute tolerance of the result.  A larger tolerance will
        generally lead to faster execution. Default is 0.

    rtol : float
        The desired relative tolerance of the result.  A larger tolerance will
        generally lead to faster execution.  Default is 1E-8.

    breadth_first : boolean
        If true (default), use a breadth-first approach to the problem.
        Otherwise use a depth-first approach.

    leaf_size : int
        Specify the leaf size of the underlying tree.  See :class:`BallTree`
        or :class:`KDTree` for details.  Default is 40.

    metric_params : dict
        Additional parameters to be passed to the tree for use with the
        metric.  For more information, see the documentation of
        :class:`BallTree` or :class:`KDTree`.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce neighbors (import-module "sklearn.neighbors"))

(defn KernelDensity 
  "Kernel Density Estimation

    Read more in the :ref:`User Guide <kernel_density>`.

    Parameters
    ----------
    bandwidth : float
        The bandwidth of the kernel.

    algorithm : string
        The tree algorithm to use.  Valid options are
        ['kd_tree'|'ball_tree'|'auto'].  Default is 'auto'.

    kernel : string
        The kernel to use.  Valid kernels are
        ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine']
        Default is 'gaussian'.

    metric : string
        The distance metric to use.  Note that not all metrics are
        valid with all algorithms.  Refer to the documentation of
        :class:`BallTree` and :class:`KDTree` for a description of
        available algorithms.  Note that the normalization of the density
        output is correct only for the Euclidean distance metric. Default
        is 'euclidean'.

    atol : float
        The desired absolute tolerance of the result.  A larger tolerance will
        generally lead to faster execution. Default is 0.

    rtol : float
        The desired relative tolerance of the result.  A larger tolerance will
        generally lead to faster execution.  Default is 1E-8.

    breadth_first : boolean
        If true (default), use a breadth-first approach to the problem.
        Otherwise use a depth-first approach.

    leaf_size : int
        Specify the leaf size of the underlying tree.  See :class:`BallTree`
        or :class:`KDTree` for details.  Default is 40.

    metric_params : dict
        Additional parameters to be passed to the tree for use with the
        metric.  For more information, see the documentation of
        :class:`BallTree` or :class:`KDTree`.
    "
  [ & {:keys [bandwidth algorithm kernel metric atol rtol breadth_first leaf_size metric_params]
       :or {bandwidth 1.0 algorithm "auto" kernel "gaussian" metric "euclidean" atol 0 rtol 0 breadth_first true leaf_size 40}} ]
  
   (py/call-attr-kw neighbors "KernelDensity" [] {:bandwidth bandwidth :algorithm algorithm :kernel kernel :metric metric :atol atol :rtol rtol :breadth_first breadth_first :leaf_size leaf_size :metric_params metric_params }))

(defn fit 
  "Fit the Kernel Density model on the data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        sample_weight : array_like, shape (n_samples,), optional
            List of sample weights attached to the data X.
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

(defn sample 
  "Generate random samples from the model.

        Currently, this is implemented only for gaussian and tophat kernels.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        random_state : int, RandomState instance or None. default to None
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by `np.random`.

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples.
        "
  [self  & {:keys [n_samples random_state]
                       :or {n_samples 1}} ]
    (py/call-attr-kw self "sample" [] {:n_samples n_samples :random_state random_state }))

(defn score 
  "Compute the total log probability density under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        "
  [ self X y ]
  (py/call-attr self "score"  self X y ))

(defn score-samples 
  "Evaluate the density model on the data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).

        Returns
        -------
        density : ndarray, shape (n_samples,)
            The array of log(density) evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        "
  [ self X ]
  (py/call-attr self "score_samples"  self X ))

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
