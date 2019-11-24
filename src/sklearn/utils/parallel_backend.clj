(ns sklearn.utils.parallel-backend
  "Change the default backend used by Parallel inside a with block.

    If ``backend`` is a string it must match a previously registered
    implementation using the ``register_parallel_backend`` function.

    By default the following backends are available:

    - 'loky': single-host, process-based parallelism (used by default),
    - 'threading': single-host, thread-based parallelism,
    - 'multiprocessing': legacy single-host, process-based parallelism.

    'loky' is recommended to run functions that manipulate Python objects.
    'threading' is a low-overhead alternative that is most efficient for
    functions that release the Global Interpreter Lock: e.g. I/O-bound code or
    CPU-bound code in a few calls to native code that explicitly releases the
    GIL.

    In addition, if the `dask` and `distributed` Python packages are installed,
    it is possible to use the 'dask' backend for better scheduling of nested
    parallel calls without over-subscription and potentially distribute
    parallel calls over a networked cluster of several hosts.

    Alternatively the backend can be passed directly as an instance.

    By default all available workers will be used (``n_jobs=-1``) unless the
    caller passes an explicit value for the ``n_jobs`` parameter.

    This is an alternative to passing a ``backend='backend_name'`` argument to
    the ``Parallel`` class constructor. It is particularly useful when calling
    into library code that uses joblib internally but does not expose the
    backend argument in its own API.

    >>> from operator import neg
    >>> with parallel_backend('threading'):
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
    ...
    [-1, -2, -3, -4, -5]

    Warning: this function is experimental and subject to change in a future
    version of joblib.

    Joblib also tries to limit the oversubscription by limiting the number of
    threads usable in some third-party library threadpools like OpenBLAS, MKL
    or OpenMP. The default limit in each worker is set to
    ``max(cpu_count() // effective_n_jobs, 1)`` but this limit can be
    overwritten with the ``inner_max_num_threads`` argument which will be used
    to set this limit in the child processes.

    .. versionadded:: 0.10

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce utils (import-module "sklearn.utils"))

(defn parallel-backend 
  "Change the default backend used by Parallel inside a with block.

    If ``backend`` is a string it must match a previously registered
    implementation using the ``register_parallel_backend`` function.

    By default the following backends are available:

    - 'loky': single-host, process-based parallelism (used by default),
    - 'threading': single-host, thread-based parallelism,
    - 'multiprocessing': legacy single-host, process-based parallelism.

    'loky' is recommended to run functions that manipulate Python objects.
    'threading' is a low-overhead alternative that is most efficient for
    functions that release the Global Interpreter Lock: e.g. I/O-bound code or
    CPU-bound code in a few calls to native code that explicitly releases the
    GIL.

    In addition, if the `dask` and `distributed` Python packages are installed,
    it is possible to use the 'dask' backend for better scheduling of nested
    parallel calls without over-subscription and potentially distribute
    parallel calls over a networked cluster of several hosts.

    Alternatively the backend can be passed directly as an instance.

    By default all available workers will be used (``n_jobs=-1``) unless the
    caller passes an explicit value for the ``n_jobs`` parameter.

    This is an alternative to passing a ``backend='backend_name'`` argument to
    the ``Parallel`` class constructor. It is particularly useful when calling
    into library code that uses joblib internally but does not expose the
    backend argument in its own API.

    >>> from operator import neg
    >>> with parallel_backend('threading'):
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
    ...
    [-1, -2, -3, -4, -5]

    Warning: this function is experimental and subject to change in a future
    version of joblib.

    Joblib also tries to limit the oversubscription by limiting the number of
    threads usable in some third-party library threadpools like OpenBLAS, MKL
    or OpenMP. The default limit in each worker is set to
    ``max(cpu_count() // effective_n_jobs, 1)`` but this limit can be
    overwritten with the ``inner_max_num_threads`` argument which will be used
    to set this limit in the child processes.

    .. versionadded:: 0.10

    "
  [ & {:keys [backend n_jobs inner_max_num_threads]
       :or {n_jobs -1}} ]
  
   (py/call-attr-kw utils "parallel_backend" [] {:backend backend :n_jobs n_jobs :inner_max_num_threads inner_max_num_threads }))

(defn unregister 
  ""
  [ self ]
  (py/call-attr utils "unregister"  self ))
