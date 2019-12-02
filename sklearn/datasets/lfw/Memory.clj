(ns sklearn.datasets.lfw.Memory
  " A context object for caching a function's return value each time it
        is called with the same input arguments.

        All values are cached on the filesystem, in a deep directory
        structure.

        Read more in the :ref:`User Guide <memory>`.

        Parameters
        ----------
        location: str or None
            The path of the base directory to use as a data store
            or None. If None is given, no caching is done and
            the Memory object is completely transparent. This option
            replaces cachedir since version 0.12.

        backend: str, optional
            Type of store backend for reading/writing cache files.
            Default: 'local'.
            The 'local' backend is using regular filesystem operations to
            manipulate data (open, mv, etc) in the backend.

        cachedir: str or None, optional

            .. deprecated: 0.12
                'cachedir' has been deprecated in 0.12 and will be
                removed in 0.14. Use the 'location' parameter instead.

        mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
            The memmapping mode used when loading from cache
            numpy arrays. See numpy.load for the meaning of the
            arguments.

        compress: boolean, or integer, optional
            Whether to zip the stored data on disk. If an integer is
            given, it should be between 1 and 9, and sets the amount
            of compression. Note that compressed arrays cannot be
            read by memmapping.

        verbose: int, optional
            Verbosity flag, controls the debug messages that are issued
            as functions are evaluated.

        bytes_limit: int, optional
            Limit in bytes of the size of the cache.

        backend_options: dict, optional
            Contains a dictionnary of named parameters used to configure
            the store backend.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lfw (import-module "sklearn.datasets.lfw"))

(defn Memory 
  " A context object for caching a function's return value each time it
        is called with the same input arguments.

        All values are cached on the filesystem, in a deep directory
        structure.

        Read more in the :ref:`User Guide <memory>`.

        Parameters
        ----------
        location: str or None
            The path of the base directory to use as a data store
            or None. If None is given, no caching is done and
            the Memory object is completely transparent. This option
            replaces cachedir since version 0.12.

        backend: str, optional
            Type of store backend for reading/writing cache files.
            Default: 'local'.
            The 'local' backend is using regular filesystem operations to
            manipulate data (open, mv, etc) in the backend.

        cachedir: str or None, optional

            .. deprecated: 0.12
                'cachedir' has been deprecated in 0.12 and will be
                removed in 0.14. Use the 'location' parameter instead.

        mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
            The memmapping mode used when loading from cache
            numpy arrays. See numpy.load for the meaning of the
            arguments.

        compress: boolean, or integer, optional
            Whether to zip the stored data on disk. If an integer is
            given, it should be between 1 and 9, and sets the amount
            of compression. Note that compressed arrays cannot be
            read by memmapping.

        verbose: int, optional
            Verbosity flag, controls the debug messages that are issued
            as functions are evaluated.

        bytes_limit: int, optional
            Limit in bytes of the size of the cache.

        backend_options: dict, optional
            Contains a dictionnary of named parameters used to configure
            the store backend.
    "
  [location & {:keys [backend cachedir mmap_mode compress verbose bytes_limit backend_options]
                       :or {backend "local" compress false verbose 1}} ]
    (py/call-attr-kw lfw "Memory" [location] {:backend backend :cachedir cachedir :mmap_mode mmap_mode :compress compress :verbose verbose :bytes_limit bytes_limit :backend_options backend_options }))

(defn cache 
  " Decorates the given function func to only compute its return
            value for input arguments not cached on disk.

            Parameters
            ----------
            func: callable, optional
                The function to be decorated
            ignore: list of strings
                A list of arguments name to ignore in the hashing
            verbose: integer, optional
                The verbosity mode of the function. By default that
                of the memory object is used.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments. By default that of the memory object is used.

            Returns
            -------
            decorated_func: MemorizedFunc object
                The returned object is a MemorizedFunc object, that is
                callable (behaves like a function), but offers extra
                methods for cache lookup and management. See the
                documentation for :class:`joblib.memory.MemorizedFunc`.
        "
  [self func ignore verbose & {:keys [mmap_mode]
                       :or {mmap_mode false}} ]
    (py/call-attr-kw self "cache" [func ignore verbose] {:mmap_mode mmap_mode }))

(defn cachedir 
  ""
  [ self ]
    (py/call-attr self "cachedir"))

(defn clear 
  " Erase the complete cache directory.
        "
  [self  & {:keys [warn]
                       :or {warn true}} ]
    (py/call-attr-kw self "clear" [] {:warn warn }))

(defn debug 
  ""
  [ self msg ]
  (py/call-attr self "debug"  self msg ))

(defn eval 
  " Eval function func with arguments `*args` and `**kwargs`,
            in the context of the memory.

            This method works similarly to the builtin `apply`, except
            that the function is called only if the cache is not
            up to date.

        "
  [ self func ]
  (py/call-attr self "eval"  self func ))

(defn format 
  "Return the formatted representation of the object."
  [self obj & {:keys [indent]
                       :or {indent 0}} ]
    (py/call-attr-kw self "format" [obj] {:indent indent }))

(defn reduce-size 
  "Remove cache elements to make cache size fit in ``bytes_limit``."
  [ self  ]
  (py/call-attr self "reduce_size"  self  ))

(defn warn 
  ""
  [ self msg ]
  (py/call-attr self "warn"  self msg ))
