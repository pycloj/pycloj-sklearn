(ns sklearn.utils.Memory
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce utils (import-module "sklearn.utils"))

(defn Memory 
  ""
  [  ]
  (py/call-attr utils "Memory"   ))

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
  [self & {:keys [func ignore verbose mmap_mode]
                       :or {mmap_mode false}} ]
    (py/call-attr-kw utils "cache" [] {:func func :ignore ignore :verbose verbose :mmap_mode mmap_mode }))

(defn cachedir 
  ""
  [ self ]
    (py/call-attr utils "cachedir"  self))

(defn clear 
  " Erase the complete cache directory.
        "
  [self & {:keys [warn]
                       :or {warn true}} ]
    (py/call-attr-kw utils "clear" [] {:warn warn }))

(defn debug 
  ""
  [self  & {:keys [msg]} ]
    (py/call-attr-kw utils "debug" [self] {:msg msg }))

(defn eval 
  " Eval function func with arguments `*args` and `**kwargs`,
            in the context of the memory.

            This method works similarly to the builtin `apply`, except
            that the function is called only if the cache is not
            up to date.

        "
  [self  & {:keys [func]} ]
    (py/call-attr-kw utils "eval" [self] {:func func }))

(defn format 
  "Return the formatted representation of the object."
  [self & {:keys [obj indent]
                       :or {indent 0}} ]
    (py/call-attr-kw utils "format" [] {:obj obj :indent indent }))

(defn reduce-size 
  "Remove cache elements to make cache size fit in ``bytes_limit``."
  [ self ]
  (py/call-attr utils "reduce_size"  self ))

(defn warn 
  ""
  [self  & {:keys [msg]} ]
    (py/call-attr-kw utils "warn" [self] {:msg msg }))
