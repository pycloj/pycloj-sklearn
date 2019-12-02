(ns sklearn.gaussian-process.kernels.NormalizedKernelMixin
  "Mixin for kernels which are normalized: k(X, X)=1.

    .. versionadded:: 0.18
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce kernels (import-module "sklearn.gaussian_process.kernels"))

(defn NormalizedKernelMixin 
  "Mixin for kernels which are normalized: k(X, X)=1.

    .. versionadded:: 0.18
    "
  [  ]
  (py/call-attr kernels "NormalizedKernelMixin"  ))

(defn diag 
  "Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        "
  [ self X ]
  (py/call-attr self "diag"  self X ))
