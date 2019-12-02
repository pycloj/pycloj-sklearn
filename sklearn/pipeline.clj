(ns sklearn.pipeline
  "
The :mod:`sklearn.pipeline` module implements utilities to build a composite
estimator, as a chain of transforms and estimators.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce pipeline (import-module "sklearn.pipeline"))

(defn check-memory 
  "Check that ``memory`` is joblib.Memory-like.

    joblib.Memory-like means that ``memory`` can be converted into a
    joblib.Memory instance (typically a str denoting the ``location``)
    or has the same interface (has a ``cache`` method).

    Parameters
    ----------
    memory : None, str or object with the joblib.Memory interface

    Returns
    -------
    memory : object with the joblib.Memory interface

    Raises
    ------
    ValueError
        If ``memory`` is not joblib.Memory-like.
    "
  [ memory ]
  (py/call-attr pipeline "check_memory"  memory ))

(defn clone 
  "Constructs a new estimator with the same parameters.

    Clone does a deep copy of the model in an estimator
    without actually copying attached data. It yields a new estimator
    with the same parameters that has not been fit on any data.

    Parameters
    ----------
    estimator : estimator object, or list, tuple or set of objects
        The estimator or group of estimators to be cloned

    safe : boolean, optional
        If safe is false, clone will fall back to a deep copy on objects
        that are not estimators.

    "
  [estimator & {:keys [safe]
                       :or {safe true}} ]
    (py/call-attr-kw pipeline "clone" [estimator] {:safe safe }))

(defn delayed 
  "Decorator used to capture the arguments of a function."
  [ function check_pickle ]
  (py/call-attr pipeline "delayed"  function check_pickle ))

(defn if-delegate-has-method 
  "Create a decorator for methods that are delegated to a sub-estimator

    This enables ducktyping by hasattr returning True according to the
    sub-estimator.

    Parameters
    ----------
    delegate : string, list of strings or tuple of strings
        Name of the sub-estimator that can be accessed as an attribute of the
        base object. If a list or a tuple of names are provided, the first
        sub-estimator that is an attribute of the base object will be used.

    "
  [ delegate ]
  (py/call-attr pipeline "if_delegate_has_method"  delegate ))

(defn make-pipeline 
  "Construct a Pipeline from the given estimators.

    This is a shorthand for the Pipeline constructor; it does not require, and
    does not permit, naming the estimators. Instead, their names will be set
    to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of estimators.

    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : boolean, optional
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    See also
    --------
    sklearn.pipeline.Pipeline : Class for creating a pipeline of
        transforms with a final estimator.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
    ... # doctest: +NORMALIZE_WHITESPACE
    Pipeline(memory=None,
             steps=[('standardscaler',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('gaussiannb',
                     GaussianNB(priors=None, var_smoothing=1e-09))],
             verbose=False)

    Returns
    -------
    p : Pipeline
    "
  [  ]
  (py/call-attr pipeline "make_pipeline"  ))

(defn make-union 
  "Construct a FeatureUnion from the given transformers.

    This is a shorthand for the FeatureUnion constructor; it does not require,
    and does not permit, naming the transformers. Instead, they will be given
    names automatically based on their types. It also does not allow weighting.

    Parameters
    ----------
    *transformers : list of estimators

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : boolean, optional(default=False)
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    Returns
    -------
    f : FeatureUnion

    See also
    --------
    sklearn.pipeline.FeatureUnion : Class for concatenating the results
        of multiple transformer objects.

    Examples
    --------
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> from sklearn.pipeline import make_union
    >>> make_union(PCA(), TruncatedSVD())  # doctest: +NORMALIZE_WHITESPACE
    FeatureUnion(n_jobs=None,
           transformer_list=[('pca',
                              PCA(copy=True, iterated_power='auto',
                                  n_components=None, random_state=None,
                                  svd_solver='auto', tol=0.0, whiten=False)),
                             ('truncatedsvd',
                              TruncatedSVD(algorithm='randomized',
                              n_components=2, n_iter=5,
                              random_state=None, tol=0.0))],
           transformer_weights=None, verbose=False)
    "
  [  ]
  (py/call-attr pipeline "make_union"  ))
