(ns sklearn.compose
  "Meta-estimators for building composite models with transformers

In addition to its current contents, this module will eventually be home to
refurbished versions of Pipeline and FeatureUnion.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce compose (import-module "sklearn.compose"))

(defn make-column-transformer 
  "Construct a ColumnTransformer from the given transformers.

    This is a shorthand for the ColumnTransformer constructor; it does not
    require, and does not permit, naming the transformers. Instead, they will
    be given names automatically based on their types. It also does not allow
    weighting with ``transformer_weights``.

    Parameters
    ----------
    *transformers : tuples of transformers and column selections

    remainder : {'drop', 'passthrough'} or estimator, default 'drop'
        By default, only the specified columns in `transformers` are
        transformed and combined in the output, and the non-specified
        columns are dropped. (default of ``'drop'``).
        By specifying ``remainder='passthrough'``, all remaining columns that
        were not specified in `transformers` will be automatically passed
        through. This subset of columns is concatenated with the output of
        the transformers.
        By setting ``remainder`` to be an estimator, the remaining
        non-specified columns will use the ``remainder`` estimator. The
        estimator must support :term:`fit` and :term:`transform`.

    sparse_threshold : float, default = 0.3
        If the transformed output consists of a mix of sparse and dense data,
        it will be stacked as a sparse matrix if the density is lower than this
        value. Use ``sparse_threshold=0`` to always return dense.
        When the transformed output consists of all sparse or all dense data,
        the stacked result will be sparse or dense, respectively, and this
        keyword will be ignored.

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
    ct : ColumnTransformer

    See also
    --------
    sklearn.compose.ColumnTransformer : Class that allows combining the
        outputs of multiple transformer objects used on column subsets
        of the data into a single feature space.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
    >>> from sklearn.compose import make_column_transformer
    >>> make_column_transformer(
    ...     (StandardScaler(), ['numerical_column']),
    ...     (OneHotEncoder(), ['categorical_column']))
    ...     # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ColumnTransformer(n_jobs=None, remainder='drop', sparse_threshold=0.3,
             transformer_weights=None,
             transformers=[('standardscaler',
                            StandardScaler(...),
                            ['numerical_column']),
                           ('onehotencoder',
                            OneHotEncoder(...),
                            ['categorical_column'])], verbose=False)

    "
  [  ]
  (py/call-attr compose "make_column_transformer"  ))
