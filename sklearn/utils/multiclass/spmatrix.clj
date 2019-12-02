(ns sklearn.utils.multiclass.spmatrix
  " This class provides a base class for all sparse matrices.  It
    cannot be instantiated.  Most of the work is provided by subclasses.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce multiclass (import-module "sklearn.utils.multiclass"))

(defn spmatrix 
  " This class provides a base class for all sparse matrices.  It
    cannot be instantiated.  Most of the work is provided by subclasses.
    "
  [ & {:keys [maxprint]
       :or {maxprint 50}} ]
  
   (py/call-attr-kw multiclass "spmatrix" [] {:maxprint maxprint }))

(defn asformat 
  "Return this matrix in the passed format.

        Parameters
        ----------
        format : {str, None}
            The desired matrix format (\"csr\", \"csc\", \"lil\", \"dok\", \"array\", ...)
            or None for no conversion.
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.

        Returns
        -------
        A : This matrix in the passed format.
        "
  [self format & {:keys [copy]
                       :or {copy false}} ]
    (py/call-attr-kw self "asformat" [format] {:copy copy }))

(defn asfptype 
  "Upcast matrix to a floating point format (if necessary)"
  [ self  ]
  (py/call-attr self "asfptype"  self  ))

(defn astype 
  "Cast the matrix elements to a specified type.

        Parameters
        ----------
        dtype : string or numpy dtype
            Typecode or data-type to which to cast the data.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
            Controls what kind of data casting may occur.
            Defaults to 'unsafe' for backwards compatibility.
            'no' means the data types should not be cast at all.
            'equiv' means only byte-order changes are allowed.
            'safe' means only casts which can preserve values are allowed.
            'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
            'unsafe' means any data conversions may be done.
        copy : bool, optional
            If `copy` is `False`, the result might share some memory with this
            matrix. If `copy` is `True`, it is guaranteed that the result and
            this matrix do not share any memory.
        "
  [self dtype & {:keys [casting copy]
                       :or {casting "unsafe" copy true}} ]
    (py/call-attr-kw self "astype" [dtype] {:casting casting :copy copy }))

(defn conj 
  "Element-wise complex conjugation.

        If the matrix is of non-complex data type and `copy` is False,
        this method does nothing and the data is not copied.

        Parameters
        ----------
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.

        Returns
        -------
        A : The element-wise complex conjugate.

        "
  [self  & {:keys [copy]
                       :or {copy true}} ]
    (py/call-attr-kw self "conj" [] {:copy copy }))

(defn conjugate 
  "Element-wise complex conjugation.

        If the matrix is of non-complex data type and `copy` is False,
        this method does nothing and the data is not copied.

        Parameters
        ----------
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.

        Returns
        -------
        A : The element-wise complex conjugate.

        "
  [self  & {:keys [copy]
                       :or {copy true}} ]
    (py/call-attr-kw self "conjugate" [] {:copy copy }))

(defn copy 
  "Returns a copy of this matrix.

        No data/indices will be shared between the returned value and current
        matrix.
        "
  [ self  ]
  (py/call-attr self "copy"  self  ))

(defn count-nonzero 
  "Number of non-zero entries, equivalent to

        np.count_nonzero(a.toarray())

        Unlike getnnz() and the nnz property, which return the number of stored
        entries (the length of the data attribute), this method counts the
        actual number of non-zero entries in data.
        "
  [ self  ]
  (py/call-attr self "count_nonzero"  self  ))

(defn diagonal 
  "Returns the k-th diagonal of the matrix.

        Parameters
        ----------
        k : int, optional
            Which diagonal to set, corresponding to elements a[i, i+k].
            Default: 0 (the main diagonal).

            .. versionadded:: 1.0

        See also
        --------
        numpy.diagonal : Equivalent numpy function.

        Examples
        --------
        >>> from scipy.sparse import csr_matrix
        >>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> A.diagonal()
        array([1, 0, 5])
        >>> A.diagonal(k=1)
        array([2, 3])
        "
  [self  & {:keys [k]
                       :or {k 0}} ]
    (py/call-attr-kw self "diagonal" [] {:k k }))

(defn dot 
  "Ordinary dot product

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.sparse import csr_matrix
        >>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> v = np.array([1, 0, -1])
        >>> A.dot(v)
        array([ 1, -3, -1], dtype=int64)

        "
  [ self other ]
  (py/call-attr self "dot"  self other ))

(defn getH 
  "Return the Hermitian transpose of this matrix.

        See Also
        --------
        numpy.matrix.getH : NumPy's implementation of `getH` for matrices
        "
  [ self  ]
  (py/call-attr self "getH"  self  ))

(defn get-shape 
  "Get shape of a matrix."
  [ self  ]
  (py/call-attr self "get_shape"  self  ))

(defn getcol 
  "Returns a copy of column j of the matrix, as an (m x 1) sparse
        matrix (column vector).
        "
  [ self j ]
  (py/call-attr self "getcol"  self j ))

(defn getformat 
  "Format of a matrix representation as a string."
  [ self  ]
  (py/call-attr self "getformat"  self  ))

(defn getmaxprint 
  "Maximum number of elements to display when printed."
  [ self  ]
  (py/call-attr self "getmaxprint"  self  ))

(defn getnnz 
  "Number of stored values, including explicit zeros.

        Parameters
        ----------
        axis : None, 0, or 1
            Select between the number of values across the whole matrix, in
            each column, or in each row.

        See also
        --------
        count_nonzero : Number of non-zero entries
        "
  [ self axis ]
  (py/call-attr self "getnnz"  self axis ))

(defn getrow 
  "Returns a copy of row i of the matrix, as a (1 x n) sparse
        matrix (row vector).
        "
  [ self i ]
  (py/call-attr self "getrow"  self i ))

(defn maximum 
  "Element-wise maximum between this and another matrix."
  [ self other ]
  (py/call-attr self "maximum"  self other ))

(defn mean 
  "
        Compute the arithmetic mean along the specified axis.

        Returns the average of the matrix elements. The average is taken
        over all elements in the matrix by default, otherwise over the
        specified axis. `float64` intermediate and return values are used
        for integer inputs.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the mean is computed. The default is to compute
            the mean of all elements in the matrix (i.e. `axis` = `None`).
        dtype : data-type, optional
            Type to use in computing the mean. For integer inputs, the default
            is `float64`; for floating point inputs, it is the same as the
            input dtype.

            .. versionadded:: 0.18.0

        out : np.matrix, optional
            Alternative output matrix in which to place the result. It must
            have the same shape as the expected output, but the type of the
            output values will be cast if necessary.

            .. versionadded:: 0.18.0

        Returns
        -------
        m : np.matrix

        See Also
        --------
        numpy.matrix.mean : NumPy's implementation of 'mean' for matrices

        "
  [ self axis dtype out ]
  (py/call-attr self "mean"  self axis dtype out ))

(defn minimum 
  "Element-wise minimum between this and another matrix."
  [ self other ]
  (py/call-attr self "minimum"  self other ))

(defn multiply 
  "Point-wise multiplication by another matrix
        "
  [ self other ]
  (py/call-attr self "multiply"  self other ))

(defn nnz 
  "Number of stored values, including explicit zeros.

        See also
        --------
        count_nonzero : Number of non-zero entries
        "
  [ self ]
    (py/call-attr self "nnz"))

(defn nonzero 
  "nonzero indices

        Returns a tuple of arrays (row,col) containing the indices
        of the non-zero elements of the matrix.

        Examples
        --------
        >>> from scipy.sparse import csr_matrix
        >>> A = csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
        >>> A.nonzero()
        (array([0, 0, 1, 2, 2]), array([0, 1, 2, 0, 2]))

        "
  [ self  ]
  (py/call-attr self "nonzero"  self  ))

(defn power 
  "Element-wise power."
  [ self n dtype ]
  (py/call-attr self "power"  self n dtype ))

(defn reshape 
  "reshape(self, shape, order='C', copy=False)

        Gives a new shape to a sparse matrix without changing its data.

        Parameters
        ----------
        shape : length-2 tuple of ints
            The new shape should be compatible with the original shape.
        order : {'C', 'F'}, optional
            Read the elements using this index order. 'C' means to read and
            write the elements using C-like index order; e.g. read entire first
            row, then second row, etc. 'F' means to read and write the elements
            using Fortran-like index order; e.g. read entire first column, then
            second column, etc.
        copy : bool, optional
            Indicates whether or not attributes of self should be copied
            whenever possible. The degree to which attributes are copied varies
            depending on the type of sparse matrix being used.

        Returns
        -------
        reshaped_matrix : sparse matrix
            A sparse matrix with the given `shape`, not necessarily of the same
            format as the current object.

        See Also
        --------
        numpy.matrix.reshape : NumPy's implementation of 'reshape' for
                               matrices
        "
  [ self  ]
  (py/call-attr self "reshape"  self  ))

(defn resize 
  "Resize the matrix in-place to dimensions given by ``shape``

        Any elements that lie within the new shape will remain at the same
        indices, while non-zero elements lying outside the new shape are
        removed.

        Parameters
        ----------
        shape : (int, int)
            number of rows and columns in the new matrix

        Notes
        -----
        The semantics are not identical to `numpy.ndarray.resize` or
        `numpy.resize`.  Here, the same data will be maintained at each index
        before and after reshape, if that index is within the new bounds.  In
        numpy, resizing maintains contiguity of the array, moving elements
        around in the logical matrix but not within a flattened representation.

        We give no guarantees about whether the underlying data attributes
        (arrays, etc.) will be modified in place or replaced with new objects.
        "
  [ self shape ]
  (py/call-attr self "resize"  self shape ))

(defn set-shape 
  "See `reshape`."
  [ self shape ]
  (py/call-attr self "set_shape"  self shape ))

(defn setdiag 
  "
        Set diagonal or off-diagonal elements of the array.

        Parameters
        ----------
        values : array_like
            New values of the diagonal elements.

            Values may have any length.  If the diagonal is longer than values,
            then the remaining diagonal entries will not be set.  If values if
            longer than the diagonal, then the remaining values are ignored.

            If a scalar value is given, all of the diagonal is set to it.

        k : int, optional
            Which off-diagonal to set, corresponding to elements a[i,i+k].
            Default: 0 (the main diagonal).

        "
  [self values & {:keys [k]
                       :or {k 0}} ]
    (py/call-attr-kw self "setdiag" [values] {:k k }))

(defn shape 
  "Get shape of a matrix."
  [ self ]
    (py/call-attr self "shape"))

(defn sum 
  "
        Sum the matrix elements over a given axis.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the sum of all the matrix elements, returning a scalar
            (i.e. `axis` = `None`).
        dtype : dtype, optional
            The type of the returned matrix and of the accumulator in which
            the elements are summed.  The dtype of `a` is used by default
            unless `a` has an integer dtype of less precision than the default
            platform integer.  In that case, if `a` is signed then the platform
            integer is used while if `a` is unsigned then an unsigned integer
            of the same precision as the platform integer is used.

            .. versionadded:: 0.18.0

        out : np.matrix, optional
            Alternative output matrix in which to place the result. It must
            have the same shape as the expected output, but the type of the
            output values will be cast if necessary.

            .. versionadded:: 0.18.0

        Returns
        -------
        sum_along_axis : np.matrix
            A matrix with the same shape as `self`, with the specified
            axis removed.

        See Also
        --------
        numpy.matrix.sum : NumPy's implementation of 'sum' for matrices

        "
  [ self axis dtype out ]
  (py/call-attr self "sum"  self axis dtype out ))

(defn toarray 
  "
        Return a dense ndarray representation of this matrix.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to store multi-dimensional data in C (row-major)
            or Fortran (column-major) order in memory. The default
            is 'None', indicating the NumPy default of C-ordered.
            Cannot be specified in conjunction with the `out`
            argument.

        out : ndarray, 2-dimensional, optional
            If specified, uses this array as the output buffer
            instead of allocating a new array to return. The provided
            array must have the same shape and dtype as the sparse
            matrix on which you are calling the method. For most
            sparse types, `out` is required to be memory contiguous
            (either C or Fortran ordered).

        Returns
        -------
        arr : ndarray, 2-dimensional
            An array with the same shape and containing the same
            data represented by the sparse matrix, with the requested
            memory order. If `out` was passed, the same object is
            returned after being modified in-place to contain the
            appropriate values.
        "
  [ self order out ]
  (py/call-attr self "toarray"  self order out ))

(defn tobsr 
  "Convert this matrix to Block Sparse Row format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant bsr_matrix.

        When blocksize=(R, C) is provided, it will be used for construction of
        the bsr_matrix.
        "
  [self blocksize & {:keys [copy]
                       :or {copy false}} ]
    (py/call-attr-kw self "tobsr" [blocksize] {:copy copy }))

(defn tocoo 
  "Convert this matrix to COOrdinate format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant coo_matrix.
        "
  [self  & {:keys [copy]
                       :or {copy false}} ]
    (py/call-attr-kw self "tocoo" [] {:copy copy }))

(defn tocsc 
  "Convert this matrix to Compressed Sparse Column format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant csc_matrix.
        "
  [self  & {:keys [copy]
                       :or {copy false}} ]
    (py/call-attr-kw self "tocsc" [] {:copy copy }))

(defn tocsr 
  "Convert this matrix to Compressed Sparse Row format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant csr_matrix.
        "
  [self  & {:keys [copy]
                       :or {copy false}} ]
    (py/call-attr-kw self "tocsr" [] {:copy copy }))

(defn todense 
  "
        Return a dense matrix representation of this matrix.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to store multi-dimensional data in C (row-major)
            or Fortran (column-major) order in memory. The default
            is 'None', indicating the NumPy default of C-ordered.
            Cannot be specified in conjunction with the `out`
            argument.

        out : ndarray, 2-dimensional, optional
            If specified, uses this array (or `numpy.matrix`) as the
            output buffer instead of allocating a new array to
            return. The provided array must have the same shape and
            dtype as the sparse matrix on which you are calling the
            method.

        Returns
        -------
        arr : numpy.matrix, 2-dimensional
            A NumPy matrix object with the same shape and containing
            the same data represented by the sparse matrix, with the
            requested memory order. If `out` was passed and was an
            array (rather than a `numpy.matrix`), it will be filled
            with the appropriate values and returned wrapped in a
            `numpy.matrix` object that shares the same memory.
        "
  [ self order out ]
  (py/call-attr self "todense"  self order out ))

(defn todia 
  "Convert this matrix to sparse DIAgonal format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant dia_matrix.
        "
  [self  & {:keys [copy]
                       :or {copy false}} ]
    (py/call-attr-kw self "todia" [] {:copy copy }))

(defn todok 
  "Convert this matrix to Dictionary Of Keys format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant dok_matrix.
        "
  [self  & {:keys [copy]
                       :or {copy false}} ]
    (py/call-attr-kw self "todok" [] {:copy copy }))

(defn tolil 
  "Convert this matrix to LInked List format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant lil_matrix.
        "
  [self  & {:keys [copy]
                       :or {copy false}} ]
    (py/call-attr-kw self "tolil" [] {:copy copy }))

(defn transpose 
  "
        Reverses the dimensions of the sparse matrix.

        Parameters
        ----------
        axes : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value.
        copy : bool, optional
            Indicates whether or not attributes of `self` should be
            copied whenever possible. The degree to which attributes
            are copied varies depending on the type of sparse matrix
            being used.

        Returns
        -------
        p : `self` with the dimensions reversed.

        See Also
        --------
        numpy.matrix.transpose : NumPy's implementation of 'transpose'
                                 for matrices
        "
  [self axes & {:keys [copy]
                       :or {copy false}} ]
    (py/call-attr-kw self "transpose" [axes] {:copy copy }))