(ns sklearn.metrics.classification.coo-matrix
  "
    A sparse matrix in COOrdinate format.

    Also known as the 'ijv' or 'triplet' format.

    This can be instantiated in several ways:
        coo_matrix(D)
            with a dense matrix D

        coo_matrix(S)
            with another sparse matrix S (equivalent to S.tocoo())

        coo_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        coo_matrix((data, (i, j)), [shape=(M, N)])
            to construct from three arrays:
                1. data[:]   the entries of the matrix, in any order
                2. i[:]      the row indices of the matrix entries
                3. j[:]      the column indices of the matrix entries

            Where ``A[i[k], j[k]] = data[k]``.  When shape is not
            specified, it is inferred from the index arrays

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements
    data
        COO format data array of the matrix
    row
        COO format row index array of the matrix
    col
        COO format column index array of the matrix

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the COO format
        - facilitates fast conversion among sparse formats
        - permits duplicate entries (see example)
        - very fast conversion to and from CSR/CSC formats

    Disadvantages of the COO format
        - does not directly support:
            + arithmetic operations
            + slicing

    Intended Usage
        - COO is a fast format for constructing sparse matrices
        - Once a matrix has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - By default when converting to CSR or CSC format, duplicate (i,j)
          entries will be summed together.  This facilitates efficient
          construction of finite element matrices and the like. (see example)

    Examples
    --------

    >>> # Constructing an empty matrix
    >>> from scipy.sparse import coo_matrix
    >>> coo_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> # Constructing a matrix using ijv format
    >>> row  = np.array([0, 3, 1, 0])
    >>> col  = np.array([0, 3, 1, 2])
    >>> data = np.array([4, 5, 7, 9])
    >>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])

    >>> # Constructing a matrix with duplicate indices
    >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
    >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
    >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
    >>> coo = coo_matrix((data, (row, col)), shape=(4, 4))
    >>> # Duplicate indices are maintained until implicitly or explicitly summed
    >>> np.max(coo.data)
    1
    >>> coo.toarray()
    array([[3, 0, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce classification (import-module "sklearn.metrics.classification"))

(defn coo-matrix 
  "
    A sparse matrix in COOrdinate format.

    Also known as the 'ijv' or 'triplet' format.

    This can be instantiated in several ways:
        coo_matrix(D)
            with a dense matrix D

        coo_matrix(S)
            with another sparse matrix S (equivalent to S.tocoo())

        coo_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        coo_matrix((data, (i, j)), [shape=(M, N)])
            to construct from three arrays:
                1. data[:]   the entries of the matrix, in any order
                2. i[:]      the row indices of the matrix entries
                3. j[:]      the column indices of the matrix entries

            Where ``A[i[k], j[k]] = data[k]``.  When shape is not
            specified, it is inferred from the index arrays

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements
    data
        COO format data array of the matrix
    row
        COO format row index array of the matrix
    col
        COO format column index array of the matrix

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the COO format
        - facilitates fast conversion among sparse formats
        - permits duplicate entries (see example)
        - very fast conversion to and from CSR/CSC formats

    Disadvantages of the COO format
        - does not directly support:
            + arithmetic operations
            + slicing

    Intended Usage
        - COO is a fast format for constructing sparse matrices
        - Once a matrix has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - By default when converting to CSR or CSC format, duplicate (i,j)
          entries will be summed together.  This facilitates efficient
          construction of finite element matrices and the like. (see example)

    Examples
    --------

    >>> # Constructing an empty matrix
    >>> from scipy.sparse import coo_matrix
    >>> coo_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> # Constructing a matrix using ijv format
    >>> row  = np.array([0, 3, 1, 0])
    >>> col  = np.array([0, 3, 1, 2])
    >>> data = np.array([4, 5, 7, 9])
    >>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])

    >>> # Constructing a matrix with duplicate indices
    >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
    >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
    >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
    >>> coo = coo_matrix((data, (row, col)), shape=(4, 4))
    >>> # Duplicate indices are maintained until implicitly or explicitly summed
    >>> np.max(coo.data)
    1
    >>> coo.toarray()
    array([[3, 0, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])

    "
  [arg1 shape dtype & {:keys [copy]
                       :or {copy false}} ]
    (py/call-attr-kw classification "coo_matrix" [arg1 shape dtype] {:copy copy }))

(defn arcsin 
  "Element-wise arcsin.

See numpy.arcsin for more information."
  [ self  ]
  (py/call-attr self "arcsin"  self  ))

(defn arcsinh 
  "Element-wise arcsinh.

See numpy.arcsinh for more information."
  [ self  ]
  (py/call-attr self "arcsinh"  self  ))

(defn arctan 
  "Element-wise arctan.

See numpy.arctan for more information."
  [ self  ]
  (py/call-attr self "arctan"  self  ))

(defn arctanh 
  "Element-wise arctanh.

See numpy.arctanh for more information."
  [ self  ]
  (py/call-attr self "arctanh"  self  ))

(defn argmax 
  "Return indices of maximum elements along an axis.

        Implicit zero elements are also taken into account. If there are
        several maximum values, the index of the first occurrence is returned.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None}, optional
            Axis along which the argmax is computed. If None (default), index
            of the maximum element in the flatten data is returned.
        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
        ind : numpy.matrix or int
            Indices of maximum elements. If matrix, its size along `axis` is 1.
        "
  [ self axis out ]
  (py/call-attr self "argmax"  self axis out ))

(defn argmin 
  "Return indices of minimum elements along an axis.

        Implicit zero elements are also taken into account. If there are
        several minimum values, the index of the first occurrence is returned.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None}, optional
            Axis along which the argmin is computed. If None (default), index
            of the minimum element in the flatten data is returned.
        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
         ind : numpy.matrix or int
            Indices of minimum elements. If matrix, its size along `axis` is 1.
        "
  [ self axis out ]
  (py/call-attr self "argmin"  self axis out ))

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

(defn ceil 
  "Element-wise ceil.

See numpy.ceil for more information."
  [ self  ]
  (py/call-attr self "ceil"  self  ))

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

(defn deg2rad 
  "Element-wise deg2rad.

See numpy.deg2rad for more information."
  [ self  ]
  (py/call-attr self "deg2rad"  self  ))

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

(defn dtype 
  ""
  [ self ]
    (py/call-attr self "dtype"))

(defn eliminate-zeros 
  "Remove zero entries from the matrix

        This is an *in place* operation
        "
  [ self  ]
  (py/call-attr self "eliminate_zeros"  self  ))

(defn expm1 
  "Element-wise expm1.

See numpy.expm1 for more information."
  [ self  ]
  (py/call-attr self "expm1"  self  ))

(defn floor 
  "Element-wise floor.

See numpy.floor for more information."
  [ self  ]
  (py/call-attr self "floor"  self  ))

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

(defn log1p 
  "Element-wise log1p.

See numpy.log1p for more information."
  [ self  ]
  (py/call-attr self "log1p"  self  ))

(defn max 
  "
        Return the maximum of the matrix or maximum along an axis.
        This takes all elements into account, not just the non-zero ones.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the maximum over all the matrix elements, returning
            a scalar (i.e. `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value, as this argument is not used.

        Returns
        -------
        amax : coo_matrix or scalar
            Maximum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        min : The minimum value of a sparse matrix along a given axis.
        numpy.matrix.max : NumPy's implementation of 'max' for matrices

        "
  [ self axis out ]
  (py/call-attr self "max"  self axis out ))

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

(defn min 
  "
        Return the minimum of the matrix or maximum along an axis.
        This takes all elements into account, not just the non-zero ones.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the minimum over all the matrix elements, returning
            a scalar (i.e. `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
        amin : coo_matrix or scalar
            Minimum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        max : The maximum value of a sparse matrix along a given axis.
        numpy.matrix.min : NumPy's implementation of 'min' for matrices

        "
  [ self axis out ]
  (py/call-attr self "min"  self axis out ))

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
  "
        This function performs element-wise power.

        Parameters
        ----------
        n : n is a scalar

        dtype : If dtype is not specified, the current dtype will be preserved.
        "
  [ self n dtype ]
  (py/call-attr self "power"  self n dtype ))

(defn rad2deg 
  "Element-wise rad2deg.

See numpy.rad2deg for more information."
  [ self  ]
  (py/call-attr self "rad2deg"  self  ))

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
  [ self  ]
  (py/call-attr self "resize"  self  ))

(defn rint 
  "Element-wise rint.

See numpy.rint for more information."
  [ self  ]
  (py/call-attr self "rint"  self  ))

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

(defn sign 
  "Element-wise sign.

See numpy.sign for more information."
  [ self  ]
  (py/call-attr self "sign"  self  ))

(defn sin 
  "Element-wise sin.

See numpy.sin for more information."
  [ self  ]
  (py/call-attr self "sin"  self  ))

(defn sinh 
  "Element-wise sinh.

See numpy.sinh for more information."
  [ self  ]
  (py/call-attr self "sinh"  self  ))

(defn sqrt 
  "Element-wise sqrt.

See numpy.sqrt for more information."
  [ self  ]
  (py/call-attr self "sqrt"  self  ))

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

(defn sum-duplicates 
  "Eliminate duplicate matrix entries by adding them together

        This is an *in place* operation
        "
  [ self  ]
  (py/call-attr self "sum_duplicates"  self  ))

(defn tan 
  "Element-wise tan.

See numpy.tan for more information."
  [ self  ]
  (py/call-attr self "tan"  self  ))

(defn tanh 
  "Element-wise tanh.

See numpy.tanh for more information."
  [ self  ]
  (py/call-attr self "tanh"  self  ))

(defn toarray 
  "See the docstring for `spmatrix.toarray`."
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
  "Convert this matrix to Compressed Sparse Column format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_matrix
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsc()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        "
  [self  & {:keys [copy]
                       :or {copy false}} ]
    (py/call-attr-kw self "tocsc" [] {:copy copy }))

(defn tocsr 
  "Convert this matrix to Compressed Sparse Row format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_matrix
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsr()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

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

(defn trunc 
  "Element-wise trunc.

See numpy.trunc for more information."
  [ self  ]
  (py/call-attr self "trunc"  self  ))
