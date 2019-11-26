(ns sklearn.utils.fixes.MaskedArray
  "
    An array class with possibly masked values.

    Masked values of True exclude the corresponding element from any
    computation.

    Construction::

      x = MaskedArray(data, mask=nomask, dtype=None, copy=False, subok=True,
                      ndmin=0, fill_value=None, keep_mask=True, hard_mask=None,
                      shrink=True, order=None)

    Parameters
    ----------
    data : array_like
        Input data.
    mask : sequence, optional
        Mask. Must be convertible to an array of booleans with the same
        shape as `data`. True indicates a masked (i.e. invalid) data.
    dtype : dtype, optional
        Data type of the output.
        If `dtype` is None, the type of the data argument (``data.dtype``)
        is used. If `dtype` is not None and different from ``data.dtype``,
        a copy is performed.
    copy : bool, optional
        Whether to copy the input data (True), or to use a reference instead.
        Default is False.
    subok : bool, optional
        Whether to return a subclass of `MaskedArray` if possible (True) or a
        plain `MaskedArray`. Default is True.
    ndmin : int, optional
        Minimum number of dimensions. Default is 0.
    fill_value : scalar, optional
        Value used to fill in the masked values when necessary.
        If None, a default based on the data-type is used.
    keep_mask : bool, optional
        Whether to combine `mask` with the mask of the input data, if any
        (True), or to use only `mask` for the output (False). Default is True.
    hard_mask : bool, optional
        Whether to use a hard mask or not. With a hard mask, masked values
        cannot be unmasked. Default is False.
    shrink : bool, optional
        Whether to force compression of an empty mask. Default is True.
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C', then the array
        will be in C-contiguous order (last-index varies the fastest).
        If order is 'F', then the returned array will be in
        Fortran-contiguous order (first-index varies the fastest).
        If order is 'A' (default), then the returned array may be
        in any order (either C-, Fortran-contiguous, or even discontiguous),
        unless a copy is required, in which case it will be C-contiguous.

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce fixes (import-module "sklearn.utils.fixes"))

(defn MaskedArray 
  "
    An array class with possibly masked values.

    Masked values of True exclude the corresponding element from any
    computation.

    Construction::

      x = MaskedArray(data, mask=nomask, dtype=None, copy=False, subok=True,
                      ndmin=0, fill_value=None, keep_mask=True, hard_mask=None,
                      shrink=True, order=None)

    Parameters
    ----------
    data : array_like
        Input data.
    mask : sequence, optional
        Mask. Must be convertible to an array of booleans with the same
        shape as `data`. True indicates a masked (i.e. invalid) data.
    dtype : dtype, optional
        Data type of the output.
        If `dtype` is None, the type of the data argument (``data.dtype``)
        is used. If `dtype` is not None and different from ``data.dtype``,
        a copy is performed.
    copy : bool, optional
        Whether to copy the input data (True), or to use a reference instead.
        Default is False.
    subok : bool, optional
        Whether to return a subclass of `MaskedArray` if possible (True) or a
        plain `MaskedArray`. Default is True.
    ndmin : int, optional
        Minimum number of dimensions. Default is 0.
    fill_value : scalar, optional
        Value used to fill in the masked values when necessary.
        If None, a default based on the data-type is used.
    keep_mask : bool, optional
        Whether to combine `mask` with the mask of the input data, if any
        (True), or to use only `mask` for the output (False). Default is True.
    hard_mask : bool, optional
        Whether to use a hard mask or not. With a hard mask, masked values
        cannot be unmasked. Default is False.
    shrink : bool, optional
        Whether to force compression of an empty mask. Default is True.
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C', then the array
        will be in C-contiguous order (last-index varies the fastest).
        If order is 'F', then the returned array will be in
        Fortran-contiguous order (first-index varies the fastest).
        If order is 'A' (default), then the returned array may be
        in any order (either C-, Fortran-contiguous, or even discontiguous),
        unless a copy is required, in which case it will be C-contiguous.

    "
  [ & {:keys [data mask dtype copy subok ndmin fill_value keep_mask hard_mask shrink order]
       :or {mask False copy false subok true ndmin 0 keep_mask true shrink true}} ]
  
   (py/call-attr-kw fixes "MaskedArray" [] {:data data :mask mask :dtype dtype :copy copy :subok subok :ndmin ndmin :fill_value fill_value :keep_mask keep_mask :hard_mask hard_mask :shrink shrink :order order }))

(defn T 
  ""
  [ self ]
    (py/call-attr self "T"))

(defn all 
  "
        Returns True if all elements evaluate to True.

        The output array is masked where all the values along the given axis
        are masked: if the output would have been a scalar and that all the
        values are masked, then the output is `masked`.

        Refer to `numpy.all` for full documentation.

        See Also
        --------
        ndarray.all : corresponding function for ndarrays
        numpy.all : equivalent function

        Examples
        --------
        >>> np.ma.array([1,2,3]).all()
        True
        >>> a = np.ma.array([1,2,3], mask=True)
        >>> (a.all() is np.ma.masked)
        True

        "
  [self  & {:keys [axis out keepdims]
                       :or {keepdims <no value>}} ]
    (py/call-attr-kw self "all" [] {:axis axis :out out :keepdims keepdims }))
(defn anom 
  "
        Compute the anomalies (deviations from the arithmetic mean)
        along the given axis.

        Returns an array of anomalies, with the same shape as the input and
        where the arithmetic mean is computed along the given axis.

        Parameters
        ----------
        axis : int, optional
            Axis over which the anomalies are taken.
            The default is to use the mean of the flattened array as reference.
        dtype : dtype, optional
            Type to use in computing the variance. For arrays of integer type
             the default is float32; for arrays of float types it is the same as
             the array type.

        See Also
        --------
        mean : Compute the mean of the array.

        Examples
        --------
        >>> a = np.ma.array([1,2,3])
        >>> a.anom()
        masked_array(data=[-1.,  0.,  1.],
                     mask=False,
               fill_value=1e+20)

        "
  [self   & {:keys [axis dtype]} ]
    (py/call-attr-kw self "anom" [] {:axis axis :dtype dtype }))

(defn any 
  "
        Returns True if any of the elements of `a` evaluate to True.

        Masked values are considered as False during computation.

        Refer to `numpy.any` for full documentation.

        See Also
        --------
        ndarray.any : corresponding function for ndarrays
        numpy.any : equivalent function

        "
  [self  & {:keys [axis out keepdims]
                       :or {keepdims <no value>}} ]
    (py/call-attr-kw self "any" [] {:axis axis :out out :keepdims keepdims }))
(defn argmax 
  "
        Returns array of indices of the maximum values along the given axis.
        Masked values are treated as if they had the value fill_value.

        Parameters
        ----------
        axis : {None, integer}
            If None, the index is into the flattened array, otherwise along
            the specified axis
        fill_value : {var}, optional
            Value used to fill in the masked values.  If None, the output of
            maximum_fill_value(self._data) is used instead.
        out : {None, array}, optional
            Array into which the result can be placed. Its type is preserved
            and it must be of the right shape to hold the output.

        Returns
        -------
        index_array : {integer_array}

        Examples
        --------
        >>> a = np.arange(6).reshape(2,3)
        >>> a.argmax()
        5
        >>> a.argmax(0)
        array([1, 1, 1])
        >>> a.argmax(1)
        array([2, 2])

        "
  [self   & {:keys [axis fill_value out]} ]
    (py/call-attr-kw self "argmax" [] {:axis axis :fill_value fill_value :out out }))
(defn argmin 
  "
        Return array of indices to the minimum values along the given axis.

        Parameters
        ----------
        axis : {None, integer}
            If None, the index is into the flattened array, otherwise along
            the specified axis
        fill_value : {var}, optional
            Value used to fill in the masked values.  If None, the output of
            minimum_fill_value(self._data) is used instead.
        out : {None, array}, optional
            Array into which the result can be placed. Its type is preserved
            and it must be of the right shape to hold the output.

        Returns
        -------
        ndarray or scalar
            If multi-dimension input, returns a new ndarray of indices to the
            minimum values along the given axis.  Otherwise, returns a scalar
            of index to the minimum values along the given axis.

        Examples
        --------
        >>> x = np.ma.array(np.arange(4), mask=[1,1,0,0])
        >>> x.shape = (2,2)
        >>> x
        masked_array(
          data=[[--, --],
                [2, 3]],
          mask=[[ True,  True],
                [False, False]],
          fill_value=999999)
        >>> x.argmin(axis=0, fill_value=-1)
        array([0, 0])
        >>> x.argmin(axis=0, fill_value=9)
        array([1, 1])

        "
  [self   & {:keys [axis fill_value out]} ]
    (py/call-attr-kw self "argmin" [] {:axis axis :fill_value fill_value :out out }))

(defn argpartition 
  ""
  [ self  ]
  (py/call-attr self "argpartition"  self  ))

(defn argsort 
  "
        Return an ndarray of indices that sort the array along the
        specified axis.  Masked values are filled beforehand to
        `fill_value`.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. If None, the default, the flattened array
            is used.

            ..  versionchanged:: 1.13.0
                Previously, the default was documented to be -1, but that was
                in error. At some future date, the default will change to -1, as
                originally intended.
                Until then, the axis should be given explicitly when
                ``arr.ndim > 1``, to avoid a FutureWarning.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            The sorting algorithm used.
        order : list, optional
            When `a` is an array with fields defined, this argument specifies
            which fields to compare first, second, etc.  Not all fields need be
            specified.
        endwith : {True, False}, optional
            Whether missing values (if any) should be treated as the largest values
            (True) or the smallest values (False)
            When the array contains unmasked values at the same extremes of the
            datatype, the ordering of these values and the masked values is
            undefined.
        fill_value : {var}, optional
            Value used internally for the masked values.
            If ``fill_value`` is not None, it supersedes ``endwith``.

        Returns
        -------
        index_array : ndarray, int
            Array of indices that sort `a` along the specified axis.
            In other words, ``a[index_array]`` yields a sorted `a`.

        See Also
        --------
        MaskedArray.sort : Describes sorting algorithms used.
        lexsort : Indirect stable sort with multiple keys.
        ndarray.sort : Inplace sort.

        Notes
        -----
        See `sort` for notes on the different sorting algorithms.

        Examples
        --------
        >>> a = np.ma.array([3,2,1], mask=[False, False, True])
        >>> a
        masked_array(data=[3, 2, --],
                     mask=[False, False,  True],
               fill_value=999999)
        >>> a.argsort()
        array([1, 0, 2])

        "
  [self  & {:keys [axis kind order endwith fill_value]
                       :or {axis <no value> endwith true}} ]
    (py/call-attr-kw self "argsort" [] {:axis axis :kind kind :order order :endwith endwith :fill_value fill_value }))

(defn baseclass 
  " Class of the underlying data (read-only). "
  [ self ]
    (py/call-attr self "baseclass"))
(defn compress 
  "
        Return `a` where condition is ``True``.

        If condition is a `MaskedArray`, missing values are considered
        as ``False``.

        Parameters
        ----------
        condition : var
            Boolean 1-d array selecting which entries to return. If len(condition)
            is less than the size of a along the axis, then output is truncated
            to length of condition array.
        axis : {None, int}, optional
            Axis along which the operation must be performed.
        out : {None, ndarray}, optional
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary.

        Returns
        -------
        result : MaskedArray
            A :class:`MaskedArray` object.

        Notes
        -----
        Please note the difference with :meth:`compressed` !
        The output of :meth:`compress` has a mask, the output of
        :meth:`compressed` does not.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.compress([1, 0, 1])
        masked_array(data=[1, 3],
                     mask=[False, False],
               fill_value=999999)

        >>> x.compress([1, 0, 1], axis=1)
        masked_array(
          data=[[1, 3],
                [--, --],
                [7, 9]],
          mask=[[False, False],
                [ True,  True],
                [False, False]],
          fill_value=999999)

        "
  [self condition  & {:keys [axis out]} ]
    (py/call-attr-kw self "compress" [condition] {:axis axis :out out }))

(defn compressed 
  "
        Return all the non-masked data as a 1-D array.

        Returns
        -------
        data : ndarray
            A new `ndarray` holding the non-masked data is returned.

        Notes
        -----
        The result is **not** a MaskedArray!

        Examples
        --------
        >>> x = np.ma.array(np.arange(5), mask=[0]*2 + [1]*3)
        >>> x.compressed()
        array([0, 1])
        >>> type(x.compressed())
        <class 'numpy.ndarray'>

        "
  [ self  ]
  (py/call-attr self "compressed"  self  ))

(defn copy 
  "a.copy(order='C')

    Return a copy of the array.

    Parameters
    ----------
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the copy. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible. (Note that this function and :func:`numpy.copy` are very
        similar, but have different default values for their order=
        arguments.)

    See also
    --------
    numpy.copy
    numpy.copyto

    Examples
    --------
    >>> x = np.array([[1,2,3],[4,5,6]], order='F')

    >>> y = x.copy()

    >>> x.fill(0)

    >>> x
    array([[0, 0, 0],
           [0, 0, 0]])

    >>> y
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> y.flags['C_CONTIGUOUS']
    True"
  [ self  ]
  (py/call-attr self "copy"  self  ))

(defn count 
  "
        Count the non-masked elements of the array along the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which the count is performed.
            The default (`axis` = `None`) performs the count over all
            the dimensions of the input array. `axis` may be negative, in
            which case it counts from the last to the first axis.

            .. versionadded:: 1.10.0

            If this is a tuple of ints, the count is performed on multiple
            axes, instead of a single axis or all the axes as before.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the array.

        Returns
        -------
        result : ndarray or scalar
            An array with the same shape as the input array, with the specified
            axis removed. If the array is a 0-d array, or if `axis` is None, a
            scalar is returned.

        See Also
        --------
        count_masked : Count masked elements in array or along a given axis.

        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = ma.arange(6).reshape((2, 3))
        >>> a[1, :] = ma.masked
        >>> a
        masked_array(
          data=[[0, 1, 2],
                [--, --, --]],
          mask=[[False, False, False],
                [ True,  True,  True]],
          fill_value=999999)
        >>> a.count()
        3

        When the `axis` keyword is specified an array of appropriate size is
        returned.

        >>> a.count(axis=0)
        array([1, 1, 1])
        >>> a.count(axis=1)
        array([3, 0])

        "
  [self  & {:keys [axis keepdims]
                       :or {keepdims <no value>}} ]
    (py/call-attr-kw self "count" [] {:axis axis :keepdims keepdims }))
(defn cumprod 
  "
        Return the cumulative product of the array elements over the given axis.

        Masked values are set to 1 internally during the computation.
        However, their position is saved, and the result will be masked at
        the same locations.

        Refer to `numpy.cumprod` for full documentation.

        Notes
        -----
        The mask is lost if `out` is not a valid MaskedArray !

        Arithmetic is modular when using integer types, and no error is
        raised on overflow.

        See Also
        --------
        ndarray.cumprod : corresponding function for ndarrays
        numpy.cumprod : equivalent function
        "
  [self   & {:keys [axis dtype out]} ]
    (py/call-attr-kw self "cumprod" [] {:axis axis :dtype dtype :out out }))
(defn cumsum 
  "
        Return the cumulative sum of the array elements over the given axis.

        Masked values are set to 0 internally during the computation.
        However, their position is saved, and the result will be masked at
        the same locations.

        Refer to `numpy.cumsum` for full documentation.

        Notes
        -----
        The mask is lost if `out` is not a valid :class:`MaskedArray` !

        Arithmetic is modular when using integer types, and no error is
        raised on overflow.

        See Also
        --------
        ndarray.cumsum : corresponding function for ndarrays
        numpy.cumsum : equivalent function

        Examples
        --------
        >>> marr = np.ma.array(np.arange(10), mask=[0,0,0,1,1,1,0,0,0,0])
        >>> marr.cumsum()
        masked_array(data=[0, 1, 3, --, --, --, 9, 16, 24, 33],
                     mask=[False, False, False,  True,  True,  True, False, False,
                           False, False],
               fill_value=999999)

        "
  [self   & {:keys [axis dtype out]} ]
    (py/call-attr-kw self "cumsum" [] {:axis axis :dtype dtype :out out }))

(defn data 
  "
        Returns the underlying data, as a view of the masked array.

        If the underlying data is a subclass of :class:`numpy.ndarray`, it is
        returned as such.

        >>> x = np.ma.array(np.matrix([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
        >>> x.data
        matrix([[1, 2],
                [3, 4]])

        The type of the data can be accessed through the :attr:`baseclass`
        attribute.
        "
  [ self ]
    (py/call-attr self "data"))

(defn diagonal 
  "a.diagonal(offset=0, axis1=0, axis2=1)

    Return specified diagonals. In NumPy 1.9 the returned array is a
    read-only view instead of a copy as in previous NumPy versions.  In
    a future version the read-only restriction will be removed.

    Refer to :func:`numpy.diagonal` for full documentation.

    See Also
    --------
    numpy.diagonal : equivalent function"
  [ self  ]
  (py/call-attr self "diagonal"  self  ))

(defn dot 
  "
        a.dot(b, out=None)

        Masked dot product of two arrays. Note that `out` and `strict` are
        located in different positions than in `ma.dot`. In order to
        maintain compatibility with the functional version, it is
        recommended that the optional arguments be treated as keyword only.
        At some point that may be mandatory.

        .. versionadded:: 1.10.0

        Parameters
        ----------
        b : masked_array_like
            Inputs array.
        out : masked_array, optional
            Output argument. This must have the exact kind that would be
            returned if it was not used. In particular, it must have the
            right type, must be C-contiguous, and its dtype must be the
            dtype that would be returned for `ma.dot(a,b)`. This is a
            performance feature. Therefore, if these conditions are not
            met, an exception is raised, instead of attempting to be
            flexible.
        strict : bool, optional
            Whether masked data are propagated (True) or set to 0 (False)
            for the computation. Default is False.  Propagating the mask
            means that if a masked value appears in a row or column, the
            whole row or column is considered masked.

            .. versionadded:: 1.10.2

        See Also
        --------
        numpy.ma.dot : equivalent function

        "
  [self b & {:keys [out strict]
                       :or {strict false}} ]
    (py/call-attr-kw self "dot" [b] {:out out :strict strict }))

(defn dtype 
  ""
  [ self ]
    (py/call-attr self "dtype"))

(defn fill-value 
  "
        The filling value of the masked array is a scalar. When setting, None
        will set to a default based on the data type.

        Examples
        --------
        >>> for dt in [np.int32, np.int64, np.float64, np.complex128]:
        ...     np.ma.array([0, 1], dtype=dt).get_fill_value()
        ...
        999999
        999999
        1e+20
        (1e+20+0j)

        >>> x = np.ma.array([0, 1.], fill_value=-np.inf)
        >>> x.fill_value
        -inf
        >>> x.fill_value = np.pi
        >>> x.fill_value
        3.1415926535897931 # may vary

        Reset to default:

        >>> x.fill_value = None
        >>> x.fill_value
        1e+20

        "
  [ self ]
    (py/call-attr self "fill_value"))
(defn filled 
  "
        Return a copy of self, with masked values filled with a given value.
        **However**, if there are no masked values to fill, self will be
        returned instead as an ndarray.

        Parameters
        ----------
        fill_value : scalar, optional
            The value to use for invalid entries (None by default).
            If None, the `fill_value` attribute of the array is used instead.

        Returns
        -------
        filled_array : ndarray
            A copy of ``self`` with invalid entries replaced by *fill_value*
            (be it the function argument or the attribute of ``self``), or
            ``self`` itself as an ndarray if there are no invalid entries to
            be replaced.

        Notes
        -----
        The result is **not** a MaskedArray!

        Examples
        --------
        >>> x = np.ma.array([1,2,3,4,5], mask=[0,0,1,0,1], fill_value=-999)
        >>> x.filled()
        array([   1,    2, -999,    4, -999])
        >>> type(x.filled())
        <class 'numpy.ndarray'>

        Subclassing is preserved. This means that if, e.g., the data part of
        the masked array is a recarray, `filled` returns a recarray:

        >>> x = np.array([(-1, 2), (-3, 4)], dtype='i8,i8').view(np.recarray)
        >>> m = np.ma.array(x, mask=[(True, False), (False, True)])
        >>> m.filled()
        rec.array([(999999,      2), (    -3, 999999)],
                  dtype=[('f0', '<i8'), ('f1', '<i8')])
        "
  [self   & {:keys [fill_value]} ]
    (py/call-attr-kw self "filled" [] {:fill_value fill_value }))

(defn flat 
  " Return a flat iterator, or set a flattened version of self to value. "
  [ self ]
    (py/call-attr self "flat"))

(defn flatten 
  "a.flatten(order='C')

    Return a copy of the array collapsed into one dimension.

    Parameters
    ----------
    order : {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-
        style) order. 'A' means to flatten in column-major
        order if `a` is Fortran *contiguous* in memory,
        row-major order otherwise. 'K' means to flatten
        `a` in the order the elements occur in memory.
        The default is 'C'.

    Returns
    -------
    y : ndarray
        A copy of the input array, flattened to one dimension.

    See Also
    --------
    ravel : Return a flattened array.
    flat : A 1-D flat iterator over the array.

    Examples
    --------
    >>> a = np.array([[1,2], [3,4]])
    >>> a.flatten()
    array([1, 2, 3, 4])
    >>> a.flatten('F')
    array([1, 3, 2, 4])"
  [ self  ]
  (py/call-attr self "flatten"  self  ))

(defn get-fill-value 
  "
        The filling value of the masked array is a scalar. When setting, None
        will set to a default based on the data type.

        Examples
        --------
        >>> for dt in [np.int32, np.int64, np.float64, np.complex128]:
        ...     np.ma.array([0, 1], dtype=dt).get_fill_value()
        ...
        999999
        999999
        1e+20
        (1e+20+0j)

        >>> x = np.ma.array([0, 1.], fill_value=-np.inf)
        >>> x.fill_value
        -inf
        >>> x.fill_value = np.pi
        >>> x.fill_value
        3.1415926535897931 # may vary

        Reset to default:

        >>> x.fill_value = None
        >>> x.fill_value
        1e+20

        "
  [ self  ]
  (py/call-attr self "get_fill_value"  self  ))

(defn get-imag 
  "
        The imaginary part of the masked array.

        This property is a view on the imaginary part of this `MaskedArray`.

        See Also
        --------
        real

        Examples
        --------
        >>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
        >>> x.imag
        masked_array(data=[1.0, --, 1.6],
                     mask=[False,  True, False],
               fill_value=1e+20)

        "
  [ self  ]
  (py/call-attr self "get_imag"  self  ))

(defn get-real 
  "
        The real part of the masked array.

        This property is a view on the real part of this `MaskedArray`.

        See Also
        --------
        imag

        Examples
        --------
        >>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
        >>> x.real
        masked_array(data=[1.0, --, 3.45],
                     mask=[False,  True, False],
               fill_value=1e+20)

        "
  [ self  ]
  (py/call-attr self "get_real"  self  ))

(defn harden-mask 
  "
        Force the mask to hard.

        Whether the mask of a masked array is hard or soft is determined by
        its `hardmask` property. `harden_mask` sets `hardmask` to True.

        See Also
        --------
        hardmask

        "
  [ self  ]
  (py/call-attr self "harden_mask"  self  ))

(defn hardmask 
  " Hardness of the mask "
  [ self ]
    (py/call-attr self "hardmask"))

(defn ids 
  "
        Return the addresses of the data and mask areas.

        Parameters
        ----------
        None

        Examples
        --------
        >>> x = np.ma.array([1, 2, 3], mask=[0, 1, 1])
        >>> x.ids()
        (166670640, 166659832) # may vary

        If the array has no mask, the address of `nomask` is returned. This address
        is typically not close to the data in memory:

        >>> x = np.ma.array([1, 2, 3])
        >>> x.ids()
        (166691080, 3083169284L) # may vary

        "
  [ self  ]
  (py/call-attr self "ids"  self  ))

(defn imag 
  "
        The imaginary part of the masked array.

        This property is a view on the imaginary part of this `MaskedArray`.

        See Also
        --------
        real

        Examples
        --------
        >>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
        >>> x.imag
        masked_array(data=[1.0, --, 1.6],
                     mask=[False,  True, False],
               fill_value=1e+20)

        "
  [ self ]
    (py/call-attr self "imag"))

(defn iscontiguous 
  "
        Return a boolean indicating whether the data is contiguous.

        Parameters
        ----------
        None

        Examples
        --------
        >>> x = np.ma.array([1, 2, 3])
        >>> x.iscontiguous()
        True

        `iscontiguous` returns one of the flags of the masked array:

        >>> x.flags
          C_CONTIGUOUS : True
          F_CONTIGUOUS : True
          OWNDATA : False
          WRITEABLE : True
          ALIGNED : True
          WRITEBACKIFCOPY : False
          UPDATEIFCOPY : False

        "
  [ self  ]
  (py/call-attr self "iscontiguous"  self  ))

(defn mask 
  " Current mask. "
  [ self ]
    (py/call-attr self "mask"))

(defn max 
  "
        Return the maximum along a given axis.

        Parameters
        ----------
        axis : {None, int}, optional
            Axis along which to operate.  By default, ``axis`` is None and the
            flattened input is used.
        out : array_like, optional
            Alternative output array in which to place the result.  Must
            be of the same shape and buffer length as the expected output.
        fill_value : {var}, optional
            Value used to fill in the masked values.
            If None, use the output of maximum_fill_value().

        Returns
        -------
        amax : array_like
            New array holding the result.
            If ``out`` was specified, ``out`` is returned.

        See Also
        --------
        maximum_fill_value
            Returns the maximum filling value for a given datatype.

        "
  [self  & {:keys [axis out fill_value keepdims]
                       :or {keepdims <no value>}} ]
    (py/call-attr-kw self "max" [] {:axis axis :out out :fill_value fill_value :keepdims keepdims }))

(defn mean 
  "
        Returns the average of the array elements along given axis.

        Masked entries are ignored, and result elements which are not
        finite will be masked.

        Refer to `numpy.mean` for full documentation.

        See Also
        --------
        ndarray.mean : corresponding function for ndarrays
        numpy.mean : Equivalent function
        numpy.ma.average: Weighted average.

        Examples
        --------
        >>> a = np.ma.array([1,2,3], mask=[False, False, True])
        >>> a
        masked_array(data=[1, 2, --],
                     mask=[False, False,  True],
               fill_value=999999)
        >>> a.mean()
        1.5

        "
  [self  & {:keys [axis dtype out keepdims]
                       :or {keepdims <no value>}} ]
    (py/call-attr-kw self "mean" [] {:axis axis :dtype dtype :out out :keepdims keepdims }))

(defn min 
  "
        Return the minimum along a given axis.

        Parameters
        ----------
        axis : {None, int}, optional
            Axis along which to operate.  By default, ``axis`` is None and the
            flattened input is used.
        out : array_like, optional
            Alternative output array in which to place the result.  Must be of
            the same shape and buffer length as the expected output.
        fill_value : {var}, optional
            Value used to fill in the masked values.
            If None, use the output of `minimum_fill_value`.

        Returns
        -------
        amin : array_like
            New array holding the result.
            If ``out`` was specified, ``out`` is returned.

        See Also
        --------
        minimum_fill_value
            Returns the minimum filling value for a given datatype.

        "
  [self  & {:keys [axis out fill_value keepdims]
                       :or {keepdims <no value>}} ]
    (py/call-attr-kw self "min" [] {:axis axis :out out :fill_value fill_value :keepdims keepdims }))
(defn mini 
  "
        Return the array minimum along the specified axis.

        .. deprecated:: 1.13.0
           This function is identical to both:

            * ``self.min(keepdims=True, axis=axis).squeeze(axis=axis)``
            * ``np.ma.minimum.reduce(self, axis=axis)``

           Typically though, ``self.min(axis=axis)`` is sufficient.

        Parameters
        ----------
        axis : int, optional
            The axis along which to find the minima. Default is None, in which case
            the minimum value in the whole array is returned.

        Returns
        -------
        min : scalar or MaskedArray
            If `axis` is None, the result is a scalar. Otherwise, if `axis` is
            given and the array is at least 2-D, the result is a masked array with
            dimension one smaller than the array on which `mini` is called.

        Examples
        --------
        >>> x = np.ma.array(np.arange(6), mask=[0 ,1, 0, 0, 0 ,1]).reshape(3, 2)
        >>> x
        masked_array(
          data=[[0, --],
                [2, 3],
                [4, --]],
          mask=[[False,  True],
                [False, False],
                [False,  True]],
          fill_value=999999)
        >>> x.mini()
        masked_array(data=0,
                     mask=False,
               fill_value=999999)
        >>> x.mini(axis=0)
        masked_array(data=[0, 3],
                     mask=[False, False],
               fill_value=999999)
        >>> x.mini(axis=1)
        masked_array(data=[0, 2, 4],
                     mask=[False, False, False],
               fill_value=999999)

        There is a small difference between `mini` and `min`:

        >>> x[:,1].mini(axis=0)
        masked_array(data=3,
                     mask=False,
               fill_value=999999)
        >>> x[:,1].min(axis=0)
        3
        "
  [self   & {:keys [axis]} ]
    (py/call-attr-kw self "mini" [] {:axis axis }))

(defn nonzero 
  "
        Return the indices of unmasked elements that are not zero.

        Returns a tuple of arrays, one for each dimension, containing the
        indices of the non-zero elements in that dimension. The corresponding
        non-zero values can be obtained with::

            a[a.nonzero()]

        To group the indices by element, rather than dimension, use
        instead::

            np.transpose(a.nonzero())

        The result of this is always a 2d array, with a row for each non-zero
        element.

        Parameters
        ----------
        None

        Returns
        -------
        tuple_of_arrays : tuple
            Indices of elements that are non-zero.

        See Also
        --------
        numpy.nonzero :
            Function operating on ndarrays.
        flatnonzero :
            Return indices that are non-zero in the flattened version of the input
            array.
        ndarray.nonzero :
            Equivalent ndarray method.
        count_nonzero :
            Counts the number of non-zero elements in the input array.

        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = ma.array(np.eye(3))
        >>> x
        masked_array(
          data=[[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]],
          mask=False,
          fill_value=1e+20)
        >>> x.nonzero()
        (array([0, 1, 2]), array([0, 1, 2]))

        Masked elements are ignored.

        >>> x[1, 1] = ma.masked
        >>> x
        masked_array(
          data=[[1.0, 0.0, 0.0],
                [0.0, --, 0.0],
                [0.0, 0.0, 1.0]],
          mask=[[False, False, False],
                [False,  True, False],
                [False, False, False]],
          fill_value=1e+20)
        >>> x.nonzero()
        (array([0, 2]), array([0, 2]))

        Indices can also be grouped by element.

        >>> np.transpose(x.nonzero())
        array([[0, 0],
               [2, 2]])

        A common use for ``nonzero`` is to find the indices of an array, where
        a condition is True.  Given an array `a`, the condition `a` > 3 is a
        boolean array and since False is interpreted as 0, ma.nonzero(a > 3)
        yields the indices of the `a` where the condition is true.

        >>> a = ma.array([[1,2,3],[4,5,6],[7,8,9]])
        >>> a > 3
        masked_array(
          data=[[False, False, False],
                [ True,  True,  True],
                [ True,  True,  True]],
          mask=False,
          fill_value=True)
        >>> ma.nonzero(a > 3)
        (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

        The ``nonzero`` method of the condition array can also be called.

        >>> (a > 3).nonzero()
        (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

        "
  [ self  ]
  (py/call-attr self "nonzero"  self  ))

(defn partition 
  ""
  [ self  ]
  (py/call-attr self "partition"  self  ))

(defn prod 
  "
        Return the product of the array elements over the given axis.

        Masked elements are set to 1 internally for computation.

        Refer to `numpy.prod` for full documentation.

        Notes
        -----
        Arithmetic is modular when using integer types, and no error is raised
        on overflow.

        See Also
        --------
        ndarray.prod : corresponding function for ndarrays
        numpy.prod : equivalent function
        "
  [self  & {:keys [axis dtype out keepdims]
                       :or {keepdims <no value>}} ]
    (py/call-attr-kw self "prod" [] {:axis axis :dtype dtype :out out :keepdims keepdims }))

(defn product 
  "
        Return the product of the array elements over the given axis.

        Masked elements are set to 1 internally for computation.

        Refer to `numpy.prod` for full documentation.

        Notes
        -----
        Arithmetic is modular when using integer types, and no error is raised
        on overflow.

        See Also
        --------
        ndarray.prod : corresponding function for ndarrays
        numpy.prod : equivalent function
        "
  [self  & {:keys [axis dtype out keepdims]
                       :or {keepdims <no value>}} ]
    (py/call-attr-kw self "product" [] {:axis axis :dtype dtype :out out :keepdims keepdims }))

(defn ptp 
  "
        Return (maximum - minimum) along the given dimension
        (i.e. peak-to-peak value).

        Parameters
        ----------
        axis : {None, int}, optional
            Axis along which to find the peaks.  If None (default) the
            flattened array is used.
        out : {None, array_like}, optional
            Alternative output array in which to place the result. It must
            have the same shape and buffer length as the expected output
            but the type will be cast if necessary.
        fill_value : {var}, optional
            Value used to fill in the masked values.

        Returns
        -------
        ptp : ndarray.
            A new array holding the result, unless ``out`` was
            specified, in which case a reference to ``out`` is returned.

        "
  [self  & {:keys [axis out fill_value keepdims]
                       :or {keepdims false}} ]
    (py/call-attr-kw self "ptp" [] {:axis axis :out out :fill_value fill_value :keepdims keepdims }))

(defn put 
  "
        Set storage-indexed locations to corresponding values.

        Sets self._data.flat[n] = values[n] for each n in indices.
        If `values` is shorter than `indices` then it will repeat.
        If `values` has some masked values, the initial mask is updated
        in consequence, else the corresponding values are unmasked.

        Parameters
        ----------
        indices : 1-D array_like
            Target indices, interpreted as integers.
        values : array_like
            Values to place in self._data copy at target indices.
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how out-of-bounds indices will behave.
            'raise' : raise an error.
            'wrap' : wrap around.
            'clip' : clip to the range.

        Notes
        -----
        `values` can be a scalar or length 1 array.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.put([0,4,8],[10,20,30])
        >>> x
        masked_array(
          data=[[10, --, 3],
                [--, 20, --],
                [7, --, 30]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)

        >>> x.put(4,999)
        >>> x
        masked_array(
          data=[[10, --, 3],
                [--, 999, --],
                [7, --, 30]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)

        "
  [self indices values & {:keys [mode]
                       :or {mode "raise"}} ]
    (py/call-attr-kw self "put" [indices values] {:mode mode }))

(defn ravel 
  "
        Returns a 1D version of self, as a view.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional
            The elements of `a` are read using this index order. 'C' means to
            index the elements in C-like order, with the last axis index
            changing fastest, back to the first axis index changing slowest.
            'F' means to index the elements in Fortran-like index order, with
            the first index changing fastest, and the last index changing
            slowest. Note that the 'C' and 'F' options take no account of the
            memory layout of the underlying array, and only refer to the order
            of axis indexing.  'A' means to read the elements in Fortran-like
            index order if `m` is Fortran *contiguous* in memory, C-like order
            otherwise.  'K' means to read the elements in the order they occur
            in memory, except for reversing the data when strides are negative.
            By default, 'C' index order is used.

        Returns
        -------
        MaskedArray
            Output view is of shape ``(self.size,)`` (or
            ``(np.ma.product(self.shape),)``).

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.ravel()
        masked_array(data=[1, --, 3, --, 5, --, 7, --, 9],
                     mask=[False,  True, False,  True, False,  True, False,  True,
                           False],
               fill_value=999999)

        "
  [self  & {:keys [order]
                       :or {order "C"}} ]
    (py/call-attr-kw self "ravel" [] {:order order }))

(defn real 
  "
        The real part of the masked array.

        This property is a view on the real part of this `MaskedArray`.

        See Also
        --------
        imag

        Examples
        --------
        >>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
        >>> x.real
        masked_array(data=[1.0, --, 3.45],
                     mask=[False,  True, False],
               fill_value=1e+20)

        "
  [ self ]
    (py/call-attr self "real"))

(defn recordmask 
  "
        Get or set the mask of the array if it has no named fields. For
        structured arrays, returns a ndarray of booleans where entries are
        ``True`` if **all** the fields are masked, ``False`` otherwise:

        >>> x = np.ma.array([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
        ...         mask=[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
        ...        dtype=[('a', int), ('b', int)])
        >>> x.recordmask
        array([False, False,  True, False, False])
        "
  [ self ]
    (py/call-attr self "recordmask"))

(defn repeat 
  "a.repeat(repeats, axis=None)

    Repeat elements of an array.

    Refer to `numpy.repeat` for full documentation.

    See Also
    --------
    numpy.repeat : equivalent function"
  [ self  ]
  (py/call-attr self "repeat"  self  ))

(defn reshape 
  "
        Give a new shape to the array without changing its data.

        Returns a masked array containing the same data, but with a new shape.
        The result is a view on the original array; if this is not possible, a
        ValueError is raised.

        Parameters
        ----------
        shape : int or tuple of ints
            The new shape should be compatible with the original shape. If an
            integer is supplied, then the result will be a 1-D array of that
            length.
        order : {'C', 'F'}, optional
            Determines whether the array data should be viewed as in C
            (row-major) or FORTRAN (column-major) order.

        Returns
        -------
        reshaped_array : array
            A new view on the array.

        See Also
        --------
        reshape : Equivalent function in the masked array module.
        numpy.ndarray.reshape : Equivalent method on ndarray object.
        numpy.reshape : Equivalent function in the NumPy module.

        Notes
        -----
        The reshaping operation cannot guarantee that a copy will not be made,
        to modify the shape in place, use ``a.shape = s``

        Examples
        --------
        >>> x = np.ma.array([[1,2],[3,4]], mask=[1,0,0,1])
        >>> x
        masked_array(
          data=[[--, 2],
                [3, --]],
          mask=[[ True, False],
                [False,  True]],
          fill_value=999999)
        >>> x = x.reshape((4,1))
        >>> x
        masked_array(
          data=[[--],
                [2],
                [3],
                [--]],
          mask=[[ True],
                [False],
                [False],
                [ True]],
          fill_value=999999)

        "
  [ self  ]
  (py/call-attr self "reshape"  self  ))

(defn resize 
  "
        .. warning::

            This method does nothing, except raise a ValueError exception. A
            masked array does not own its data and therefore cannot safely be
            resized in place. Use the `numpy.ma.resize` function instead.

        This method is difficult to implement safely and may be deprecated in
        future releases of NumPy.

        "
  [self newshape & {:keys [refcheck order]
                       :or {refcheck true order false}} ]
    (py/call-attr-kw self "resize" [newshape] {:refcheck refcheck :order order }))

(defn round 
  "
        Return each element rounded to the given number of decimals.

        Refer to `numpy.around` for full documentation.

        See Also
        --------
        ndarray.around : corresponding function for ndarrays
        numpy.around : equivalent function
        "
  [self  & {:keys [decimals out]
                       :or {decimals 0}} ]
    (py/call-attr-kw self "round" [] {:decimals decimals :out out }))
(defn set-fill-value 
  ""
  [self   & {:keys [value]} ]
    (py/call-attr-kw self "set_fill_value" [] {:value value }))

(defn shape 
  ""
  [ self ]
    (py/call-attr self "shape"))

(defn sharedmask 
  " Share status of the mask (read-only). "
  [ self ]
    (py/call-attr self "sharedmask"))

(defn shrink-mask 
  "
        Reduce a mask to nomask when possible.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        >>> x = np.ma.array([[1,2 ], [3, 4]], mask=[0]*4)
        >>> x.mask
        array([[False, False],
               [False, False]])
        >>> x.shrink_mask()
        masked_array(
          data=[[1, 2],
                [3, 4]],
          mask=False,
          fill_value=999999)
        >>> x.mask
        False

        "
  [ self  ]
  (py/call-attr self "shrink_mask"  self  ))

(defn soften-mask 
  "
        Force the mask to soft.

        Whether the mask of a masked array is hard or soft is determined by
        its `hardmask` property. `soften_mask` sets `hardmask` to False.

        See Also
        --------
        hardmask

        "
  [ self  ]
  (py/call-attr self "soften_mask"  self  ))

(defn sort 
  "
        Sort the array, in-place

        Parameters
        ----------
        a : array_like
            Array to be sorted.
        axis : int, optional
            Axis along which to sort. If None, the array is flattened before
            sorting. The default is -1, which sorts along the last axis.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            The sorting algorithm used.
        order : list, optional
            When `a` is a structured array, this argument specifies which fields
            to compare first, second, and so on.  This list does not need to
            include all of the fields.
        endwith : {True, False}, optional
            Whether missing values (if any) should be treated as the largest values
            (True) or the smallest values (False)
            When the array contains unmasked values sorting at the same extremes of the
            datatype, the ordering of these values and the masked values is
            undefined.
        fill_value : {var}, optional
            Value used internally for the masked values.
            If ``fill_value`` is not None, it supersedes ``endwith``.

        Returns
        -------
        sorted_array : ndarray
            Array of the same type and shape as `a`.

        See Also
        --------
        ndarray.sort : Method to sort an array in-place.
        argsort : Indirect sort.
        lexsort : Indirect stable sort on multiple keys.
        searchsorted : Find elements in a sorted array.

        Notes
        -----
        See ``sort`` for notes on the different sorting algorithms.

        Examples
        --------
        >>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
        >>> # Default
        >>> a.sort()
        >>> a
        masked_array(data=[1, 3, 5, --, --],
                     mask=[False, False, False,  True,  True],
               fill_value=999999)

        >>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
        >>> # Put missing values in the front
        >>> a.sort(endwith=False)
        >>> a
        masked_array(data=[--, --, 1, 3, 5],
                     mask=[ True,  True, False, False, False],
               fill_value=999999)

        >>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
        >>> # fill_value takes over endwith
        >>> a.sort(endwith=False, fill_value=3)
        >>> a
        masked_array(data=[1, --, --, 3, 5],
                     mask=[False,  True,  True, False, False],
               fill_value=999999)

        "
  [self  & {:keys [axis kind order endwith fill_value]
                       :or {axis -1 endwith true}} ]
    (py/call-attr-kw self "sort" [] {:axis axis :kind kind :order order :endwith endwith :fill_value fill_value }))

(defn squeeze 
  "a.squeeze(axis=None)

    Remove single-dimensional entries from the shape of `a`.

    Refer to `numpy.squeeze` for full documentation.

    See Also
    --------
    numpy.squeeze : equivalent function"
  [ self  ]
  (py/call-attr self "squeeze"  self  ))

(defn std 
  "
        Returns the standard deviation of the array elements along given axis.

        Masked entries are ignored.

        Refer to `numpy.std` for full documentation.

        See Also
        --------
        ndarray.std : corresponding function for ndarrays
        numpy.std : Equivalent function
        "
  [self  & {:keys [axis dtype out ddof keepdims]
                       :or {ddof 0 keepdims <no value>}} ]
    (py/call-attr-kw self "std" [] {:axis axis :dtype dtype :out out :ddof ddof :keepdims keepdims }))

(defn sum 
  "
        Return the sum of the array elements over the given axis.

        Masked elements are set to 0 internally.

        Refer to `numpy.sum` for full documentation.

        See Also
        --------
        ndarray.sum : corresponding function for ndarrays
        numpy.sum : equivalent function

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.sum()
        25
        >>> x.sum(axis=1)
        masked_array(data=[4, 5, 16],
                     mask=[False, False, False],
               fill_value=999999)
        >>> x.sum(axis=0)
        masked_array(data=[8, 5, 12],
                     mask=[False, False, False],
               fill_value=999999)
        >>> print(type(x.sum(axis=0, dtype=np.int64)[0]))
        <class 'numpy.int64'>

        "
  [self  & {:keys [axis dtype out keepdims]
                       :or {keepdims <no value>}} ]
    (py/call-attr-kw self "sum" [] {:axis axis :dtype dtype :out out :keepdims keepdims }))

(defn swapaxes 
  "a.swapaxes(axis1, axis2)

    Return a view of the array with `axis1` and `axis2` interchanged.

    Refer to `numpy.swapaxes` for full documentation.

    See Also
    --------
    numpy.swapaxes : equivalent function"
  [ self  ]
  (py/call-attr self "swapaxes"  self  ))

(defn take 
  "
        "
  [self indices & {:keys [axis out mode]
                       :or {mode "raise"}} ]
    (py/call-attr-kw self "take" [indices] {:axis axis :out out :mode mode }))

(defn tobytes 
  "
        Return the array data as a string containing the raw bytes in the array.

        The array is filled with a fill value before the string conversion.

        .. versionadded:: 1.9.0

        Parameters
        ----------
        fill_value : scalar, optional
            Value used to fill in the masked values. Default is None, in which
            case `MaskedArray.fill_value` is used.
        order : {'C','F','A'}, optional
            Order of the data item in the copy. Default is 'C'.

            - 'C'   -- C order (row major).
            - 'F'   -- Fortran order (column major).
            - 'A'   -- Any, current order of array.
            - None  -- Same as 'A'.

        See Also
        --------
        ndarray.tobytes
        tolist, tofile

        Notes
        -----
        As for `ndarray.tobytes`, information about the shape, dtype, etc.,
        but also about `fill_value`, will be lost.

        Examples
        --------
        >>> x = np.ma.array(np.array([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
        >>> x.tobytes()
        b'\x01\x00\x00\x00\x00\x00\x00\x00?B\x0f\x00\x00\x00\x00\x00?B\x0f\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00'

        "
  [self  & {:keys [fill_value order]
                       :or {order "C"}} ]
    (py/call-attr-kw self "tobytes" [] {:fill_value fill_value :order order }))

(defn tofile 
  "
        Save a masked array to a file in binary format.

        .. warning::
          This function is not implemented yet.

        Raises
        ------
        NotImplementedError
            When `tofile` is called.

        "
  [self fid & {:keys [sep format]
                       :or {sep "" format "%s"}} ]
    (py/call-attr-kw self "tofile" [fid] {:sep sep :format format }))

(defn toflex 
  "
        Transforms a masked array into a flexible-type array.

        The flexible type array that is returned will have two fields:

        * the ``_data`` field stores the ``_data`` part of the array.
        * the ``_mask`` field stores the ``_mask`` part of the array.

        Parameters
        ----------
        None

        Returns
        -------
        record : ndarray
            A new flexible-type `ndarray` with two fields: the first element
            containing a value, the second element containing the corresponding
            mask boolean. The returned record shape matches self.shape.

        Notes
        -----
        A side-effect of transforming a masked array into a flexible `ndarray` is
        that meta information (``fill_value``, ...) will be lost.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.toflex()
        array([[(1, False), (2,  True), (3, False)],
               [(4,  True), (5, False), (6,  True)],
               [(7, False), (8,  True), (9, False)]],
              dtype=[('_data', '<i8'), ('_mask', '?')])

        "
  [ self  ]
  (py/call-attr self "toflex"  self  ))
(defn tolist 
  "
        Return the data portion of the masked array as a hierarchical Python list.

        Data items are converted to the nearest compatible Python type.
        Masked values are converted to `fill_value`. If `fill_value` is None,
        the corresponding entries in the output list will be ``None``.

        Parameters
        ----------
        fill_value : scalar, optional
            The value to use for invalid entries. Default is None.

        Returns
        -------
        result : list
            The Python list representation of the masked array.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3], [4,5,6], [7,8,9]], mask=[0] + [1,0]*4)
        >>> x.tolist()
        [[1, None, 3], [None, 5, None], [7, None, 9]]
        >>> x.tolist(-999)
        [[1, -999, 3], [-999, 5, -999], [7, -999, 9]]

        "
  [self   & {:keys [fill_value]} ]
    (py/call-attr-kw self "tolist" [] {:fill_value fill_value }))

(defn torecords 
  "
        Transforms a masked array into a flexible-type array.

        The flexible type array that is returned will have two fields:

        * the ``_data`` field stores the ``_data`` part of the array.
        * the ``_mask`` field stores the ``_mask`` part of the array.

        Parameters
        ----------
        None

        Returns
        -------
        record : ndarray
            A new flexible-type `ndarray` with two fields: the first element
            containing a value, the second element containing the corresponding
            mask boolean. The returned record shape matches self.shape.

        Notes
        -----
        A side-effect of transforming a masked array into a flexible `ndarray` is
        that meta information (``fill_value``, ...) will be lost.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.toflex()
        array([[(1, False), (2,  True), (3, False)],
               [(4,  True), (5, False), (6,  True)],
               [(7, False), (8,  True), (9, False)]],
              dtype=[('_data', '<i8'), ('_mask', '?')])

        "
  [ self  ]
  (py/call-attr self "torecords"  self  ))

(defn tostring 
  "
        This function is a compatibility alias for tobytes. Despite its name it
        returns bytes not strings.
        "
  [self  & {:keys [fill_value order]
                       :or {order "C"}} ]
    (py/call-attr-kw self "tostring" [] {:fill_value fill_value :order order }))

(defn trace 
  "a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)

    Return the sum along diagonals of the array.

    Refer to `numpy.trace` for full documentation.

    See Also
    --------
    numpy.trace : equivalent function"
  [self  & {:keys [offset axis1 axis2 dtype out]
                       :or {offset 0 axis1 0 axis2 1}} ]
    (py/call-attr-kw self "trace" [] {:offset offset :axis1 axis1 :axis2 axis2 :dtype dtype :out out }))

(defn transpose 
  "a.transpose(*axes)

    Returns a view of the array with axes transposed.

    For a 1-D array this has no effect, as a transposed vector is simply the
    same vector. To convert a 1-D array into a 2D column vector, an additional
    dimension must be added. `np.atleast2d(a).T` achieves this, as does
    `a[:, np.newaxis]`.
    For a 2-D array, this is a standard matrix transpose.
    For an n-D array, if axes are given, their order indicates how the
    axes are permuted (see Examples). If axes are not provided and
    ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
    ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

    Parameters
    ----------
    axes : None, tuple of ints, or `n` ints

     * None or no argument: reverses the order of the axes.

     * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
       `i`-th axis becomes `a.transpose()`'s `j`-th axis.

     * `n` ints: same as an n-tuple of the same ints (this form is
       intended simply as a \"convenience\" alternative to the tuple form)

    Returns
    -------
    out : ndarray
        View of `a`, with axes suitably permuted.

    See Also
    --------
    ndarray.T : Array property returning the array transposed.
    ndarray.reshape : Give a new shape to an array without changing its data.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2],
           [3, 4]])
    >>> a.transpose()
    array([[1, 3],
           [2, 4]])
    >>> a.transpose((1, 0))
    array([[1, 3],
           [2, 4]])
    >>> a.transpose(1, 0)
    array([[1, 3],
           [2, 4]])"
  [ self  ]
  (py/call-attr self "transpose"  self  ))

(defn unshare-mask 
  "
        Copy the mask and set the sharedmask flag to False.

        Whether the mask is shared between masked arrays can be seen from
        the `sharedmask` property. `unshare_mask` ensures the mask is not shared.
        A copy of the mask is only made if it was shared.

        See Also
        --------
        sharedmask

        "
  [ self  ]
  (py/call-attr self "unshare_mask"  self  ))

(defn var 
  "
    Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired.  If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed.  The default is to
        compute the variance of the flattened array.

        .. versionadded:: 1.7.0

        If this is a tuple of ints, a variance is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type
        the default is `float32`; for arrays of float types it is the same as
        the array type.
    out : ndarray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output, but the type is cast if
        necessary.
    ddof : int, optional
        \"Delta Degrees of Freedom\": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of elements. By
        default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `var` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    variance : ndarray, see dtype parameter above
        If ``out=None``, returns a new array containing the variance;
        otherwise, a reference to the output array is returned.

    See Also
    --------
    std, mean, nanmean, nanstd, nanvar
    numpy.doc.ufuncs : Section \"Output arguments\"

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite population.
    ``ddof=0`` provides a maximum likelihood estimate of the variance for
    normally distributed variables.

    Note that for complex numbers, the absolute value is taken before
    squaring, so that the result is always real and nonnegative.

    For floating-point input, the variance is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32` (see example
    below).  Specifying a higher-accuracy accumulator using the ``dtype``
    keyword can alleviate this issue.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.var(a)
    1.25
    >>> np.var(a, axis=0)
    array([1.,  1.])
    >>> np.var(a, axis=1)
    array([0.25,  0.25])

    In single precision, var() can be inaccurate:

    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.var(a)
    0.20250003

    Computing the variance in float64 is more accurate:

    >>> np.var(a, dtype=np.float64)
    0.20249999932944759 # may vary
    >>> ((1-0.55)**2 + (0.1-0.55)**2)/2
    0.2025

    "
  [self  & {:keys [axis dtype out ddof keepdims]
                       :or {ddof 0 keepdims <no value>}} ]
    (py/call-attr-kw self "var" [] {:axis axis :dtype dtype :out out :ddof ddof :keepdims keepdims }))
(defn view 
  "a.view(dtype=None, type=None)

    New view of array with the same data.

    Parameters
    ----------
    dtype : data-type or ndarray sub-class, optional
        Data-type descriptor of the returned view, e.g., float32 or int16. The
        default, None, results in the view having the same data-type as `a`.
        This argument can also be specified as an ndarray sub-class, which
        then specifies the type of the returned object (this is equivalent to
        setting the ``type`` parameter).
    type : Python type, optional
        Type of the returned view, e.g., ndarray or matrix.  Again, the
        default None results in type preservation.

    Notes
    -----
    ``a.view()`` is used two different ways:

    ``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view
    of the array's memory with a different data-type.  This can cause a
    reinterpretation of the bytes of memory.

    ``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just
    returns an instance of `ndarray_subclass` that looks at the same array
    (same shape, dtype, etc.)  This does not cause a reinterpretation of the
    memory.

    For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
    bytes per entry than the previous dtype (for example, converting a
    regular array to a structured array), then the behavior of the view
    cannot be predicted just from the superficial appearance of ``a`` (shown
    by ``print(a)``). It also depends on exactly how ``a`` is stored in
    memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus
    defined as a slice or transpose, etc., the view may give different
    results.


    Examples
    --------
    >>> x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])

    Viewing array data using a different type and dtype:

    >>> y = x.view(dtype=np.int16, type=np.matrix)
    >>> y
    matrix([[513]], dtype=int16)
    >>> print(type(y))
    <class 'numpy.matrix'>

    Creating a view on a structured array so it can be used in calculations

    >>> x = np.array([(1, 2),(3,4)], dtype=[('a', np.int8), ('b', np.int8)])
    >>> xv = x.view(dtype=np.int8).reshape(-1,2)
    >>> xv
    array([[1, 2],
           [3, 4]], dtype=int8)
    >>> xv.mean(0)
    array([2.,  3.])

    Making changes to the view changes the underlying array

    >>> xv[0,1] = 20
    >>> x
    array([(1, 20), (3,  4)], dtype=[('a', 'i1'), ('b', 'i1')])

    Using a view to convert an array to a recarray:

    >>> z = x.view(np.recarray)
    >>> z.a
    array([1, 3], dtype=int8)

    Views share data:

    >>> x[0] = (9, 10)
    >>> z[0]
    (9, 10)

    Views that change the dtype size (bytes per entry) should normally be
    avoided on arrays defined by slices, transposes, fortran-ordering, etc.:

    >>> x = np.array([[1,2,3],[4,5,6]], dtype=np.int16)
    >>> y = x[:, 0:2]
    >>> y
    array([[1, 2],
           [4, 5]], dtype=int16)
    >>> y.view(dtype=[('width', np.int16), ('length', np.int16)])
    Traceback (most recent call last):
        ...
    ValueError: To change to a dtype of a different size, the array must be C-contiguous
    >>> z = y.copy()
    >>> z.view(dtype=[('width', np.int16), ('length', np.int16)])
    array([[(1, 2)],
           [(4, 5)]], dtype=[('width', '<i2'), ('length', '<i2')])"
  [self   & {:keys [dtype type fill_value]} ]
    (py/call-attr-kw self "view" [] {:dtype dtype :type type :fill_value fill_value }))
