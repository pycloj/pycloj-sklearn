(ns sklearn.utils.testing
  "Testing utilities."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce testing (import-module "sklearn.utils.testing"))

(defn all-estimators 
  "Get a list of all estimators from sklearn.

    This function crawls the module and gets all classes that inherit
    from BaseEstimator. Classes that are defined in test-modules are not
    included.
    By default meta_estimators such as GridSearchCV are also not included.

    Parameters
    ----------
    include_meta_estimators : boolean, default=False
        Deprecated, ignored.
        .. deprecated:: 0.21
           ``include_meta_estimators`` has been deprecated and has no effect in
           0.21 and will be removed in 0.23.

    include_other : boolean, default=False
        Deprecated, ignored.
        .. deprecated:: 0.21
           ``include_other`` has been deprecated and has not effect in 0.21 and
           will be removed in 0.23.

    type_filter : string, list of string,  or None, default=None
        Which kind of estimators should be returned. If None, no filter is
        applied and all estimators are returned.  Possible values are
        'classifier', 'regressor', 'cluster' and 'transformer' to get
        estimators only of these specific types, or a list of these to
        get the estimators that fit at least one of the types.

    include_dont_test : boolean, default=False
        Deprecated, ignored.
        .. deprecated:: 0.21
           ``include_dont_test`` has been deprecated and has no effect in 0.21
           and will be removed in 0.23.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actuall type of the class.
    "
  [ include_meta_estimators include_other type_filter include_dont_test ]
  (py/call-attr testing "all_estimators"  include_meta_estimators include_other type_filter include_dont_test ))

(defn assert-allclose 
  "
    Raises an AssertionError if two objects are not equal up to desired
    tolerance.

    The test is equivalent to ``allclose(actual, desired, rtol, atol)`` (note
    that ``allclose`` has different default values). It compares the difference
    between `actual` and `desired` to ``atol + rtol * abs(desired)``.

    .. versionadded:: 1.5.0

    Parameters
    ----------
    actual : array_like
        Array obtained.
    desired : array_like
        Array desired.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    equal_nan : bool, optional.
        If True, NaNs will compare equal.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_array_almost_equal_nulp, assert_array_max_ulp

    Examples
    --------
    >>> x = [1e-5, 1e-3, 1e-1]
    >>> y = np.arccos(np.cos(x))
    >>> np.testing.assert_allclose(x, y, rtol=1e-5, atol=0)

    "
  [actual desired & {:keys [rtol atol equal_nan err_msg verbose]
                       :or {rtol 1e-07 atol 0 equal_nan true err_msg "" verbose true}} ]
    (py/call-attr-kw testing "assert_allclose" [actual desired] {:rtol rtol :atol atol :equal_nan equal_nan :err_msg err_msg :verbose verbose }))

(defn assert-allclose-dense-sparse 
  "Assert allclose for sparse and dense data.

    Both x and y need to be either sparse or dense, they
    can't be mixed.

    Parameters
    ----------
    x : array-like or sparse matrix
        First array to compare.

    y : array-like or sparse matrix
        Second array to compare.

    rtol : float, optional
        relative tolerance; see numpy.allclose

    atol : float, optional
        absolute tolerance; see numpy.allclose. Note that the default here is
        more tolerant than the default for numpy.testing.assert_allclose, where
        atol=0.

    err_msg : string, default=''
        Error message to raise.
    "
  [x y & {:keys [rtol atol err_msg]
                       :or {rtol 1e-07 atol 1e-09 err_msg ""}} ]
    (py/call-attr-kw testing "assert_allclose_dense_sparse" [x y] {:rtol rtol :atol atol :err_msg err_msg }))

(defn assert-almost-equal 
  "
    Raises an AssertionError if two items are not equal up to desired
    precision.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    The test verifies that the elements of ``actual`` and ``desired`` satisfy.

        ``abs(desired-actual) < 1.5 * 10**(-decimal)``

    That is a looser test than originally documented, but agrees with what the
    actual implementation in `assert_array_almost_equal` did up to rounding
    vagaries. An exception is raised at conflicting values. For ndarrays this
    delegates to assert_array_almost_equal

    Parameters
    ----------
    actual : array_like
        The object to check.
    desired : array_like
        The expected object.
    decimal : int, optional
        Desired precision, default is 7.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    >>> import numpy.testing as npt
    >>> npt.assert_almost_equal(2.3333333333333, 2.33333334)
    >>> npt.assert_almost_equal(2.3333333333333, 2.33333334, decimal=10)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 10 decimals
     ACTUAL: 2.3333333333333
     DESIRED: 2.33333334

    >>> npt.assert_almost_equal(np.array([1.0,2.3333333333333]),
    ...                         np.array([1.0,2.33333334]), decimal=9)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 9 decimals
    Mismatch: 50%
    Max absolute difference: 6.66669964e-09
    Max relative difference: 2.85715698e-09
     x: array([1.         , 2.333333333])
     y: array([1.        , 2.33333334])

    "
  [actual desired & {:keys [decimal err_msg verbose]
                       :or {decimal 7 err_msg "" verbose true}} ]
    (py/call-attr-kw testing "assert_almost_equal" [actual desired] {:decimal decimal :err_msg err_msg :verbose verbose }))

(defn assert-approx-equal 
  "
    Raises an AssertionError if two items are not equal up to significant
    digits.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    Given two numbers, check that they are approximately equal.
    Approximately equal is defined as the number of significant digits
    that agree.

    Parameters
    ----------
    actual : scalar
        The object to check.
    desired : scalar
        The expected object.
    significant : int, optional
        Desired precision, default is 7.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    >>> np.testing.assert_approx_equal(0.12345677777777e-20, 0.1234567e-20)
    >>> np.testing.assert_approx_equal(0.12345670e-20, 0.12345671e-20,
    ...                                significant=8)
    >>> np.testing.assert_approx_equal(0.12345670e-20, 0.12345672e-20,
    ...                                significant=8)
    Traceback (most recent call last):
        ...
    AssertionError:
    Items are not equal to 8 significant digits:
     ACTUAL: 1.234567e-21
     DESIRED: 1.2345672e-21

    the evaluated condition that raises the exception is

    >>> abs(0.12345670e-20/1e-21 - 0.12345672e-20/1e-21) >= 10**-(8-1)
    True

    "
  [actual desired & {:keys [significant err_msg verbose]
                       :or {significant 7 err_msg "" verbose true}} ]
    (py/call-attr-kw testing "assert_approx_equal" [actual desired] {:significant significant :err_msg err_msg :verbose verbose }))

(defn assert-array-almost-equal 
  "
    Raises an AssertionError if two objects are not equal up to desired
    precision.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    The test verifies identical shapes and that the elements of ``actual`` and
    ``desired`` satisfy.

        ``abs(desired-actual) < 1.5 * 10**(-decimal)``

    That is a looser test than originally documented, but agrees with what the
    actual implementation did up to rounding vagaries. An exception is raised
    at shape mismatch or conflicting values. In contrast to the standard usage
    in numpy, NaNs are compared like numbers, no assertion is raised if both
    objects have NaNs in the same positions.

    Parameters
    ----------
    x : array_like
        The actual object to check.
    y : array_like
        The desired, expected object.
    decimal : int, optional
        Desired precision, default is 6.
    err_msg : str, optional
      The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    the first assert does not raise an exception

    >>> np.testing.assert_array_almost_equal([1.0,2.333,np.nan],
    ...                                      [1.0,2.333,np.nan])

    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    ...                                      [1.0,2.33339,np.nan], decimal=5)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    Mismatch: 33.3%
    Max absolute difference: 6.e-05
    Max relative difference: 2.57136612e-05
     x: array([1.     , 2.33333,     nan])
     y: array([1.     , 2.33339,     nan])

    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    ...                                      [1.0,2.33333, 5], decimal=5)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    x and y nan location mismatch:
     x: array([1.     , 2.33333,     nan])
     y: array([1.     , 2.33333, 5.     ])

    "
  [x y & {:keys [decimal err_msg verbose]
                       :or {decimal 6 err_msg "" verbose true}} ]
    (py/call-attr-kw testing "assert_array_almost_equal" [x y] {:decimal decimal :err_msg err_msg :verbose verbose }))

(defn assert-array-equal 
  "
    Raises an AssertionError if two array_like objects are not equal.

    Given two array_like objects, check that the shape is equal and all
    elements of these objects are equal. An exception is raised at
    shape mismatch or conflicting values. In contrast to the standard usage
    in numpy, NaNs are compared like numbers, no assertion is raised if
    both objects have NaNs in the same positions.

    The usual caution for verifying equality with floating point numbers is
    advised.

    Parameters
    ----------
    x : array_like
        The actual object to check.
    y : array_like
        The desired, expected object.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired objects are not equal.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    The first assert does not raise an exception:

    >>> np.testing.assert_array_equal([1.0,2.33333,np.nan],
    ...                               [np.exp(0),2.33333, np.nan])

    Assert fails with numerical inprecision with floats:

    >>> np.testing.assert_array_equal([1.0,np.pi,np.nan],
    ...                               [1, np.sqrt(np.pi)**2, np.nan])
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    Mismatch: 33.3%
    Max absolute difference: 4.4408921e-16
    Max relative difference: 1.41357986e-16
     x: array([1.      , 3.141593,      nan])
     y: array([1.      , 3.141593,      nan])

    Use `assert_allclose` or one of the nulp (number of floating point values)
    functions for these cases instead:

    >>> np.testing.assert_allclose([1.0,np.pi,np.nan],
    ...                            [1, np.sqrt(np.pi)**2, np.nan],
    ...                            rtol=1e-10, atol=0)

    "
  [x y & {:keys [err_msg verbose]
                       :or {err_msg "" verbose true}} ]
    (py/call-attr-kw testing "assert_array_equal" [x y] {:err_msg err_msg :verbose verbose }))

(defn assert-array-less 
  "
    Raises an AssertionError if two array_like objects are not ordered by less
    than.

    Given two array_like objects, check that the shape is equal and all
    elements of the first object are strictly smaller than those of the
    second object. An exception is raised at shape mismatch or incorrectly
    ordered values. Shape mismatch does not raise if an object has zero
    dimension. In contrast to the standard usage in numpy, NaNs are
    compared, no assertion is raised if both objects have NaNs in the same
    positions.



    Parameters
    ----------
    x : array_like
      The smaller object to check.
    y : array_like
      The larger object to compare.
    err_msg : string
      The error message to be printed in case of failure.
    verbose : bool
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired objects are not equal.

    See Also
    --------
    assert_array_equal: tests objects for equality
    assert_array_almost_equal: test objects for equality up to precision



    Examples
    --------
    >>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1.1, 2.0, np.nan])
    >>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1, 2.0, np.nan])
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    Mismatch: 33.3%
    Max absolute difference: 1.
    Max relative difference: 0.5
     x: array([ 1.,  1., nan])
     y: array([ 1.,  2., nan])

    >>> np.testing.assert_array_less([1.0, 4.0], 3)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    Mismatch: 50%
    Max absolute difference: 2.
    Max relative difference: 0.66666667
     x: array([1., 4.])
     y: array(3)

    >>> np.testing.assert_array_less([1.0, 2.0, 3.0], [4])
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    (shapes (3,), (1,) mismatch)
     x: array([1., 2., 3.])
     y: array([4])

    "
  [x y & {:keys [err_msg verbose]
                       :or {err_msg "" verbose true}} ]
    (py/call-attr-kw testing "assert_array_less" [x y] {:err_msg err_msg :verbose verbose }))

(defn assert-false 
  "DEPRECATED: 'assert_false' is deprecated in version 0.21 and will be removed in version 0.23. Please use 'assert' instead.

Check that the expression is false."
  [ expr msg ]
  (py/call-attr testing "assert_false"  expr msg ))

(defn assert-no-warnings 
  "
    Parameters
    ----------
    func
    *args
    **kw
    "
  [ func ]
  (py/call-attr testing "assert_no_warnings"  func ))

(defn assert-raise-message 
  "Helper function to test the message raised in an exception.

    Given an exception, a callable to raise the exception, and
    a message string, tests that the correct exception is raised and
    that the message is a substring of the error thrown. Used to test
    that the specific message thrown during an exception is correct.

    Parameters
    ----------
    exceptions : exception or tuple of exception
        An Exception object.

    message : str
        The error message or a substring of the error message.

    function : callable
        Callable object to raise error.

    *args : the positional arguments to `function`.

    **kwargs : the keyword arguments to `function`.
    "
  [ exceptions message function ]
  (py/call-attr testing "assert_raise_message"  exceptions message function ))

(defn assert-run-python-script 
  "Utility to check assertions in an independent Python subprocess.

    The script provided in the source code should return 0 and not print
    anything on stderr or stdout.

    This is a port from cloudpickle https://github.com/cloudpipe/cloudpickle

    Parameters
    ----------
    source_code : str
        The Python source code to execute.
    timeout : int
        Time in seconds before timeout.
    "
  [source_code & {:keys [timeout]
                       :or {timeout 60}} ]
    (py/call-attr-kw testing "assert_run_python_script" [source_code] {:timeout timeout }))

(defn assert-true 
  "DEPRECATED: 'assert_true' is deprecated in version 0.21 and will be removed in version 0.23. Please use 'assert' instead.

Check that the expression is true."
  [ expr msg ]
  (py/call-attr testing "assert_true"  expr msg ))

(defn assert-warns 
  "Test that a certain warning occurs.

    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.

    func : callable
        Callable object to trigger warnings.

    *args : the positional arguments to `func`.

    **kw : the keyword arguments to `func`

    Returns
    -------

    result : the return value of `func`

    "
  [ warning_class func ]
  (py/call-attr testing "assert_warns"  warning_class func ))

(defn assert-warns-div0 
  "Assume that numpy's warning for divide by zero is raised

    Handles the case of platforms that do not support warning on divide by zero

    Parameters
    ----------
    func
    *args
    **kw
    "
  [ func ]
  (py/call-attr testing "assert_warns_div0"  func ))

(defn assert-warns-message 
  "Test that a certain warning occurs and with a certain message.

    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.

    message : str | callable
        The message or a substring of the message to test for. If callable,
        it takes a string as the argument and will trigger an AssertionError
        if the callable returns `False`.

    func : callable
        Callable object to trigger warnings.

    *args : the positional arguments to `func`.

    **kw : the keyword arguments to `func`.

    Returns
    -------
    result : the return value of `func`

    "
  [ warning_class message func ]
  (py/call-attr testing "assert_warns_message"  warning_class message func ))

(defn check-docstring-parameters 
  "Helper to check docstring

    Parameters
    ----------
    func : callable
        The function object to test.
    doc : str, optional (default: None)
        Docstring if it is passed manually to the test.
    ignore : None | list
        Parameters to ignore.
    class_name : string, optional (default: None)
       If ``func`` is a class method and the class name is known specify
       class_name for the error message.

    Returns
    -------
    incorrect : list
        A list of string describing the incorrect results.
    "
  [ func doc ignore class_name ]
  (py/call-attr testing "check_docstring_parameters"  func doc ignore class_name ))

(defn check-output 
  "Run command with arguments and return its output.

    If the exit code was non-zero it raises a CalledProcessError.  The
    CalledProcessError object will have the return code in the returncode
    attribute and output in the output attribute.

    The arguments are the same as for the Popen constructor.  Example:

    >>> check_output([\"ls\", \"-l\", \"/dev/null\"])
    b'crw-rw-rw- 1 root root 1, 3 Oct 18  2007 /dev/null\n'

    The stdout argument is not allowed as it is used internally.
    To capture standard error in the result, use stderr=STDOUT.

    >>> check_output([\"/bin/sh\", \"-c\",
    ...               \"ls -l non_existent_file ; exit 0\"],
    ...              stderr=STDOUT)
    b'ls: non_existent_file: No such file or directory\n'

    There is an additional optional argument, \"input\", allowing you to
    pass a string to the subprocess's stdin.  If you use this argument
    you may not also use the Popen constructor's \"stdin\" argument, as
    it too will be used internally.  Example:

    >>> check_output([\"sed\", \"-e\", \"s/foo/bar/\"],
    ...              input=b\"when in the course of fooman events\n\")
    b'when in the course of barman events\n'

    By default, all communication is in bytes, and therefore any \"input\"
    should be bytes, and the return value will be bytes.  If in text mode,
    any \"input\" should be a string, and the return value will be a string
    decoded according to locale encoding, or by \"encoding\" if set. Text mode
    is triggered by setting any of text, encoding, errors or universal_newlines.
    "
  [ timeout ]
  (py/call-attr testing "check_output"  timeout ))

(defn check-skip-network 
  ""
  [  ]
  (py/call-attr testing "check_skip_network"  ))

(defn clean-warning-registry 
  "Clean Python warning registry for easier testing of warning messages.

    We may not need to do this any more when getting rid of Python 2, not
    entirely sure. See https://bugs.python.org/issue4180 and
    https://bugs.python.org/issue21724 for more details.

    "
  [  ]
  (py/call-attr testing "clean_warning_registry"  ))

(defn create-memmap-backed-data 
  "
    Parameters
    ----------
    data
    mmap_mode
    return_folder
    "
  [data & {:keys [mmap_mode return_folder]
                       :or {mmap_mode "r" return_folder false}} ]
    (py/call-attr-kw testing "create_memmap_backed_data" [data] {:mmap_mode mmap_mode :return_folder return_folder }))

(defn fake-mldata 
  "DEPRECATED: deprecated in version 0.20 to be removed in version 0.22

Create a fake mldata data set.

    .. deprecated:: 0.20
        Will be removed in version 0.22

    Parameters
    ----------
    columns_dict : dict, keys=str, values=ndarray
        Contains data as columns_dict[column_name] = array of data.

    dataname : string
        Name of data set.

    matfile : string or file object
        The file name string or the file-like object of the output file.

    ordering : list, default None
        List of column_names, determines the ordering in the data set.

    Notes
    -----
    This function transposes all arrays, while fetch_mldata only transposes
    'data', keep that into account in the tests.
    "
  [ columns_dict dataname matfile ordering ]
  (py/call-attr testing "fake_mldata"  columns_dict dataname matfile ordering ))

(defn ignore-warnings 
  "Context manager and decorator to ignore warnings.

    Note: Using this (in both variants) will clear all warnings
    from all python modules loaded. In case you need to test
    cross-module-warning-logging, this is not your tool of choice.

    Parameters
    ----------
    obj : callable or None
        callable where you want to ignore the warnings.
    category : warning class, defaults to Warning.
        The category to filter. If Warning, all categories will be muted.

    Examples
    --------
    >>> with ignore_warnings():
    ...     warnings.warn('buhuhuhu')

    >>> def nasty_warn():
    ...    warnings.warn('buhuhuhu')
    ...    print(42)

    >>> ignore_warnings(nasty_warn)()
    42
    "
  [obj & {:keys [category]
                       :or {category <class 'Warning'>}} ]
    (py/call-attr-kw testing "ignore_warnings" [obj] {:category category }))

(defn install-mldata-mock 
  "
    Parameters
    ----------
    mock_datasets : dict
        A dictionary of {dataset_name: data_dict}, or
        {dataset_name: (data_dict, ordering). `data_dict` itself is a
        dictionary of {column_name: data_array}, and `ordering` is a list of
        column_names to determine the ordering in the data set (see
        :func:`fake_mldata` for details).
    "
  [ mock_datasets ]
  (py/call-attr testing "install_mldata_mock"  mock_datasets ))

(defn set-random-state 
  "Set random state of an estimator if it has the `random_state` param.

    Parameters
    ----------
    estimator : object
        The estimator
    random_state : int, RandomState instance or None, optional, default=0
        Pseudo random number generator state.  If int, random_state is the seed
        used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    "
  [estimator & {:keys [random_state]
                       :or {random_state 0}} ]
    (py/call-attr-kw testing "set_random_state" [estimator] {:random_state random_state }))

(defn signature 
  "Get a signature object for the passed callable."
  [obj & {:keys [follow_wrapped]
                       :or {follow_wrapped true}} ]
    (py/call-attr-kw testing "signature" [obj] {:follow_wrapped follow_wrapped }))

(defn uninstall-mldata-mock 
  ""
  [  ]
  (py/call-attr testing "uninstall_mldata_mock"  ))

(defn urlopen 
  "Open the URL url, which can be either a string or a Request object.

    *data* must be an object specifying additional data to be sent to
    the server, or None if no such data is needed.  See Request for
    details.

    urllib.request module uses HTTP/1.1 and includes a \"Connection:close\"
    header in its HTTP requests.

    The optional *timeout* parameter specifies a timeout in seconds for
    blocking operations like the connection attempt (if not specified, the
    global default timeout setting will be used). This only works for HTTP,
    HTTPS and FTP connections.

    If *context* is specified, it must be a ssl.SSLContext instance describing
    the various SSL options. See HTTPSConnection for more details.

    The optional *cafile* and *capath* parameters specify a set of trusted CA
    certificates for HTTPS requests. cafile should point to a single file
    containing a bundle of CA certificates, whereas capath should point to a
    directory of hashed certificate files. More information can be found in
    ssl.SSLContext.load_verify_locations().

    The *cadefault* parameter is ignored.

    This function always returns an object which can work as a context
    manager and has methods such as

    * geturl() - return the URL of the resource retrieved, commonly used to
      determine if a redirect was followed

    * info() - return the meta-information of the page, such as headers, in the
      form of an email.message_from_string() instance (see Quick Reference to
      HTTP Headers)

    * getcode() - return the HTTP status code of the response.  Raises URLError
      on errors.

    For HTTP and HTTPS URLs, this function returns a http.client.HTTPResponse
    object slightly modified. In addition to the three new methods above, the
    msg attribute contains the same information as the reason attribute ---
    the reason phrase returned by the server --- instead of the response
    headers as it is specified in the documentation for HTTPResponse.

    For FTP, file, and data URLs and requests explicitly handled by legacy
    URLopener and FancyURLopener classes, this function returns a
    urllib.response.addinfourl object.

    Note that None may be returned if no handler handles the request (though
    the default installed global OpenerDirector uses UnknownHandler to ensure
    this never happens).

    In addition, if proxy settings are detected (for example, when a *_proxy
    environment variable like http_proxy is set), ProxyHandler is default
    installed and makes sure the requests are handled through the proxy.

    "
  [url data & {:keys [timeout cafile capath cadefault context]
                       :or {timeout <object object at 0x110c34e40> cadefault false}} ]
    (py/call-attr-kw testing "urlopen" [url data] {:timeout timeout :cafile cafile :capath capath :cadefault cadefault :context context }))

(defn wraps 
  "Decorator factory to apply update_wrapper() to a wrapper function

       Returns a decorator that invokes update_wrapper() with the decorated
       function as the wrapper argument and the arguments to wraps() as the
       remaining arguments. Default arguments are as for update_wrapper().
       This is a convenience function to simplify applying partial() to
       update_wrapper().
    "
  [wrapped & {:keys [assigned updated]
                       :or {assigned ('__module__', '__name__', '__qualname__', '__doc__', '__annotations__') updated ('__dict__',)}} ]
    (py/call-attr-kw testing "wraps" [wrapped] {:assigned assigned :updated updated }))
