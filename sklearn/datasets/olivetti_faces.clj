(ns sklearn.datasets.olivetti-faces
  "Modified Olivetti faces dataset.

The original database was available from (now defunct)

    https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

The version retrieved here comes in MATLAB format from the personal
web page of Sam Roweis:

    https://cs.nyu.edu/~roweis/
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce olivetti-faces (import-module "sklearn.datasets.olivetti_faces"))

(defn check-random-state 
  "Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    "
  [ seed ]
  (py/call-attr olivetti-faces "check_random_state"  seed ))

(defn dirname 
  "Returns the directory component of a pathname"
  [ p ]
  (py/call-attr olivetti-faces "dirname"  p ))

(defn exists 
  "Test whether a path exists.  Returns False for broken symbolic links"
  [ path ]
  (py/call-attr olivetti-faces "exists"  path ))

(defn fetch-olivetti-faces 
  "Load the Olivetti faces data-set from AT&T (classification).

    Download it if necessary.

    =================   =====================
    Classes                                40
    Samples total                         400
    Dimensionality                       4096
    Features            real, between 0 and 1
    =================   =====================

    Read more in the :ref:`User Guide <olivetti_faces_dataset>`.

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    shuffle : boolean, optional
        If True the order of the dataset is shuffled to avoid having
        images of the same person grouped.

    random_state : int, RandomState instance or None (default=0)
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    download_if_missing : optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    An object with the following attributes:

    data : numpy array of shape (400, 4096)
        Each row corresponds to a ravelled face image of original size
        64 x 64 pixels.

    images : numpy array of shape (400, 64, 64)
        Each row is a face image corresponding to one of the 40 subjects
        of the dataset.

    target : numpy array of shape (400, )
        Labels associated to each face image. Those labels are ranging from
        0-39 and correspond to the Subject IDs.

    DESCR : string
        Description of the modified Olivetti Faces Dataset.
    "
  [data_home & {:keys [shuffle random_state download_if_missing]
                       :or {shuffle false random_state 0 download_if_missing true}} ]
    (py/call-attr-kw olivetti-faces "fetch_olivetti_faces" [data_home] {:shuffle shuffle :random_state random_state :download_if_missing download_if_missing }))

(defn get-data-home 
  "Return the path of the scikit-learn data dir.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data dir is set to a folder named 'scikit_learn_data' in the
    user home folder.

    Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str | None
        The path to scikit-learn data dir.
    "
  [ data_home ]
  (py/call-attr olivetti-faces "get_data_home"  data_home ))

(defn join 
  "Join two or more pathname components, inserting '/' as needed.
    If any component is an absolute path, all previous path components
    will be discarded.  An empty last part will result in a path that
    ends with a separator."
  [ a ]
  (py/call-attr olivetti-faces "join"  a ))

(defn loadmat 
  "
    Load MATLAB file.

    Parameters
    ----------
    file_name : str
       Name of the mat file (do not need .mat extension if
       appendmat==True). Can also pass open file-like object.
    mdict : dict, optional
        Dictionary in which to insert matfile variables.
    appendmat : bool, optional
       True to append the .mat extension to the end of the given
       filename, if not already present.
    byte_order : str or None, optional
       None by default, implying byte order guessed from mat
       file. Otherwise can be one of ('native', '=', 'little', '<',
       'BIG', '>').
    mat_dtype : bool, optional
       If True, return arrays in same dtype as would be loaded into
       MATLAB (instead of the dtype with which they are saved).
    squeeze_me : bool, optional
       Whether to squeeze unit matrix dimensions or not.
    chars_as_strings : bool, optional
       Whether to convert char arrays to string arrays.
    matlab_compatible : bool, optional
       Returns matrices as would be loaded by MATLAB (implies
       squeeze_me=False, chars_as_strings=False, mat_dtype=True,
       struct_as_record=True).
    struct_as_record : bool, optional
       Whether to load MATLAB structs as numpy record arrays, or as
       old-style numpy arrays with dtype=object.  Setting this flag to
       False replicates the behavior of scipy version 0.7.x (returning
       numpy object arrays).  The default setting is True, because it
       allows easier round-trip load and save of MATLAB files.
    verify_compressed_data_integrity : bool, optional
        Whether the length of compressed sequences in the MATLAB file
        should be checked, to ensure that they are not longer than we expect.
        It is advisable to enable this (the default) because overlong
        compressed sequences in MATLAB files generally indicate that the
        files have experienced some sort of corruption.
    variable_names : None or sequence
        If None (the default) - read all variables in file. Otherwise
        `variable_names` should be a sequence of strings, giving names of the
        MATLAB variables to read from the file.  The reader will skip any
        variable with a name not in this sequence, possibly saving some read
        processing.

    Returns
    -------
    mat_dict : dict
       dictionary with variable names as keys, and loaded matrices as
       values.

    Notes
    -----
    v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

    You will need an HDF5 python library to read MATLAB 7.3 format mat
    files.  Because scipy does not supply one, we do not implement the
    HDF5 / 7.3 interface here.

    Examples
    --------
    >>> from os.path import dirname, join as pjoin
    >>> import scipy.io as sio

    Get the filename for an example .mat file from the tests/data directory.

    >>> data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
    >>> mat_fname = pjoin(data_dir, 'testdouble_7.4_GLNX86.mat')

    Load the .mat file contents.

    >>> mat_contents = sio.loadmat(mat_fname)

    The result is a dictionary, one key/value pair for each variable:

    >>> sorted(mat_contents.keys())
    ['__globals__', '__header__', '__version__', 'testdouble']
    >>> mat_contents['testdouble']
    array([[0.        , 0.78539816, 1.57079633, 2.35619449, 3.14159265,
            3.92699082, 4.71238898, 5.49778714, 6.28318531]])

    By default SciPy reads MATLAB structs as structured NumPy arrays where the
    dtype fields are of type `object` and the names correspond to the MATLAB
    struct field names. This can be disabled by setting the optional argument
    `struct_as_record=False`.

    Get the filename for an example .mat file that contains a MATLAB struct
    called `teststruct` and load the contents.

    >>> matstruct_fname = pjoin(data_dir, 'teststruct_7.4_GLNX86.mat')
    >>> matstruct_contents = sio.loadmat(matstruct_fname)
    >>> teststruct = matstruct_contents['teststruct']
    >>> teststruct.dtype
    dtype([('stringfield', 'O'), ('doublefield', 'O'), ('complexfield', 'O')])

    The size of the structured array is the size of the MATLAB struct, not the
    number of elements in any particular field. The shape defaults to 2-D
    unless the optional argument `squeeze_me=True`, in which case all length 1
    dimensions are removed.

    >>> teststruct.size
    1
    >>> teststruct.shape
    (1, 1)

    Get the 'stringfield' of the first element in the MATLAB struct.

    >>> teststruct[0, 0]['stringfield']
    array(['Rats live on no evil star.'],
      dtype='<U26')

    Get the first element of the 'doublefield'.

    >>> teststruct['doublefield'][0, 0]
    array([[ 1.41421356,  2.71828183,  3.14159265]])

    Load the MATLAB struct, squeezing out length 1 dimensions, and get the item
    from the 'complexfield'.

    >>> matstruct_squeezed = sio.loadmat(matstruct_fname, squeeze_me=True)
    >>> matstruct_squeezed['teststruct'].shape
    ()
    >>> matstruct_squeezed['teststruct']['complexfield'].shape
    ()
    >>> matstruct_squeezed['teststruct']['complexfield'].item()
    array([ 1.41421356+1.41421356j,  2.71828183+2.71828183j,
        3.14159265+3.14159265j])
    "
  [file_name mdict & {:keys [appendmat]
                       :or {appendmat true}} ]
    (py/call-attr-kw olivetti-faces "loadmat" [file_name mdict] {:appendmat appendmat }))

(defn makedirs 
  "makedirs(name [, mode=0o777][, exist_ok=False])

    Super-mkdir; create a leaf directory and all intermediate ones.  Works like
    mkdir, except that any intermediate path segment (not just the rightmost)
    will be created if it does not exist. If the target directory already
    exists, raise an OSError if exist_ok is False. Otherwise no exception is
    raised.  This is recursive.

    "
  [name & {:keys [mode exist_ok]
                       :or {mode 511 exist_ok false}} ]
    (py/call-attr-kw olivetti-faces "makedirs" [name] {:mode mode :exist_ok exist_ok }))
