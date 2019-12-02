(ns sklearn.datasets.lfw
  "Labeled Faces in the Wild (LFW) dataset

This dataset is a collection of JPEG pictures of famous people collected
over the internet, all details are available on the official website:

    http://vis-www.cs.umass.edu/lfw/
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

(defn dirname 
  "Returns the directory component of a pathname"
  [ p ]
  (py/call-attr lfw "dirname"  p ))

(defn exists 
  "Test whether a path exists.  Returns False for broken symbolic links"
  [ path ]
  (py/call-attr lfw "exists"  path ))

(defn fetch-lfw-pairs 
  "Load the Labeled Faces in the Wild (LFW) pairs dataset (classification).

    Download it if necessary.

    =================   =======================
    Classes                                5749
    Samples total                         13233
    Dimensionality                         5828
    Features            real, between 0 and 255
    =================   =======================

    In the official `README.txt`_ this task is described as the
    \"Restricted\" task.  As I am not sure as to implement the
    \"Unrestricted\" variant correctly, I left it as unsupported for now.

      .. _`README.txt`: http://vis-www.cs.umass.edu/lfw/README.txt

    The original images are 250 x 250 pixels, but the default slice and resize
    arguments reduce them to 62 x 47.

    Read more in the :ref:`User Guide <labeled_faces_in_the_wild_dataset>`.

    Parameters
    ----------
    subset : optional, default: 'train'
        Select the dataset to load: 'train' for the development training
        set, 'test' for the development test set, and '10_folds' for the
        official evaluation set that is meant to be used with a 10-folds
        cross validation.

    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By
        default all scikit-learn data is stored in '~/scikit_learn_data'
        subfolders.

    funneled : boolean, optional, default: True
        Download and use the funneled variant of the dataset.

    resize : float, optional, default 0.5
        Ratio used to resize the each face picture.

    color : boolean, optional, default False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than the shape with color = False.

    slice_ : optional
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background

    download_if_missing : optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    The data is returned as a Bunch object with the following attributes:

    data : numpy array of shape (2200, 5828). Shape depends on ``subset``.
        Each row corresponds to 2 ravel'd face images of original size 62 x 47
        pixels. Changing the ``slice_``, ``resize`` or ``subset`` parameters
        will change the shape of the output.

    pairs : numpy array of shape (2200, 2, 62, 47). Shape depends on ``subset``
        Each row has 2 face images corresponding to same or different person
        from the dataset containing 5749 people. Changing the ``slice_``,
        ``resize`` or ``subset`` parameters will change the shape of the
        output.

    target : numpy array of shape (2200,). Shape depends on ``subset``.
        Labels associated to each pair of images. The two label values being
        different persons or the same person.

    DESCR : string
        Description of the Labeled Faces in the Wild (LFW) dataset.

    "
  [ & {:keys [subset data_home funneled resize color slice_ download_if_missing]
       :or {subset "train" funneled true resize 0.5 color false slice_ (slice(70, 195, None), slice(78, 172, None)) download_if_missing true}} ]
  
   (py/call-attr-kw lfw "fetch_lfw_pairs" [] {:subset subset :data_home data_home :funneled funneled :resize resize :color color :slice_ slice_ :download_if_missing download_if_missing }))

(defn fetch-lfw-people 
  "Load the Labeled Faces in the Wild (LFW) people dataset (classification).

    Download it if necessary.

    =================   =======================
    Classes                                5749
    Samples total                         13233
    Dimensionality                         5828
    Features            real, between 0 and 255
    =================   =======================

    Read more in the :ref:`User Guide <labeled_faces_in_the_wild_dataset>`.

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    funneled : boolean, optional, default: True
        Download and use the funneled variant of the dataset.

    resize : float, optional, default 0.5
        Ratio used to resize the each face picture.

    min_faces_per_person : int, optional, default None
        The extracted dataset will only retain pictures of people that have at
        least `min_faces_per_person` different pictures.

    color : boolean, optional, default False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than the shape with color = False.

    slice_ : optional
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background

    download_if_missing : optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : boolean, default=False.
        If True, returns ``(dataset.data, dataset.target)`` instead of a Bunch
        object. See below for more information about the `dataset.data` and
        `dataset.target` object.

        .. versionadded:: 0.20

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : numpy array of shape (13233, 2914)
        Each row corresponds to a ravelled face image of original size 62 x 47
        pixels. Changing the ``slice_`` or resize parameters will change the
        shape of the output.

    dataset.images : numpy array of shape (13233, 62, 47)
        Each row is a face image corresponding to one of the 5749 people in
        the dataset. Changing the ``slice_`` or resize parameters will change
        the shape of the output.

    dataset.target : numpy array of shape (13233,)
        Labels associated to each face image. Those labels range from 0-5748
        and correspond to the person IDs.

    dataset.DESCR : string
        Description of the Labeled Faces in the Wild (LFW) dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.20

    "
  [data_home & {:keys [funneled resize min_faces_per_person color slice_ download_if_missing return_X_y]
                       :or {funneled true resize 0.5 min_faces_per_person 0 color false slice_ (slice(70, 195, None), slice(78, 172, None)) download_if_missing true return_X_y false}} ]
    (py/call-attr-kw lfw "fetch_lfw_people" [data_home] {:funneled funneled :resize resize :min_faces_per_person min_faces_per_person :color color :slice_ slice_ :download_if_missing download_if_missing :return_X_y return_X_y }))

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
  (py/call-attr lfw "get_data_home"  data_home ))

(defn isdir 
  "Return true if the pathname refers to an existing directory."
  [ s ]
  (py/call-attr lfw "isdir"  s ))

(defn join 
  "Join two or more pathname components, inserting '/' as needed.
    If any component is an absolute path, all previous path components
    will be discarded.  An empty last part will result in a path that
    ends with a separator."
  [ a ]
  (py/call-attr lfw "join"  a ))

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
    (py/call-attr-kw lfw "makedirs" [name] {:mode mode :exist_ok exist_ok }))

(defn scale-face 
  "DEPRECATED: This function was deprecated in version 0.20 and will be removed in 0.22.

Scale back to 0-1 range in case of normalization for plotting.

    .. deprecated:: 0.20
    This function was deprecated in version 0.20 and will be removed in 0.22.


    Parameters
    ----------
    face : array_like
        The array to scale

    Returns
    -------
    array_like
        The scaled array
    "
  [ face ]
  (py/call-attr lfw "scale_face"  face ))
