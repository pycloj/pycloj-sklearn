(ns sklearn.feature-extraction
  "
The :mod:`sklearn.feature_extraction` module deals with feature extraction
from raw data. It currently includes methods to extract features from text and
images.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce feature-extraction (import-module "sklearn.feature_extraction"))

(defn grid-to-graph 
  "Graph of the pixel-to-pixel connections

    Edges exist if 2 voxels are connected.

    Parameters
    ----------
    n_x : int
        Dimension in x axis
    n_y : int
        Dimension in y axis
    n_z : int, optional, default 1
        Dimension in z axis
    mask : ndarray of booleans, optional
        An optional mask of the image, to consider only part of the
        pixels.
    return_as : np.ndarray or a sparse matrix class, optional
        The class to use to build the returned adjacency matrix.
    dtype : dtype, optional, default int
        The data of the returned sparse matrix. By default it is int

    Notes
    -----
    For scikit-learn versions 0.14.1 and prior, return_as=np.ndarray was
    handled by returning a dense np.matrix instance.  Going forward, np.ndarray
    returns an np.ndarray, as expected.

    For compatibility, user code relying on this method should wrap its
    calls in ``np.asarray`` to avoid type issues.
    "
  [n_x n_y & {:keys [n_z mask return_as dtype]
                       :or {n_z 1 return_as <class 'scipy.sparse.coo.coo_matrix'> dtype <class 'int'>}} ]
    (py/call-attr-kw feature-extraction "grid_to_graph" [n_x n_y] {:n_z n_z :mask mask :return_as return_as :dtype dtype }))

(defn img-to-graph 
  "Graph of the pixel-to-pixel gradient connections

    Edges are weighted with the gradient values.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    img : ndarray, 2D or 3D
        2D or 3D image
    mask : ndarray of booleans, optional
        An optional mask of the image, to consider only part of the
        pixels.
    return_as : np.ndarray or a sparse matrix class, optional
        The class to use to build the returned adjacency matrix.
    dtype : None or dtype, optional
        The data of the returned sparse matrix. By default it is the
        dtype of img

    Notes
    -----
    For scikit-learn versions 0.14.1 and prior, return_as=np.ndarray was
    handled by returning a dense np.matrix instance.  Going forward, np.ndarray
    returns an np.ndarray, as expected.

    For compatibility, user code relying on this method should wrap its
    calls in ``np.asarray`` to avoid type issues.
    "
  [img mask & {:keys [return_as dtype]
                       :or {return_as <class 'scipy.sparse.coo.coo_matrix'>}} ]
    (py/call-attr-kw feature-extraction "img_to_graph" [img mask] {:return_as return_as :dtype dtype }))
