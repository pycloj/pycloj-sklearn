(ns sklearn.datasets.species-distributions
  "
=============================
Species distribution dataset
=============================

This dataset represents the geographic distribution of species.
The dataset is provided by Phillips et. al. (2006).

The two species are:

 - `\"Bradypus variegatus\"
   <http://www.iucnredlist.org/details/3038/0>`_ ,
   the Brown-throated Sloth.

 - `\"Microryzomys minutus\"
   <http://www.iucnredlist.org/details/13408/0>`_ ,
   also known as the Forest Small Rice Rat, a rodent that lives in Peru,
   Colombia, Ecuador, Peru, and Venezuela.

References
----------

`\"Maximum entropy modeling of species geographic distributions\"
<http://rob.schapire.net/papers/ecolmod.pdf>`_ S. J. Phillips,
R. P. Anderson, R. E. Schapire - Ecological Modelling, 190:231-259, 2006.

Notes
-----

For an example of using this dataset, see
:ref:`examples/applications/plot_species_distribution_modeling.py
<sphx_glr_auto_examples_applications_plot_species_distribution_modeling.py>`.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce species-distributions (import-module "sklearn.datasets.species_distributions"))

(defn construct-grids 
  "Construct the map grid from the batch object

    Parameters
    ----------
    batch : Batch object
        The object returned by :func:`fetch_species_distributions`

    Returns
    -------
    (xgrid, ygrid) : 1-D arrays
        The grid corresponding to the values in batch.coverages
    "
  [ batch ]
  (py/call-attr species-distributions "construct_grids"  batch ))

(defn exists 
  "Test whether a path exists.  Returns False for broken symbolic links"
  [ path ]
  (py/call-attr species-distributions "exists"  path ))

(defn fetch-species-distributions 
  "Loader for species distribution dataset from Phillips et. al. (2006)

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    The data is returned as a Bunch object with the following attributes:

    coverages : array, shape = [14, 1592, 1212]
        These represent the 14 features measured at each point of the map grid.
        The latitude/longitude values for the grid are discussed below.
        Missing data is represented by the value -9999.

    train : record array, shape = (1624,)
        The training points for the data.  Each point has three fields:

        - train['species'] is the species name
        - train['dd long'] is the longitude, in degrees
        - train['dd lat'] is the latitude, in degrees

    test : record array, shape = (620,)
        The test points for the data.  Same format as the training data.

    Nx, Ny : integers
        The number of longitudes (x) and latitudes (y) in the grid

    x_left_lower_corner, y_left_lower_corner : floats
        The (x,y) position of the lower-left corner, in degrees

    grid_size : float
        The spacing between points of the grid, in degrees

    References
    ----------

    * `\"Maximum entropy modeling of species geographic distributions\"
      <http://rob.schapire.net/papers/ecolmod.pdf>`_
      S. J. Phillips, R. P. Anderson, R. E. Schapire - Ecological Modelling,
      190:231-259, 2006.

    Notes
    -----

    This dataset represents the geographic distribution of species.
    The dataset is provided by Phillips et. al. (2006).

    The two species are:

    - `\"Bradypus variegatus\"
      <http://www.iucnredlist.org/details/3038/0>`_ ,
      the Brown-throated Sloth.

    - `\"Microryzomys minutus\"
      <http://www.iucnredlist.org/details/13408/0>`_ ,
      also known as the Forest Small Rice Rat, a rodent that lives in Peru,
      Colombia, Ecuador, Peru, and Venezuela.

    - For an example of using this dataset with scikit-learn, see
      :ref:`examples/applications/plot_species_distribution_modeling.py
      <sphx_glr_auto_examples_applications_plot_species_distribution_modeling.py>`.
    "
  [data_home & {:keys [download_if_missing]
                       :or {download_if_missing true}} ]
    (py/call-attr-kw species-distributions "fetch_species_distributions" [data_home] {:download_if_missing download_if_missing }))

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
  (py/call-attr species-distributions "get_data_home"  data_home ))

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
    (py/call-attr-kw species-distributions "makedirs" [name] {:mode mode :exist_ok exist_ok }))
