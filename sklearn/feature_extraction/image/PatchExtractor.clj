(ns sklearn.feature-extraction.image.PatchExtractor
  "Extracts patches from a collection of images

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    patch_size : tuple of ints (patch_height, patch_width)
        the dimensions of one patch

    max_patches : integer or float, optional default is None
        The maximum number of patches per image to extract. If max_patches is a
        float in (0, 1), it is taken to mean a proportion of the total number
        of patches.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples
    --------
    >>> from sklearn.datasets import load_sample_images
    >>> from sklearn.feature_extraction import image
    >>> # Use the array data from the second image in this dataset:
    >>> X = load_sample_images().images[1]
    >>> print('Image shape: {}'.format(X.shape))
    Image shape: (427, 640, 3)
    >>> pe = image.PatchExtractor(patch_size=(2, 2))
    >>> pe_fit = pe.fit(X)
    >>> pe_trans = pe.transform(X)
    >>> print('Patches shape: {}'.format(pe_trans.shape))
    Patches shape: (545706, 2, 2)
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce image (import-module "sklearn.feature_extraction.image"))

(defn PatchExtractor 
  "Extracts patches from a collection of images

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    patch_size : tuple of ints (patch_height, patch_width)
        the dimensions of one patch

    max_patches : integer or float, optional default is None
        The maximum number of patches per image to extract. If max_patches is a
        float in (0, 1), it is taken to mean a proportion of the total number
        of patches.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples
    --------
    >>> from sklearn.datasets import load_sample_images
    >>> from sklearn.feature_extraction import image
    >>> # Use the array data from the second image in this dataset:
    >>> X = load_sample_images().images[1]
    >>> print('Image shape: {}'.format(X.shape))
    Image shape: (427, 640, 3)
    >>> pe = image.PatchExtractor(patch_size=(2, 2))
    >>> pe_fit = pe.fit(X)
    >>> pe_trans = pe.transform(X)
    >>> print('Patches shape: {}'.format(pe_trans.shape))
    Patches shape: (545706, 2, 2)
    "
  [ patch_size max_patches random_state ]
  (py/call-attr image "PatchExtractor"  patch_size max_patches random_state ))

(defn fit 
  "Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training data.
        "
  [ self X y ]
  (py/call-attr self "fit"  self X y ))

(defn get-params 
  "Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        "
  [self  & {:keys [deep]
                       :or {deep true}} ]
    (py/call-attr-kw self "get_params" [] {:deep deep }))

(defn set-params 
  "Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        "
  [ self  ]
  (py/call-attr self "set_params"  self  ))

(defn transform 
  "Transforms the image samples in X into a matrix of patch data.

        Parameters
        ----------
        X : array, shape = (n_samples, image_height, image_width) or
            (n_samples, image_height, image_width, n_channels)
            Array of images from which to extract patches. For color images,
            the last dimension specifies the channel: a RGB image would have
            `n_channels=3`.

        Returns
        -------
        patches : array, shape = (n_patches, patch_height, patch_width) or
             (n_patches, patch_height, patch_width, n_channels)
             The collection of patches extracted from the images, where
             `n_patches` is either `n_samples * max_patches` or the total
             number of patches that can be extracted.
        "
  [ self X ]
  (py/call-attr self "transform"  self X ))
