(ns sklearn.utils.testing.mock-mldata-urlopen
  "Object that mocks the urlopen function to fake requests to mldata.

    When requesting a dataset with a name that is in mock_datasets, this object
    creates a fake dataset in a StringIO object and returns it. Otherwise, it
    raises an HTTPError.

    .. deprecated:: 0.20
        Will be removed in version 0.22

    Parameters
    ----------
    mock_datasets : dict
        A dictionary of {dataset_name: data_dict}, or
        {dataset_name: (data_dict, ordering). `data_dict` itself is a
        dictionary of {column_name: data_array}, and `ordering` is a list of
        column_names to determine the ordering in the data set (see
        :func:`fake_mldata` for details).
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce testing (import-module "sklearn.utils.testing"))

(defn mock-mldata-urlopen 
  "Object that mocks the urlopen function to fake requests to mldata.

    When requesting a dataset with a name that is in mock_datasets, this object
    creates a fake dataset in a StringIO object and returns it. Otherwise, it
    raises an HTTPError.

    .. deprecated:: 0.20
        Will be removed in version 0.22

    Parameters
    ----------
    mock_datasets : dict
        A dictionary of {dataset_name: data_dict}, or
        {dataset_name: (data_dict, ordering). `data_dict` itself is a
        dictionary of {column_name: data_array}, and `ordering` is a list of
        column_names to determine the ordering in the data set (see
        :func:`fake_mldata` for details).
    "
  [  ]
  (py/call-attr testing "mock_mldata_urlopen"  ))
