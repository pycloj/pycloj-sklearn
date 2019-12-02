(ns sklearn.datasets.mldata
  "Automatically download MLdata datasets."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce mldata (import-module "sklearn.datasets.mldata"))

(defn copyfileobj 
  "copy data from file-like object fsrc to file-like object fdst"
  [fsrc fdst & {:keys [length]
                       :or {length 16384}} ]
    (py/call-attr-kw mldata "copyfileobj" [fsrc fdst] {:length length }))

(defn exists 
  "Test whether a path exists.  Returns False for broken symbolic links"
  [ path ]
  (py/call-attr mldata "exists"  path ))

(defn fetch-mldata 
  "DEPRECATED: fetch_mldata was deprecated in version 0.20 and will be removed in version 0.22. Please use fetch_openml.

Fetch an mldata.org data set

    mldata.org is no longer operational.

    If the file does not exist yet, it is downloaded from mldata.org .

    mldata.org does not have an enforced convention for storing data or
    naming the columns in a data set. The default behavior of this function
    works well with the most common cases:

      1) data values are stored in the column 'data', and target values in the
         column 'label'
      2) alternatively, the first column stores target values, and the second
         data values
      3) the data array is stored as `n_features x n_samples` , and thus needs
         to be transposed to match the `sklearn` standard

    Keyword arguments allow to adapt these defaults to specific data sets
    (see parameters `target_name`, `data_name`, `transpose_data`, and
    the examples below).

    mldata.org data sets may have multiple columns, which are stored in the
    Bunch object with their original name.

    .. deprecated:: 0.20
        Will be removed in version 0.22

    Parameters
    ----------

    dataname : str
        Name of the data set on mldata.org,
        e.g.: \"leukemia\", \"Whistler Daily Snowfall\", etc.
        The raw name is automatically converted to a mldata.org URL .

    target_name : optional, default: 'label'
        Name or index of the column containing the target values.

    data_name : optional, default: 'data'
        Name or index of the column containing the data.

    transpose_data : optional, default: True
        If True, transpose the downloaded data array.

    data_home : optional, default: None
        Specify another download and cache folder for the data sets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    Returns
    -------

    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'DESCR', the full description of the dataset, and
        'COL_NAMES', the original names of the dataset columns.
    "
  [dataname & {:keys [target_name data_name transpose_data data_home]
                       :or {target_name "label" data_name "data" transpose_data true}} ]
    (py/call-attr-kw mldata "fetch_mldata" [dataname] {:target_name target_name :data_name data_name :transpose_data transpose_data :data_home data_home }))

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
  (py/call-attr mldata "get_data_home"  data_home ))

(defn join 
  "Join two or more pathname components, inserting '/' as needed.
    If any component is an absolute path, all previous path components
    will be discarded.  An empty last part will result in a path that
    ends with a separator."
  [ a ]
  (py/call-attr mldata "join"  a ))

(defn mldata-filename 
  "DEPRECATED: mldata_filename was deprecated in version 0.20 and will be removed in version 0.22. Please use fetch_openml.

Convert a raw name for a data set in a mldata.org filename.

    .. deprecated:: 0.20
        Will be removed in version 0.22

    Parameters
    ----------
    dataname : str
        Name of dataset

    Returns
    -------
    fname : str
        The converted dataname.
    "
  [ dataname ]
  (py/call-attr mldata "mldata_filename"  dataname ))

(defn quote 
  "quote('abc def') -> 'abc%20def'

    Each part of a URL, e.g. the path info, the query, etc., has a
    different set of reserved characters that must be quoted. The
    quote function offers a cautious (not minimal) way to quote a
    string for most of these parts.

    RFC 3986 Uniform Resource Identifier (URI): Generic Syntax lists
    the following (un)reserved characters.

    unreserved    = ALPHA / DIGIT / \"-\" / \".\" / \"_\" / \"~\"
    reserved      = gen-delims / sub-delims
    gen-delims    = \":\" / \"/\" / \"?\" / \"#\" / \"[\" / \"]\" / \"@\"
    sub-delims    = \"!\" / \"$\" / \"&\" / \"'\" / \"(\" / \")\"
                  / \"*\" / \"+\" / \",\" / \";\" / \"=\"

    Each of the reserved characters is reserved in some component of a URL,
    but not necessarily in all of them.

    The quote function %-escapes all characters that are neither in the
    unreserved chars (\"always safe\") nor the additional chars set via the
    safe arg.

    The default for the safe arg is '/'. The character is reserved, but in
    typical usage the quote function is being called on a path where the
    existing slash characters are to be preserved.

    Python 3.7 updates from using RFC 2396 to RFC 3986 to quote URL strings.
    Now, \"~\" is included in the set of unreserved characters.

    string and safe may be either str or bytes objects. encoding and errors
    must not be specified if string is a bytes object.

    The optional encoding and errors parameters specify how to deal with
    non-ASCII characters, as accepted by the str.encode method.
    By default, encoding='utf-8' (characters are encoded with UTF-8), and
    errors='strict' (unsupported characters raise a UnicodeEncodeError).
    "
  [string & {:keys [safe encoding errors]
                       :or {safe "/"}} ]
    (py/call-attr-kw mldata "quote" [string] {:safe safe :encoding encoding :errors errors }))

(defn setup-module 
  ""
  [ module ]
  (py/call-attr mldata "setup_module"  module ))

(defn teardown-module 
  ""
  [ module ]
  (py/call-attr mldata "teardown_module"  module ))

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
    (py/call-attr-kw mldata "urlopen" [url data] {:timeout timeout :cafile cafile :capath capath :cadefault cadefault :context context }))
