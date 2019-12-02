(ns sklearn.datasets.kddcup99.GzipFile
  "The GzipFile class simulates most of the methods of a file object with
    the exception of the truncate() method.

    This class only supports opening files in binary mode. If you need to open a
    compressed file in text mode, use the gzip.open() function.

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce kddcup99 (import-module "sklearn.datasets.kddcup99"))

(defn GzipFile 
  "The GzipFile class simulates most of the methods of a file object with
    the exception of the truncate() method.

    This class only supports opening files in binary mode. If you need to open a
    compressed file in text mode, use the gzip.open() function.

    "
  [filename mode & {:keys [compresslevel fileobj mtime]
                       :or {compresslevel 9}} ]
    (py/call-attr-kw kddcup99 "GzipFile" [filename mode] {:compresslevel compresslevel :fileobj fileobj :mtime mtime }))

(defn close 
  ""
  [ self  ]
  (py/call-attr self "close"  self  ))

(defn closed 
  ""
  [ self ]
    (py/call-attr self "closed"))

(defn filename 
  ""
  [ self ]
    (py/call-attr self "filename"))

(defn fileno 
  "Invoke the underlying file object's fileno() method.

        This will raise AttributeError if the underlying file object
        doesn't support fileno().
        "
  [ self  ]
  (py/call-attr self "fileno"  self  ))

(defn flush 
  ""
  [self  & {:keys [zlib_mode]
                       :or {zlib_mode 2}} ]
    (py/call-attr-kw self "flush" [] {:zlib_mode zlib_mode }))

(defn mtime 
  "Last modification time read from stream, or None"
  [ self ]
    (py/call-attr self "mtime"))

(defn peek 
  ""
  [ self n ]
  (py/call-attr self "peek"  self n ))

(defn read 
  ""
  [self  & {:keys [size]
                       :or {size -1}} ]
    (py/call-attr-kw self "read" [] {:size size }))

(defn read1 
  "Implements BufferedIOBase.read1()

        Reads up to a buffer's worth of data is size is negative."
  [self  & {:keys [size]
                       :or {size -1}} ]
    (py/call-attr-kw self "read1" [] {:size size }))

(defn readable 
  ""
  [ self  ]
  (py/call-attr self "readable"  self  ))

(defn readline 
  ""
  [self  & {:keys [size]
                       :or {size -1}} ]
    (py/call-attr-kw self "readline" [] {:size size }))

(defn rewind 
  "Return the uncompressed stream file position indicator to the
        beginning of the file"
  [ self  ]
  (py/call-attr self "rewind"  self  ))

(defn seek 
  ""
  [self offset & {:keys [whence]
                       :or {whence 0}} ]
    (py/call-attr-kw self "seek" [offset] {:whence whence }))

(defn seekable 
  ""
  [ self  ]
  (py/call-attr self "seekable"  self  ))

(defn writable 
  ""
  [ self  ]
  (py/call-attr self "writable"  self  ))

(defn write 
  ""
  [ self data ]
  (py/call-attr self "write"  self data ))
