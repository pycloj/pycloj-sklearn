(ns sklearn.feature-extraction.text.VectorizerMixin
  "Provides common code for text vectorizers (tokenization logic)."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce text (import-module "sklearn.feature_extraction.text"))

(defn VectorizerMixin 
  "Provides common code for text vectorizers (tokenization logic)."
  [  ]
  (py/call-attr text "VectorizerMixin"  ))

(defn build-analyzer 
  "Return a callable that handles preprocessing and tokenization"
  [ self  ]
  (py/call-attr self "build_analyzer"  self  ))

(defn build-preprocessor 
  "Return a function to preprocess the text before tokenization"
  [ self  ]
  (py/call-attr self "build_preprocessor"  self  ))

(defn build-tokenizer 
  "Return a function that splits a string into a sequence of tokens"
  [ self  ]
  (py/call-attr self "build_tokenizer"  self  ))

(defn decode 
  "Decode the input into a string of unicode symbols

        The decoding strategy depends on the vectorizer parameters.

        Parameters
        ----------
        doc : string
            The string to decode
        "
  [ self doc ]
  (py/call-attr self "decode"  self doc ))

(defn get-stop-words 
  "Build or fetch the effective stop words list"
  [ self  ]
  (py/call-attr self "get_stop_words"  self  ))
