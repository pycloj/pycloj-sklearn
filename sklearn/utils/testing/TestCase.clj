(ns sklearn.utils.testing.TestCase
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce testing (import-module "sklearn.utils.testing"))

(defn TestCase 
  ""
  [ & {:keys [methodName]
       :or {methodName "runTest"}} ]
  
   (py/call-attr-kw testing "TestCase" [] {:methodName methodName }))

(defn addCleanup 
  "Add a function, with arguments, to be called when the test is
        completed. Functions added are called on a LIFO basis and are
        called after tearDown on test failure or success.

        Cleanup items are called even if setUp fails (unlike tearDown)."
  [ self  ]
  (py/call-attr self "addCleanup"  self  ))

(defn addTypeEqualityFunc 
  "Add a type specific assertEqual style function to compare a type.

        This method is for use by TestCase subclasses that need to register
        their own type equality functions to provide nicer error messages.

        Args:
            typeobj: The data type to call this function on when both values
                    are of the same type in assertEqual().
            function: The callable taking two arguments and an optional
                    msg= argument that raises self.failureException with a
                    useful error message when the two arguments are not equal.
        "
  [ self typeobj function ]
  (py/call-attr self "addTypeEqualityFunc"  self typeobj function ))

(defn assertAlmostEqual 
  "Fail if the two objects are unequal as determined by their
           difference rounded to the given number of decimal places
           (default 7) and comparing to zero, or by comparing that the
           difference between the two objects is more than the given
           delta.

           Note that decimal places (from zero) are usually not the same
           as significant digits (measured from the most significant digit).

           If the two objects compare equal then they will automatically
           compare almost equal.
        "
  [ self first second places msg delta ]
  (py/call-attr self "assertAlmostEqual"  self first second places msg delta ))

(defn assertAlmostEquals 
  ""
  [ self  ]
  (py/call-attr self "assertAlmostEquals"  self  ))

(defn assertCountEqual 
  "An unordered sequence comparison asserting that the same elements,
        regardless of order.  If the same element occurs more than once,
        it verifies that the elements occur the same number of times.

            self.assertEqual(Counter(list(first)),
                             Counter(list(second)))

         Example:
            - [0, 1, 1] and [1, 0, 1] compare equal.
            - [0, 0, 1] and [0, 1] compare unequal.

        "
  [ self first second msg ]
  (py/call-attr self "assertCountEqual"  self first second msg ))

(defn assertDictContainsSubset 
  "Checks whether dictionary is a superset of subset."
  [ self subset dictionary msg ]
  (py/call-attr self "assertDictContainsSubset"  self subset dictionary msg ))

(defn assertDictEqual 
  ""
  [ self d1 d2 msg ]
  (py/call-attr self "assertDictEqual"  self d1 d2 msg ))

(defn assertEqual 
  "Fail if the two objects are unequal as determined by the '=='
           operator.
        "
  [ self first second msg ]
  (py/call-attr self "assertEqual"  self first second msg ))

(defn assertEquals 
  ""
  [ self  ]
  (py/call-attr self "assertEquals"  self  ))

(defn assertFalse 
  "Check that the expression is false."
  [ self expr msg ]
  (py/call-attr self "assertFalse"  self expr msg ))

(defn assertGreater 
  "Just like self.assertTrue(a > b), but with a nicer default message."
  [ self a b msg ]
  (py/call-attr self "assertGreater"  self a b msg ))

(defn assertGreaterEqual 
  "Just like self.assertTrue(a >= b), but with a nicer default message."
  [ self a b msg ]
  (py/call-attr self "assertGreaterEqual"  self a b msg ))

(defn assertIn 
  "Just like self.assertTrue(a in b), but with a nicer default message."
  [ self member container msg ]
  (py/call-attr self "assertIn"  self member container msg ))

(defn assertIs 
  "Just like self.assertTrue(a is b), but with a nicer default message."
  [ self expr1 expr2 msg ]
  (py/call-attr self "assertIs"  self expr1 expr2 msg ))

(defn assertIsInstance 
  "Same as self.assertTrue(isinstance(obj, cls)), with a nicer
        default message."
  [ self obj cls msg ]
  (py/call-attr self "assertIsInstance"  self obj cls msg ))

(defn assertIsNone 
  "Same as self.assertTrue(obj is None), with a nicer default message."
  [ self obj msg ]
  (py/call-attr self "assertIsNone"  self obj msg ))

(defn assertIsNot 
  "Just like self.assertTrue(a is not b), but with a nicer default message."
  [ self expr1 expr2 msg ]
  (py/call-attr self "assertIsNot"  self expr1 expr2 msg ))

(defn assertIsNotNone 
  "Included for symmetry with assertIsNone."
  [ self obj msg ]
  (py/call-attr self "assertIsNotNone"  self obj msg ))

(defn assertLess 
  "Just like self.assertTrue(a < b), but with a nicer default message."
  [ self a b msg ]
  (py/call-attr self "assertLess"  self a b msg ))

(defn assertLessEqual 
  "Just like self.assertTrue(a <= b), but with a nicer default message."
  [ self a b msg ]
  (py/call-attr self "assertLessEqual"  self a b msg ))

(defn assertListEqual 
  "A list-specific equality assertion.

        Args:
            list1: The first list to compare.
            list2: The second list to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.

        "
  [ self list1 list2 msg ]
  (py/call-attr self "assertListEqual"  self list1 list2 msg ))

(defn assertLogs 
  "Fail unless a log message of level *level* or higher is emitted
        on *logger_name* or its children.  If omitted, *level* defaults to
        INFO and *logger* defaults to the root logger.

        This method must be used as a context manager, and will yield
        a recording object with two attributes: `output` and `records`.
        At the end of the context manager, the `output` attribute will
        be a list of the matching formatted log messages and the
        `records` attribute will be a list of the corresponding LogRecord
        objects.

        Example::

            with self.assertLogs('foo', level='INFO') as cm:
                logging.getLogger('foo').info('first message')
                logging.getLogger('foo.bar').error('second message')
            self.assertEqual(cm.output, ['INFO:foo:first message',
                                         'ERROR:foo.bar:second message'])
        "
  [ self logger level ]
  (py/call-attr self "assertLogs"  self logger level ))

(defn assertMultiLineEqual 
  "Assert that two multi-line strings are equal."
  [ self first second msg ]
  (py/call-attr self "assertMultiLineEqual"  self first second msg ))

(defn assertNotAlmostEqual 
  "Fail if the two objects are equal as determined by their
           difference rounded to the given number of decimal places
           (default 7) and comparing to zero, or by comparing that the
           difference between the two objects is less than the given delta.

           Note that decimal places (from zero) are usually not the same
           as significant digits (measured from the most significant digit).

           Objects that are equal automatically fail.
        "
  [ self first second places msg delta ]
  (py/call-attr self "assertNotAlmostEqual"  self first second places msg delta ))

(defn assertNotAlmostEquals 
  ""
  [ self  ]
  (py/call-attr self "assertNotAlmostEquals"  self  ))

(defn assertNotEqual 
  "Fail if the two objects are equal as determined by the '!='
           operator.
        "
  [ self first second msg ]
  (py/call-attr self "assertNotEqual"  self first second msg ))

(defn assertNotEquals 
  ""
  [ self  ]
  (py/call-attr self "assertNotEquals"  self  ))

(defn assertNotIn 
  "Just like self.assertTrue(a not in b), but with a nicer default message."
  [ self member container msg ]
  (py/call-attr self "assertNotIn"  self member container msg ))

(defn assertNotIsInstance 
  "Included for symmetry with assertIsInstance."
  [ self obj cls msg ]
  (py/call-attr self "assertNotIsInstance"  self obj cls msg ))

(defn assertNotRegex 
  "Fail the test if the text matches the regular expression."
  [ self text unexpected_regex msg ]
  (py/call-attr self "assertNotRegex"  self text unexpected_regex msg ))

(defn assertNotRegexpMatches 
  ""
  [ self  ]
  (py/call-attr self "assertNotRegexpMatches"  self  ))

(defn assertRaises 
  "Fail unless an exception of class expected_exception is raised
           by the callable when invoked with specified positional and
           keyword arguments. If a different type of exception is
           raised, it will not be caught, and the test case will be
           deemed to have suffered an error, exactly as for an
           unexpected exception.

           If called with the callable and arguments omitted, will return a
           context object used like this::

                with self.assertRaises(SomeException):
                    do_something()

           An optional keyword argument 'msg' can be provided when assertRaises
           is used as a context object.

           The context manager keeps a reference to the exception as
           the 'exception' attribute. This allows you to inspect the
           exception after the assertion::

               with self.assertRaises(SomeException) as cm:
                   do_something()
               the_exception = cm.exception
               self.assertEqual(the_exception.error_code, 3)
        "
  [ self expected_exception ]
  (py/call-attr self "assertRaises"  self expected_exception ))

(defn assertRaisesRegex 
  "Asserts that the message in a raised exception matches a regex.

        Args:
            expected_exception: Exception class expected to be raised.
            expected_regex: Regex (re pattern object or string) expected
                    to be found in error message.
            args: Function to be called and extra positional args.
            kwargs: Extra kwargs.
            msg: Optional message used in case of failure. Can only be used
                    when assertRaisesRegex is used as a context manager.
        "
  [ self expected_exception expected_regex ]
  (py/call-attr self "assertRaisesRegex"  self expected_exception expected_regex ))

(defn assertRaisesRegexp 
  ""
  [ self  ]
  (py/call-attr self "assertRaisesRegexp"  self  ))

(defn assertRegex 
  "Fail the test unless the text matches the regular expression."
  [ self text expected_regex msg ]
  (py/call-attr self "assertRegex"  self text expected_regex msg ))

(defn assertRegexpMatches 
  ""
  [ self  ]
  (py/call-attr self "assertRegexpMatches"  self  ))

(defn assertSequenceEqual 
  "An equality assertion for ordered sequences (like lists and tuples).

        For the purposes of this function, a valid ordered sequence type is one
        which can be indexed, has a length, and has an equality operator.

        Args:
            seq1: The first sequence to compare.
            seq2: The second sequence to compare.
            seq_type: The expected datatype of the sequences, or None if no
                    datatype should be enforced.
            msg: Optional message to use on failure instead of a list of
                    differences.
        "
  [ self seq1 seq2 msg seq_type ]
  (py/call-attr self "assertSequenceEqual"  self seq1 seq2 msg seq_type ))

(defn assertSetEqual 
  "A set-specific equality assertion.

        Args:
            set1: The first set to compare.
            set2: The second set to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.

        assertSetEqual uses ducktyping to support different types of sets, and
        is optimized for sets specifically (parameters must support a
        difference method).
        "
  [ self set1 set2 msg ]
  (py/call-attr self "assertSetEqual"  self set1 set2 msg ))

(defn assertTrue 
  "Check that the expression is true."
  [ self expr msg ]
  (py/call-attr self "assertTrue"  self expr msg ))

(defn assertTupleEqual 
  "A tuple-specific equality assertion.

        Args:
            tuple1: The first tuple to compare.
            tuple2: The second tuple to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.
        "
  [ self tuple1 tuple2 msg ]
  (py/call-attr self "assertTupleEqual"  self tuple1 tuple2 msg ))

(defn assertWarns 
  "Fail unless a warning of class warnClass is triggered
           by the callable when invoked with specified positional and
           keyword arguments.  If a different type of warning is
           triggered, it will not be handled: depending on the other
           warning filtering rules in effect, it might be silenced, printed
           out, or raised as an exception.

           If called with the callable and arguments omitted, will return a
           context object used like this::

                with self.assertWarns(SomeWarning):
                    do_something()

           An optional keyword argument 'msg' can be provided when assertWarns
           is used as a context object.

           The context manager keeps a reference to the first matching
           warning as the 'warning' attribute; similarly, the 'filename'
           and 'lineno' attributes give you information about the line
           of Python code from which the warning was triggered.
           This allows you to inspect the warning after the assertion::

               with self.assertWarns(SomeWarning) as cm:
                   do_something()
               the_warning = cm.warning
               self.assertEqual(the_warning.some_attribute, 147)
        "
  [ self expected_warning ]
  (py/call-attr self "assertWarns"  self expected_warning ))

(defn assertWarnsRegex 
  "Asserts that the message in a triggered warning matches a regexp.
        Basic functioning is similar to assertWarns() with the addition
        that only warnings whose messages also match the regular expression
        are considered successful matches.

        Args:
            expected_warning: Warning class expected to be triggered.
            expected_regex: Regex (re.Pattern object or string) expected
                    to be found in error message.
            args: Function to be called and extra positional args.
            kwargs: Extra kwargs.
            msg: Optional message used in case of failure. Can only be used
                    when assertWarnsRegex is used as a context manager.
        "
  [ self expected_warning expected_regex ]
  (py/call-attr self "assertWarnsRegex"  self expected_warning expected_regex ))

(defn assert- 
  ""
  [ self  ]
  (py/call-attr self "assert_"  self  ))

(defn countTestCases 
  ""
  [ self  ]
  (py/call-attr self "countTestCases"  self  ))

(defn debug 
  "Run the test without collecting errors in a TestResult"
  [ self  ]
  (py/call-attr self "debug"  self  ))

(defn defaultTestResult 
  ""
  [ self  ]
  (py/call-attr self "defaultTestResult"  self  ))

(defn doCleanups 
  "Execute all cleanup functions. Normally called for you after
        tearDown."
  [ self  ]
  (py/call-attr self "doCleanups"  self  ))

(defn fail 
  "Fail immediately, with the given message."
  [ self msg ]
  (py/call-attr self "fail"  self msg ))

(defn failIf 
  ""
  [ self  ]
  (py/call-attr self "failIf"  self  ))

(defn failIfAlmostEqual 
  ""
  [ self  ]
  (py/call-attr self "failIfAlmostEqual"  self  ))

(defn failIfEqual 
  ""
  [ self  ]
  (py/call-attr self "failIfEqual"  self  ))

(defn failUnless 
  ""
  [ self  ]
  (py/call-attr self "failUnless"  self  ))

(defn failUnlessAlmostEqual 
  ""
  [ self  ]
  (py/call-attr self "failUnlessAlmostEqual"  self  ))

(defn failUnlessEqual 
  ""
  [ self  ]
  (py/call-attr self "failUnlessEqual"  self  ))

(defn failUnlessRaises 
  ""
  [ self  ]
  (py/call-attr self "failUnlessRaises"  self  ))

(defn id 
  ""
  [ self  ]
  (py/call-attr self "id"  self  ))

(defn run 
  ""
  [ self result ]
  (py/call-attr self "run"  self result ))

(defn setUp 
  "Hook method for setting up the test fixture before exercising it."
  [ self  ]
  (py/call-attr self "setUp"  self  ))

(defn shortDescription 
  "Returns a one-line description of the test, or None if no
        description has been provided.

        The default implementation of this method returns the first line of
        the specified test method's docstring.
        "
  [ self  ]
  (py/call-attr self "shortDescription"  self  ))

(defn skipTest 
  "Skip this test."
  [ self reason ]
  (py/call-attr self "skipTest"  self reason ))

(defn subTest 
  "Return a context manager that will return the enclosed block
        of code in a subtest identified by the optional message and
        keyword parameters.  A failure in the subtest marks the test
        case as failed but resumes execution at the end of the enclosed
        block, allowing further test code to be executed.
        "
  [self  & {:keys [msg]
                       :or {msg <object object at 0x110c349e0>}} ]
    (py/call-attr-kw self "subTest" [] {:msg msg }))

(defn tearDown 
  "Hook method for deconstructing the test fixture after testing it."
  [ self  ]
  (py/call-attr self "tearDown"  self  ))
