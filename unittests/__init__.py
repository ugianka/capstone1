

import unittest
import getopt
import sys
import os
from unittests.LogTests import LogTest


# parse inputs
try:
    optlist, args = getopt.getopt(sys.argv[1:], 'v')
except getopt.GetoptError:
    print(getopt.GetoptError)
    print(sys.argv[0] + "-v")
    print("... the verbose flag (-v) may be used")
    sys.exit()

VERBOSE = False
RUNALL = False

sys.path.append(os.path.realpath(os.path.dirname(__file__)))

for o, a in optlist:
    if o == '-v':
        VERBOSE = True


# # api tests
from ApiTests import *
ApiTestSuite = unittest.TestLoader().loadTestsFromTestCase(ApiTest)


# # api tests
from ModelTests import *
ModelTestSuite = unittest.TestLoader().loadTestsFromTestCase(ModelTest)


# api tests
from LogTests import *
LogTestSuite = unittest.TestLoader().loadTestsFromTestCase(LogTest)


# MainSuite = unittest.TestSuite([ApiTestSuite, ModelTestSuite, LogTestSuite])
MainSuite = unittest.TestSuite([LogTestSuite, ApiTestSuite, ModelTestSuite])
