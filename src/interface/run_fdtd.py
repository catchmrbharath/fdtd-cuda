#!/usr/bin
from optparse import OptionParser
from run_simulation import parse_file

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                   help="Path to the input file about structure")
(options, args) = parser.parse_args()
parse_file(options.filename)
