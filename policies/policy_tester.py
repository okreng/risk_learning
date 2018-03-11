"""
A file for testing different state vector inputs on policies
This function will print out which state vectors are being tested and the results
Policy functions must share name with file
"""

import numpy as np
import sys
import argparse
import importlib


def parse_arguments():
	# This function helps main read command line arguments
	parser = argparse.ArgumentParser(description=
		'Risk Environment Argument Parser')
	parser.add_argument('--dir',dest='directory',type=str)
	parser.add_argument('--pol',dest='policy',type=str)
	return parser.parse_args()

def main(args):
	"""
	Reads a policy filepath and runs state vectors on the given policy
	Prints results to console
	:param policy_filepath: string that contains directory and policy function dir.policy
	:return : none
	"""

	#import attack.max_success as pol0
	args = parse_arguments()
	directory = args.directory
	policy = args.policy
	module_name = directory + '.' + policy
	importlib.import_module(module_name, package=None)


	

	return



if __name__ == '__main__':
    main(sys.argv)
