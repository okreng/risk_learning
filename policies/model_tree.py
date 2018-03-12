"""
This module holds the function for creating and loading models
"""

import sys

def model_tree(model_instance, module_name, action_type_nome):
	"""
	Returns a model string of the model name to be loaded
	:param model_instance: string in the form c1-c2-c3... branches from head
	:param module_name: string the name of the module, to locate the log folder
	:param action_type_name: string the name of the action type, to locate the log folder
	'head' is returned if no model name exists
	

	
