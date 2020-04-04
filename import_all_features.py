import os


if __name__ == '__main__':

	"""Notice: Direct Use is Required

	When packed into a function, these import codes won't work outside the function.
	So you have to copy them to where needed.
	"""

	path = 'features'   # path may differ.
	model_list = []

	for x in os.listdir(path):
		if x.endswith('.py'):
			if x == '__init__.py': continue
			model_list.append(x.replace('.py', ''))

	print('-' * 50)
	for model in model_list:
		command = 'from %s import %s' % (path, model)
		exec(command)
		print(command)
	print('-' * 50)

	# how to use - call the filename:
	# test_features.cal()
