import os

"""Notice: Direct Use is Required

These import codes won't work when not in first level of the file.
Neither in functions nor in "if block" can import be successful.
So copy them at the top if you need to import all features at once.
"""


# path of features, change if needed.
path = 'features'


# get module name list
model_list = [x.replace('.py', '') for x in os.listdir(path) if x.endswith('.py')]
if '__init__' in model_list:
	model_list.remove('__init__')


# import modules
print('-' * 50)
for model in model_list:
	command = 'from %s import %s' % (path, model)
	exec(command)
	print(command)
print('-' * 50)


if __name__ == '__main__':

	# Test. And this is how to use:
	test_features.cal()

