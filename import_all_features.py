import os

"""Notice: Direct Use is Required

These import codes won't work when not in top block.
Neither in functions nor "if block" can import be successful.
So you have to copy them at the top where needed.
"""

path = 'features'  # path may differ.
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


if __name__ == '__main__':

	# Test. And this is how to use:
	test_features.cal()

