import os


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

test_features.cal()


def test():
	test_features.cal()


if __name__ == '__main__':
    test_features.cal()
    test()