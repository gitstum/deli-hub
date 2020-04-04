import os

"""
Codes in this file help in a way to import all feature at once.

Notice: Direct Use is Required
These import codes won't work when not in first level of the file.
Neither in functions nor in "if block" can import be successful.

Clear this file if you don't want to import all features at once.
"""


# get module name list
model_list = [x.replace('.py', '') for x in os.listdir('features') if x.endswith('.py')]
if '__init__' in model_list:
    model_list.remove('__init__')

# import modules
print('-' * 50)
for model in model_list:
    command = 'from features import %s' % model
    exec(command)
    print(command)
print('-' * 50, '\n')
