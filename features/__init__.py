import os

"""
Codes in this file help in a way to import all feature at once.
Clear this file if you don't want so.

Notice: Direct Use is Required
These import codes won't work when not in first level of the file.
Neither in functions nor in "if block" can the import be successful.
"""

# path of features, change to fit different working directory.
path = 'features'

# get module name list
model_list = [x.replace('.py', '') for x in os.listdir(path) if x.endswith('.py')]
if '__init__' in model_list:
    model_list.remove('__init__')

# import all modules in this feature folder
print('')
print('-' * 41)
print('[Batch import implementing on "features"]', end='\n\n')
for model in model_list:
    command = 'from features import %s' % model
    exec(command)  # see? so don't name sys command on files, dangerous.
    print(command)
print('')
print('-' * 41, '\n')
