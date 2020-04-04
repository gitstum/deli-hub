import os
import time

'''
def import_all_models(path='features'):

    model_list = []

    for x in os.listdir(path):
        if x.endswith('.py'):
            if x == '__init__.py' \
            or x == 'all.py':
                continue
            model_list.append(x.replace('.py', ''))

    for model in model_list:
        command = 'from %s.%s import *' % (path, model)
        exec(command)
        print('%s | feature "%s" has been imported.' % (
            (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            model)
        )
'''