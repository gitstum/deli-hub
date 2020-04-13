#--coding:utf-8--

import talib

for x in dir(talib):
    print(x)

print('-----------------------------------')

g = talib.get_function_groups()

for x in g:
    print(x)