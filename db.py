import json
from collections import OrderedDict

d = {}
d['a'] = 1
d['b'] = 2
d['c'] = 3
d['d'] = 4
json.dump(d, open('db.json', 'w'))