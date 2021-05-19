import sys
from random import randint

f = open(sys.argv[2], "w")
for i in range(randint(0, 5)):
    x = randint(0, 1199)
    y = randint(0, 1199)
    f.write(f'{x},{y}\n')
f.close()
