import linecache
import codecs
import random

data = linecache.getlines("./home.arff")
result0 = codecs.open('./home_0'+'.arff', 'w', 'utf-8')
result1 = codecs.open('./home_1'+'.arff', 'w', 'utf-8')

cnt = 0
size = 10000
deleteNum = 0;
s = set()
while len(s) < size * 2:
    s.add(random.randint(35, 133235))
l = list(s)

l.sort()

for line in data:
    cnt += 1
    if line[0] == "@" or len(line) == 0:
        result0.write(line)
        result1.write(line)
    elif len(l) > 0 and cnt == l[0]:
        l.remove(l[0])
        deleteNum+=1
        if deleteNum <= size:
            result0.write(line)
        else:
            result1.write(line)
    elif len(l) == 0:
        break



