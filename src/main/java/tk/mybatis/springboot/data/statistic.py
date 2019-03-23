import linecache
import codecs
import random

data = linecache.getlines("./home.arff")

size_list = [500,1000,5000,10000]
for size in size_list:
    result = codecs.open('./home_'+ str(size) +'.arff', 'w', 'utf-8')
    cnt = 0    
    s = set()
    while len(s) < size:
        s.add(random.randint(35, 133235))
    l = list(s)

    l.sort()

    for line in data:
        cnt += 1
        if line[0] == "@" or len(line) == 0:
            result.write(line)
        elif len(l) > 0 and cnt == l[0]:
            l.remove(l[0])
            result.write(line)
        elif len(l) == 0:
            break



