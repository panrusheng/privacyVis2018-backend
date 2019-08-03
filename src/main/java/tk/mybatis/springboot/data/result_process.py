import linecache
import codecs
import json
import traceback

data = linecache.getlines("./result.txt")
result = codecs.open('./processed_result.txt', 'w', 'utf-8')

row = {'500':0, '1000':1, '5000':2, '10000':3}
col = {'5':0, '10':1, '15':2}
gbn = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
rec = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
grp = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
averageGBN = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
averageREC = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
totalNum = 0
for line in data:
    try:
        if line[3] == '-':
            totalNum += 1
            json.dump(gbn, codecs.open('./result/processed_result_gbn'+str(totalNum)+'.txt', 'w', 'utf-8'), ensure_ascii = False, indent=4)
            json.dump(rec, codecs.open('./result/processed_result_rec'+str(totalNum)+'.txt', 'w', 'utf-8'), ensure_ascii = False, indent=4)
            json.dump(grp, codecs.open('./result/processed_result_num'+str(totalNum)+'.txt', 'w', 'utf-8'), ensure_ascii = False, indent=4)        
            for i in range(len(row)):
                for j in range(len(col)):
                    averageGBN[i][j] += int(gbn[i][j])
                    averageREC[i][j] += int(rec[i][j])
            gbn = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
            rec = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
            grp = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        else:
            isGBN = line.split("_")[0] == "GBN"
            rowIndex = row[line.split("_")[1]]
            colIndex = col[line.split("_")[2].split(":")[0]]
            if isGBN:
                gbn[rowIndex][colIndex] = line.split(":")[1][:-3]
            else:
                rec[rowIndex][colIndex] = line.split(":")[1].split(", ")[0][:-2]
                grp[rowIndex][colIndex] = line.split(":")[1].split(", ")[1][:-1]
    except (Exception,BaseException) as e:
        print(line)
        print(traceback.format_exc())


for i in range(len(row)):
    for j in range(len(col)):
        averageGBN[i][j] = averageGBN[i][j] / totalNum
        averageREC[i][j] = averageREC[i][j] / totalNum

json.dump(averageGBN, codecs.open('./result/processed_result_averageGBN.txt', 'w', 'utf-8'), ensure_ascii = False, indent=4)        
json.dump(averageREC, codecs.open('./result/processed_result_averageREC.txt', 'w', 'utf-8'), ensure_ascii = False, indent=4)        

