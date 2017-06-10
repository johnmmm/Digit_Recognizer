import codecs


f1=codecs.open('/Users/mac/Desktop/programme/Python/dlworking/testing_answer.csv')
s1=f1.read()
arr1=s1.split('\n')

f2=codecs.open('/Users/mac/Desktop/programme/Python/dlworking/testing_answer1.csv')
s2=f2.read()
arr2=s2.split('\n')

f3=codecs.open('/Users/mac/Desktop/programme/Python/dlworking/testing_answer2.csv')
s3=f3.read()
arr3=s3.split('\n')

answer=arr1[0]
ll1=arr2[1].split(',')
print ll1[1]
for i in range(1,28001):
    ll1=arr1[i].split(',')
    ll2=arr2[i].split(',')
    ll3=arr3[i].split(',')
    if ll1[1]==ll2[1]:
        answer = answer + '\n' + str(i) + ',' + str(ll1[1])
    elif ll2[1]==ll3[1]:
        answer = answer + '\n' + str(i) + ',' + str(ll2[1])
    elif ll3[1]==ll1[1]:
        answer = answer + '\n' + str(i) + ',' + str(ll3[1])
    else:
        answer = answer + '\n' + str(i) + ',' + str(ll3[1])

fout = codecs.open('/Users/mac/Desktop/programme/Python/dlworking/output_final.csv','w','utf-8')
fout.write(answer)
fout.close()
