import re 
import numpy as np
file1 = open('../sadList','r')
file2 = open('../nonSadList', 'r')
domain = 'random70_all'
out = open(domain + '/testData1', 'w')
out1 = open(domain + '/sadList', 'w')
out2 = open(domain + '/nonSadList', 'w')
out3 = open(domain + '/trainData', 'w')

strs1, strs2 = [], []
for line in file1.readlines():
	line = re.sub(',', ' ', line)
	strs1 += line.strip('"').strip(',').strip('\n'),
for line in file2.readlines():
	line = re.sub(',', ' ', line)
	strs2 += line.strip('"').strip(',').strip('\n'),

#n = 1000
#n = 6000
n1, n2 = len(strs1), len(strs2)
n11, n22 = n1 * 3 / 10,  n2 * 3 / 10
np.random.shuffle(strs1)
np.random.shuffle(strs2)
test_strs1 ,test_strs2 = strs1[:n11], strs2[:n22]
train_strs1, train_strs2 = strs1[n11:n1], strs2[n22:n2]
print len(test_strs1), len(test_strs2), len(train_strs1), len(train_strs2)
for s1, s2 in zip(test_strs1, test_strs2):
	out.write('%s, 1\n' % (s1))
	out.write('%s, 0\n' % (s2))
out.close()

for s1, s2 in zip(train_strs1, train_strs2):
	out1.write('%s\n' %(s1))
	out2.write('%s\n' %(s2))
	out3.write('%s, 1\n' % (s1))
	out3.write('%s, 0\n' % (s2))
out1.close()
out2.close()
out3.close()

 
