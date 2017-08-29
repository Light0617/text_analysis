import re
file1 = open('sadList', 'r')
file2 = open('nonSadList', 'r')
out = open('trainData', 'w')
for line in file1.readlines():
	line = re.sub('[,\"]', ' ', line)
	line = line.strip('\n')
	out.write('%s, 1\n' % (line))
for line in file2.readlines():
	line = re.sub('[,\"]', ' ', line)
	line = line.strip('\n')
	out.write('%s, 0\n' % (line))
file1.close()
file2.close()
out.close()
