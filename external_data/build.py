import re
import numpy as np
#anger,anticipation,disgust,fear,joy,negative,positive,sadness,surprise,trust
file = open('emotion.csv', 'r')
out1 = open('ex_pos_lexicon', 'w')
out2 = open('ex_neg_lexicon', 'w')
for line in file.readlines():
	if 'anger' in line: continue
	line = re.sub('[\[\]"\\n]', '', line)	
	items = line.split(',')
	w , vals = items[0], [float(x) for x in items[1:]]
	total = sum(vals) 
	if total > 0:
		neg, pos = vals[5] / total, vals[6] / total
	else:
		neg, pos = 0, 0
	out1.write('%s,%f\n' % (w, pos))
	out2.write('%s,%f\n' % (w, neg))
out1.close()
out2.close()
file.close()

