import re
import numpy as np
import json
#anger,anticipation,disgust,fear,joy,negative,positive,sadness,surprise,trust

def build_data(data, w, vals):
	total = sum(vals)
	if w not in data: data[w] = {}	
	if total == 0 :
		data[w]['anger'] = 0
		data[w]['anticipation'] = 0
		data[w]['disgust'] = 0
		data[w]['fear'] = 0
		data[w]['joy'] = 0
		data[w]['neg'] = 0
		data[w]['pos'] = 0
		data[w]['sad'] = 0
		data[w]['surprise'] = 0
		data[w]['trust'] = 0
	else:
		data[w]['anger'] = float(vals[0]) / float(total) 
		data[w]['anticipation'] = float(vals[1]) / float(total)
		data[w]['disgust'] = float(vals[2]) / float(total)
		data[w]['fear'] = float(vals[3]) / float(total)
		data[w]['joy'] = float(vals[4]) / float(total)
		data[w]['neg'] = float(vals[5]) / float(total)
		data[w]['pos'] = float(vals[6]) / float(total)
		data[w]['sad'] = float(vals[7]) / float(total)
		data[w]['surprise'] = float(vals[8]) / float(total)
		data[w]['trust'] = float(vals[9]) / float(total)
		

data = {}
with open('emotion.csv', 'r') as f:
	for line in f:
		if 'anger' in line: continue
		line = re.sub('[\[\]"\\n]', '', line)	
		items = line.split(',')
		w , vals = items[0], [float(x) for x in items[1:]]
		build_data(data, w, vals)

with open('lexicon.json', 'w') as f:
     json.dump(data, f, indent = 2)


