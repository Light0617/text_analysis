from textblob import TextBlob
import numpy as np
import heapq
import os, sys
import csv

def predict_text(text):
	blob = TextBlob(text)
	num_sentence = len(text)
	scores = []
	for sentence in blob.sentences:
		score = sentence.sentiment.polarity
		if len(scores) > 2 and len(scores) >= num_sentence // 2:
			heapq.heappop(scores)
		heapq.heappush(scores, score)
	scores = np.array(scores)							
	return np.mean(scores)

def load_text(in_file):
	with open(in_file, 'r') as text_file:
		return text_file.read()

if __name__ == '__main__':
	in_dir_name, out_dir_name = sys.argv[1], sys.argv[2]
	results, files = [], []
	for file in os.listdir(in_dir_name):
		print file
		text = load_text('%s/%s' % (in_dir_name, file))
		results.append(predict_text(text))
		files.append(file)

	with open('%s_%s_polarity.csv' %(out_dir_name, file), 'wb') as output:
		wr = csv.writer(output, quoting=csv.QUOTE_ALL)
		wr.writerow(['file', 'positive'])
		for item, f_name in zip(results, files):
			wr.writerow([f_name] + [item])


