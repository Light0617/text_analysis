import json
from nltk import corpus
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
from lib import *
import heapq

EMOTIONS = ['joy', 'sad', 'disgust', 'anger', 'surprise', 'fear']
LEXICON = 'lexicon/Semeval_2007/lexicon.json'

def predict_sentence(wp, classifiers, testSetence, emotions):
	return [test_line(wp, classifiers[emo], testSetence, emo) for emo in emotions]

def predict_text(text, lexicon = None, wp = None, classifiers = None):
    if not lexicon:
        lexicon = load_json(LEXICON)
    if not wp or not classifiers:
        ml_model = 'logistic'
        path = 'pickled'
        domain = 'Semeval_2007'
        model = 'model1'
        classifiers = {}
        for emotion in ['joy', 'sad', 'disgust', 'anger', 'surprise', 'fear']:
            classifiers[emotion], wp = predict_init(path, domain, model, emotion, ml_model)
        
	text = text.split(". ")
	num_sentence = len(text)
	emotions_score = [[] for _ in range(6)]
	for sentence in text:
		total, none_word_in_lexicon = 0, True
		for w in sentence.split(' '):
			try:
				 w = wn.morphy(w.strip('\.\n'))
			except:
				nltk.download('wordnet')
			if w in lexicon:
				total = sum([float(lexicon[w][emotion]) for emotion in lexicon[w]])
				if total > 0: 
					none_word_in_lexicon = False
					break
		if not none_word_in_lexicon:
			try:
				score = predict_sentence(wp, classifiers, sentence, EMOTIONS)
			except:
				nltk.download('stopwords')
				nltk.download('punkt')
				score = predict_sentence(wp, classifiers, sentence, EMOTIONS)
			for i in range(len(EMOTIONS)):
				if len(emotions_score[i]) > 2 and \
					len(emotions_score[i]) >= num_sentence // 2:
					heapq.heappop(emotions_score[i])
				heapq.heappush(emotions_score[i], score[i])				
	emotions_score = np.array(emotions_score)							
	return np.mean(emotions_score, axis = 1).tolist()

def load_json(in_file):
	with open(in_file, 'r') as json_file:
		return json.load(json_file)
def load_text(in_file):
	with open(in_file, 'r') as text_file:
		return text_file.read()


if __name__ == '__main__':
	lexicon = load_json(LEXICON)
	ml_model = 'logistic'
	path = 'pickled'
	domain = 'Semeval_2007'
	model = 'model1'
	classifiers = {}
	for emotion in EMOTIONS:
		classifiers[emotion], wp = predict_init(path, domain, model, emotion, ml_model)

	in_dir_name, out_dir_name = sys.argv[1], sys.argv[2]
	results, files = [], []
	for file in os.listdir(in_dir_name):
		print file
		text = load_text('%s/%s' % (in_dir_name, file))
		results.append(predict_text(text, lexicon, wp, classifiers))
		files.append(file)

	with open('%s_%s_emotion.csv' %(out_dir_name, file), 'wb') as output:
		wr = csv.writer(output, quoting=csv.QUOTE_ALL)
		wr.writerow(['file'] + EMOTIONS)
		for item, f_name in zip(results, files):
			print item, f_name
			wr.writerow([f_name] + item)



