import json
from nltk import corpus
from nltk.corpus import wordnet as wn
import numpy as np
from lib import *
import heapq

EMOTIONS = ['joy', 'sad', 'disgust', 'anger', 'surprise', 'fear']
LEXICON = 'lexicon/Semeval_2007/lexicon.json'

def predict_sentence(wp, classifiers, testSetence, emotions):
	return [test_line(wp, classifiers[emo], testSetence, emo) for emo in emotions]

def predict_text(text, lexicon, wp, classifiers):
	text = text.split(". ")
	num_sentence = len(text)
	emotions_score = [[] for _ in range(6)]
	for sentence in text:
		total, none_word_in_lexicon = 0, True
		for w in sentence.split(' '):
			w = wn.morphy(w.strip('\.\n'))
			if w in lexicon:
				total = sum([float(lexicon[w][emotion]) for emotion in lexicon[w]])
				if total > 0: 
					none_word_in_lexicon = False
					break
		if not none_word_in_lexicon:
			score = predict_sentence(wp, classifiers, sentence, EMOTIONS)
			for i in range(len(EMOTIONS)):
				if len(emotions_score[i]) >= num_sentence // 2:
					heapq.heappop(emotions_score[i])
				heapq.heappush(emotions_score[i], score[i])				
	emotions_score = np.array(emotions_score)							
	return np.mean(emotions_score, axis = 1)

def load_json(in_file):
	with open(in_file, 'r') as json_file:
		return json.load(json_file)
def load_text(in_file):
	with open(in_file, 'r') as text_file:
		return text_file.read()
lexicon = load_json(LEXICON)
ml_model = 'logistic'
path = 'pickled'
domain = 'Semeval_2007'
model = 'model1'
classifiers = {}
for emotion in EMOTIONS:
	classifiers[emotion], wp = predict_init(path, domain, model, emotion, ml_model)


text = load_text('test_text/ex1')
print predict_text(text, lexicon, wp, classifiers)
