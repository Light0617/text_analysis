import os
import json
import csv
import pickle
import numpy as np
from pattern.en import lemma
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify import NaiveBayesClassifier, MaxentClassifier
from nltk.classify.util import accuracy

#anger,anticipation,disgust,fear,joy,negative,positive,sadness,surprise,trust

def get_feature(word):
    return dict([(word, True)])

def bag_of_words(words):
	return dict([(word, True) for word in words])

def extract_tokens(line, opt = 1):
	stemmer = PorterStemmer()
	tokenizer = WordPunctTokenizer()
	lemmatizer = WordNetLemmatizer()
	if not isinstance(line, unicode):
		line = unicode(line, errors='ignore')
	tokens = tokenizer.tokenize(line)
	# bigram 
	if opt == 2:
		bigram_finder = BigramCollocationFinder.from_words(tokens)
		bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)
		for bigram_tuple in bigrams:
			x = "%s %s" % bigram_tuple
			print x
			tokens.append(x)
	tokens = [ str(lemma(x).lower()) \
				for x in tokens
				if isinstance(x, unicode)
				and x not in stopwords.words('english') 
				and len(x) > 1 
				and len(wn.synsets(x)) > 0]
	return tokens

def get_data(infile, model, emotion, opt = 1):
	opt = 2 if model == 'bigram' else 1
	with open(infile, 'r') as f:
		data = json.load(f)
		opt = 2 if model == 'bigram' else 1
		tokens = [ (token, data[s][emotion]) for s in data for token in extract_tokens(s, opt)]
		X = [ (get_feature(token[0]), token[1] > 0)  for token in tokens]
		return X

def prediction_NB(domain, model, emotion, part, classifier):
#def prediction_NB(testData, classifier):
	error, count = 0, 0
	opt = 2 if model == 'bigram' else 1
	infile = 'Data/%s/%s.json' % (domain, part)
	data = {}
	with open(infile, 'r') as f:
		data = json.load(f)
	pos = 0
	for w in data:
		line, y = w, data[w][emotion]
		tokens, y = bag_of_words(extract_tokens(line, opt)), int(y) > 0
		#print tokens, y
		count += 1
		prob_pos = classifier.prob_classify(tokens).prob('pos')
		prob_neg = classifier.prob_classify(tokens).prob('neg')
		predict = 1 if prob_pos >= prob_neg else 0
		error += abs(predict - y) 
		pos += prob_pos
	print pos
	return 1 - (float(error) / float(count))
	
def trian_NB_model(trainData, pickled_classifier):
	if not os.path.exists(pickled_classifier):
		print 'classifier does not exist, now train and write'
		classifier = NaiveBayesClassifier.train(trainData)
		pickle.dump(classifier, open(pickled_classifier, 'w'))
	else:
		classifier = pickle.load(open(pickled_classifier, 'r'))
	return classifier

def get_emotion_data(domain, model, emotion, part, pickled_data):
	if not os.path.exists(pickled_data):
		print '%s pickled_data does not exist, now get and write' % (part)
		infile = 'Data/%s/%s.json' % (domain, part)
		data = get_data(infile, model, emotion, opt = 1)
		pickle.dump(data, open(pickled_data, 'w'))
	else:
		data = pickle.load(open(pickled_data, 'r'))
	return data
if __name__ == '__main__':

	domain = 'Semeval_2007'
	model = 'bigram'
	emolist = 'joy,sad,disgust,anger,surprise,fear'
	emotions = emolist.split(',')
	for emotion in emotions:
		print emotion
		pickled_classifier = 'pickled/%s/NaiveBayes/%s/%s_%s.pickle' % \
					(domain, emotion, 'classifier', model)
		pickled_trainData = 'pickled/%s/NaiveBayes/%s/%s_%s.pickle' % \
					(domain, emotion, 'trainData', model)
		pickled_testdData = 'pickled/%s/NaiveBayes/%s/%s_%s.pickle' % \
					(domain, emotion, 'testdData', model)


		trainData = get_emotion_data(domain, model, emotion, 'trainData', pickled_trainData)
		classifier = trian_NB_model(trainData, pickled_classifier)
		#given a sentence, to calculate the performance
		#testData = get_emotion_data(domain, model, emotion, 'testData', pickled_testdData)
		#measure performance
		#print emotion, prediction_NB(testData, classifier)
		#print emotion, prediction_NB(domain, model, emotion, 'testData', classifier)
		print emotion, prediction_NB(domain, model, emotion, 'trainData', classifier)
		break	

