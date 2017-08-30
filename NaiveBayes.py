import os
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
		    tokens.append(x)
	tokens = [ str(lemma(x).lower()) \
				for x in tokens
				if isinstance(x, unicode)
				and x not in stopwords.words('english') 
				and len(x) > 1 
				and len(wn.synsets(x)) > 0]
	return tokens

def get_data(train_file, model):
	lines = [ line for line in open(train_file, 'r').readlines()]
	opt = 2 if model == 'bigram' else 1
	tokens = [ (token, int(line.strip('\n').split(',')[1])) for line in lines for token in extract_tokens(line.split(',')[0], opt)]
	#for token in tokens:
	#	print token[0], token[1]
	train_set = [ (get_feature(token[0]), token[1])  for token in tokens]
	return train_set

def prediction_NB(infile, model):
	error, count = 0, 0
	opt = 2 if model == 'bigram' else 1
	for item in open(infile, 'r').readlines():
		line, y = item.split(',')
		count += 1
		tokens, y = bag_of_words(extract_tokens(line, opt)), int(y)
		prob_pos = classifier.prob_classify(tokens).prob('pos')
		prob_neg = classifier.prob_classify(tokens).prob('neg')
		predict = 1 if prob_pos >= prob_neg else 0
		error += abs(predict - y) 
	return 1 - (float(error) / float(count))
	
def build_dict_NB(infile):
	wordDict = {}
	for item in open(infile, 'r').readlines():
		line, _ = item.split(',')
		tokens = extract_tokens(line)
		for token in tokens:
			if token in wordDict: continue
			bag = bag_of_words(token)
			prob_pos = classifier.prob_classify(bag).prob('pos')
			wordDict[token] = prob_pos
	return wordDict

def build_dict_external_data(INFILE):
	##['anger,anticipation,disgust,fear,joy,negative,positive,sadness,surprise,trust']	
	lexicon = {}
	with open(INFILE, 'r') as infile:
		rows = csv.reader(infile, delimiter='\t')
		i = 0
		for row in rows:
			row = row[0].split(',')
			if i == 0: emotionsList = row	
			if i > 0:
				vals = [float(x) for x in row[1:]]
				#onyl focus 6 emotions
				count = np.sum(vals[:5] + [vals[7]])
				tmp = np.array(vals) / count if count > 0 else np.array([0] * 10)
				lexicon[row[0]], k = {}, 0
				for key in emotionsList:
					lexicon[row[0]][key], k = tmp[k], k + 1
			i += 1
	dict = {}
	for word in lexicon:
		dict[word] = lexicon[word]['sadness']
	return dict

def write_dict(test_file, output_file, external_file):
	out = open(output_file, 'w')

	#build two different dictionaries, one is with Naive Bayes, 
	# the other one is with external dictionary
	NB_dict = build_dict_NB(test_file)
	external_dict = build_dict_external_data(external_file)

	#write the content of two dictionaries to the output file
	for word in external_dict:
		out.write('%s, %f\n' % (word, external_dict[word]))
	for word in NB_dict:
		if word not in external_dict:
			out.write('%s, %f\n' % (word, NB_dict[word]))
	out.close()
	
def build_NB_lexicon(path, output_file):
	input_file = '%s/all/testData' % (path)
	external_file = 'external_data/emotion.csv'
	write_dict(test_file, output_file, external_file)

if __name__ == '__main__':
	text = {}
	domain = 'random70_all'
	path = 'Data'
	train_file = '%s/%s/trainData' % (path, domain)
	test_file = '%s/%s/testData' % (path, domain)
	model = 'bigram1'
	pickled_classifier = 'trained/classifier-NaiveBayes-%s_%s.pickle' % (domain, model)
	if not os.path.exists(pickled_classifier):
		print 'classifier does not exist, now train and write'
		train_set = get_data(train_file, model)
		print len(train_set)
		classifier = NaiveBayesClassifier.train(train_set)
		pickle.dump(classifier, open(pickled_classifier, 'w'))
	else:
		classifier = pickle.load(open(pickled_classifier, 'r'))

	#build a NB model and test the performance	
	#print prediction_NB(test_file, model)

	#build a NB lexicon
	#output_file = 'sadList'
	#build_NB_lexicon(path, output_file)


