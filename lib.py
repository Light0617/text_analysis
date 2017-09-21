import os
import pickle
import re 
import sys
from get_features import *
from sklearn import linear_model, datasets, svm
from sklearn.metrics import precision_recall_fscore_support
def getData(infile):
    if 'sem' in infile:
        return getData1(infile)
    lines = [ line for line in open(infile, 'r').readlines()]
    text, Y = [line.split(',')[0] for line in lines], \
                        [int(line.split(',')[1]) for line in lines]
    wp = word_processor()
    emotion = 'sad'
    data = [wp.get_features1(x, emotion) for x in text]
    return [data, Y]

def getData1(infile):
	lines = [ line for line in open(infile, 'r').readlines()]
	text, Y = [line.split(',')[0] for line in lines], \
                        [int(line.split(',')[-2]) > 0 for line in lines]
	wp = word_processor()
	emotion = 'sad'
	data = [wp.get_features1(x, emotion) for x in text]
	return [data, Y]
def get_json_data(infile, emotion):
	data = {}
	wp = word_processor()
	with open(infile, 'r') as json_file:
		data = json.load(json_file)
	X = [wp.get_features1(s, emotion) for s in data]
	Y = [data[s][emotion] > 0 for s in data]
	return [X, Y]
	
def model1(trainX, trainY, ml_model, c = 1e5):
	if ml_model == 'logistic':
		print 'logistic'
		classifier = linear_model.LogisticRegression(C = c)
	else:
		print 'svm'
		classifier = svm.SVC()
	trainX, trainY= np.array(trainX), np.array(trainY)
	print trainX.shape, trainY.shape 
	classifier.fit(trainX, trainY)
	return classifier

def evaluate(classifier, X, Y):
	y_pred, y_true = classifier.predict(X), Y
	print 'precision ', precision_recall_fscore_support(y_true, y_pred, average='macro')[0]
	print 'recall ', precision_recall_fscore_support(y_true, y_pred, average='macro')[1]
	return classifier.score(X, Y)

def compare(classifier, X, Y):
	y_pred, y_true = classifier.predict(X), Y
	print 'pred, true'
	for y1, y2, x in zip(y_pred, y_true, X):
		if y1 != y2:
			print y1, y2

def test_line(wp, classifier, testSetence, emotion):
	vec = np.array(wp.get_features1(testSetence, emotion)).reshape(1, -1)
	return classifier.predict_proba(vec)[0,1]

def get_classifier(path, domain, model, emotion, pickled_classifier, ml_model):
	if not os.path.exists(pickled_classifier):
		print 'Classifier does not exist...........'
		pickled_trainX = path + '/%s/%s/%s/trainX.pickle' % (domain, model, emotion)
		pickled_trainY = path + '/%s/%s/%s/trainY.pickle' % (domain, model, emotion)
		trainX = pickle.load(open(pickled_trainX, 'r'))
		trainY = pickle.load(open(pickled_trainY, 'r'))
		classifier = model1(trainX, trainY, ml_model)	
		pickle.dump(classifier, open(pickled_classifier, 'w'))
	else:
		classifier = pickle.load(open(pickled_classifier, 'r'))
	return classifier

def get_wp(pickled_wp):
	if not os.path.exists(pickled_wp):
		print 'WP does not exist'
		wp = word_processor() 
		pickle.dump(wp, open(pickled_wp, 'w'))
	else:
		wp = pickle.load(open(pickled_wp, 'r'))
	return wp

def predict_emotion_text(wp, classifier, text, emotion):
	lines = [x for x in re.split('[!.?]+', text.strip('\n')) if len(x) > 0]
	scores = 0
	for line in lines:
		score = test_line(wp, classifier, line, emotion)
		print line
		print score
		scores += score
	return scores / float (len(lines))	

def predict_init(path, domain, model, emotion, ml_model):
	path = os.getcwd() + '/' + path
	print path, domain
	pickled_classifier = path + '/%s/%s/%s/%s_classifier.pickle' % (domain, model, emotion, ml_model)
	pickled_wp = path + '/%s/%s/%s/wp.pickle' % (domain, model, emotion)
	print pickled_classifier
	print pickled_wp
	return [get_classifier(path, domain, model, emotion, pickled_classifier, ml_model), get_wp(pickled_wp)]	

def predict_emotion_textList(textList, emotion, ml_model = 'logistic', path = 'pickled', domain = 'Semeval_2007', model = 'model1'):
	classifier, wp = predict_init(path, domain, model, emotion, ml_model)
	return [ predict_emotion_text(wp, classifier, text, emotion) for text in textList ]

def predict_emotion_fileList(fileList, emotion, ml_model = 'logistic', path = 'pickled', domain = 'Semeval_2007', model = 'model1'):
	classifier, wp = predict_init(path, domain, model, emotion, ml_model)
	res = []
	for target_file in fileList:
		with open(target_file, 'r') as text:
			res += predict_emotion_text(wp, classifier, text, emotion),
	return res

if __name__ == '__main__':
	'''
	path, domain = 'trained/', 'random70_all'
	if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
		print predict_emotion_fileList(['story1'], path, domain)
	else:	
		print predict_emotion_fileList([sys.argv[1]], path, domain)
	'''
	texts = [open('Data/testSample/story2', 'r').readlines()]
	#texts = [open('story1', 'r').readlines(), open('story1', 'r').readlines()]	
	#print predict_emotion_textList(texts, path = path, domain = domain)

	for emotion in 'joy, sad, disgust, anger, surprise, fear'.split(','):
		#emotion = 'sad'
		print predict_emotion_textList(texts, emotion)
		#print predict_emotion_docsList(textList)
 
