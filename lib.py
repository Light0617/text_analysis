import os
import pickle
import re 
import sys
from get_features import *
from sklearn import linear_model, datasets
def getData(infile):
    lines = [ line for line in open(infile, 'r').readlines()]
    text, Y = [line.split(',')[0] for line in lines], \
                        [int(line.split(',')[1]) for line in lines]
    print len(text), len(Y)
    wp = word_processor()
    data = [wp.get_features1(x) for x in text]
    return [data, Y]


def model1(trainX, trainY, c = 1e5):
    logistic = linear_model.LogisticRegression(C = c)
    trainX, trainY= np.array(trainX), np.array(trainY)
    print trainX.shape, trainY.shape 
    logistic.fit(trainX, trainY)
    return logistic

def evaluate(logistic, X, Y):
	return logistic.score(X, Y)

def test_line(wp, logistic, testSetence):
	#print testSetence
	vec = np.array(wp.get_features1(testSetence)).reshape(1, -1)
	return logistic.predict_proba(vec)[0,1]

def get_logistic(path, domain):
	pickled_logistic = '%slogistic%s.pickle' % (path, domain)

	if not os.path.exists(pickled_logistic):
		print 'Logistic does not exist'
		pickled_trainX = '%strainX%s.pickle' % (path, domain)
		pickled_trainY = '%strainY%s.pickle' % (path, domain)
		trainX = pickle.load(open(pickled_trainX, 'r'))
		trainY = pickle.load(open(pickled_trainY, 'r'))
		logistic = model1(trainX, trainY)	
		pickle.dump(logistic, open(pickled_logistic, 'w'))
	else:
		logistic = pickle.load(open(pickled_logistic, 'r'))
	return logistic

def get_wp(path, domain):
	pickled_wp = '%s_wp_%s.pickle' % (path, domain)
	if not os.path.exists(pickled_wp):
		print 'WP does not exist'
		wp = word_processor() 
		pickle.dump(wp, open(pickled_wp, 'w'))
	else:
		wp = pickle.load(open(pickled_wp, 'r'))
	return wp

def predict_emotion_text(wp, logistic, text):
	lines = ' '.join(line.strip('\n') for line in text)
	lines = [x for x in re.split('[!.?]+', lines) if len(x) > 0]
	scores = 0
	for line in lines:
		score = test_line(wp, logistic, line)
		print line, score
		scores += score
	return scores / float (len(lines))	

def predict_init(path = 'trained/', domain = 'random70_all'):
	return [get_logistic(path, domain), get_wp(path, domain)]	

def predict_emotion_textList(textList, path = 'trained/', domain = 'random70_all'):
	logistic, wp = predict_init(path, domain)
	return [ predict_emotion_text(wp, logistic, text) for text in textList ]

def predict_emotion_fileList(fileList, path = 'trained/', domain = 'random70_all'):
	logistic, wp = predict_init(path, domain)
	res = []
	for target_file in fileList:
		text = open(target_file, 'r').readlines()
		res += predict_emotion_text(wp, logistic, text),
	return res

if __name__ == '__main__':
	'''
	path, domain = 'trained/', 'random70_all'
	if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
		print predict_emotion_fileList(['story1'], path, domain)
	else:	
		print predict_emotion_fileList([sys.argv[1]], path, domain)
	'''
	texts = [open('story1', 'r').readlines(), open('story1', 'r').readlines()]	
	#print predict_emotion_textList(texts, path = path, domain = domain)
	print predict_emotion_textList(texts)
	#print predict_emotion_docsList(textList)
 
