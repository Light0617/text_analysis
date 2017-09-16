import os
import pickle
import re 
import sys
from get_features import *
from sklearn import linear_model, datasets
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
	with open(infile, 'r') as json_file:
		data = json.load(json_file)
	X = [wp.get_features1(s, emotion) for s in data]
	Y = [data[s][emotion] for s in data]
	return [X, Y]
	
def model1(trainX, trainY, c = 1e5):
    logistic = linear_model.LogisticRegression(C = c)
    trainX, trainY= np.array(trainX), np.array(trainY)
    print trainX.shape, trainY.shape 
    logistic.fit(trainX, trainY)
    return logistic

def evaluate(logistic, X, Y):
	return logistic.score(X, Y)

def test_line(wp, logistic, testSetence, emotion):
	vec = np.array(wp.get_features1(testSetence, emotion)).reshape(1, -1)
	return logistic.predict_proba(vec)[0,1]

def get_logistic(path, domain, model, emotion, pickled_logistic):

	if not os.path.exists(pickled_logistic):
		print 'Logistic does not exist'
		pickled_trainX = path + '/%s/%s/%s/trainX.pickle' % (domain, model, emotion)
		pickled_trainY = path + '/%s/%s/%s/trainY.pickle' % (domain, model, emotion)
		trainX = pickle.load(open(pickled_trainX, 'r'))
		trainY = pickle.load(open(pickled_trainY, 'r'))
		logistic = model1(trainX, trainY)	
		pickle.dump(logistic, open(pickled_logistic, 'w'))
	else:
		logistic = pickle.load(open(pickled_logistic, 'r'))
	return logistic

def get_wp(pickled_wp):
	if not os.path.exists(pickled_wp):
		print 'WP does not exist'
		wp = word_processor() 
		pickle.dump(wp, open(pickled_wp, 'w'))
	else:
		wp = pickle.load(open(pickled_wp, 'r'))
	return wp

def predict_emotion_text(wp, logistic, text, emotion):
	lines = [x for x in re.split('[!.?]+', text.strip('\n')) if len(x) > 0]
	scores = 0
	for line in lines:
		score = test_line(wp, logistic, line, emotion)
		print line
		print score
		scores += score
	return scores / float (len(lines))	

def predict_init(path, domain, model, emotion):
	path = os.getcwd() + '/' + path
	print path, domain
	pickled_logistic = path + '/%s/%s/%s/logistic.pickle' % (domain, model, emotion)
	pickled_wp = path + '/%s/%s/%s/wp.pickle' % (domain, model, emotion)
	print pickled_logistic
	print pickled_wp
	return [get_logistic(path, domain, model, emotion, pickled_logistic), get_wp(pickled_wp)]	

def predict_emotion_textList(textList, emotion, path = 'pickled', domain = 'Semeval_2007', model = 'model1'):
	logistic, wp = predict_init(path, domain, model, emotion)
	return [ predict_emotion_text(wp, logistic, text, emotion) for text in textList ]

def predict_emotion_fileList(fileList, emotion, path = 'pickled', domain = 'Semeval_2007', model = 'model1'):
	logistic, wp = predict_init(path, domain, model, emotion)
	res = []
	for target_file in fileList:
		with open(target_file, 'r') as text:
			res += predict_emotion_text(wp, logistic, text, emotion),
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
	emotion = 'sad'
	print predict_emotion_textList(texts, emotion)
	#print predict_emotion_docsList(textList)
 
