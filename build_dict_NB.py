import json
from NaiveBayes import *
def build_dict_NB(test_file, classifier, emotion, data):
    with open(test_file, 'r') as json_file:
        fromData = json.load(json_file)
        for setence in fromData:
            #print setence
            tokens = extract_tokens(setence)
            for token in tokens:
                bag = bag_of_words(token)
                prob_pos = classifier.prob_classify(bag).prob('pos')
                if token not in data: data[token] = {}
                data[token][emotion] = prob_pos
    return data

def get_lexicon_external_json_to_dict(json_file, data):
    with open(json_file, 'r') as infile:
        dict1 = json.load(infile)
        for word in dict1:
            if word not in data: data[word] = {}
            for sent in dict1[word]:
                data[word][sent] = dict1[word][sent]

def build_lexicon_from_NB_external(test_file, outfile):
	data = {}
	externalFile = 'external_data/lexicon.json'
	emotions = 'joy,sad,disgust,anger,surprise,fear'.split(',')
	for emotion in emotions:
		print emotion
		pickled_classifier = 'pickled/Semeval_2007/NaiveBayes/%s/classifier_bigram1.pickle'\
							% (emotion)
		classifier = pickle.load(open(pickled_classifier, 'r')) 

		#build two different dictionaries, one is with Naive Bayes,
		# the other one is with external dictionary
		build_dict_NB(test_file, classifier, emotion, data)
		print 'After add NB classifier, the size of data is', len(data)

	get_lexicon_external_json_to_dict(externalFile, data)
	print 'After add external lexicon, the size of data is', len(data)

	with open(outfile, 'w') as f:
		json.dump(data, f, indent = 2)

if __name__ == '__main__':
	domain = 'Semeval_2007'
	filePath = 'Data/%s/allData.json' % (domain)
	outfile = 'lexicon/%s/lexicon.json'	% (domain)	
	build_lexicon_from_NB_external(filePath, outfile)

