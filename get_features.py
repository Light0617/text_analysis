#!usr/bin/python
import numpy as np
import ast, re, csv
import json
import os
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from pattern.en import lemma
from nltk.corpus import wordnet as wn
from nltk.tag import StanfordPOSTagger
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk import word_tokenize
from nltk.tree import *
from get_story import *

## for self.lexicon file
LEXICON = 'lexicon/Semeval_2007/lexicon.json'
class word_processor():
	## store word with sad emotion and {word: np.array with its score}
	## since the original vector contains positive, negtive, surprise,trust, so we remove them and do normalization
	## it means we normalize 6 basic emotions, if one word only express sadness emotion, we weigh more than another word express multiple emotions 

	def __init__(self):
		self.lexicon = {}	

		## for POS tagger
		path = os.path.abspath('../') + '/'
		JAR = path + 'stanford-postagger/stanford-postagger.jar'
		MODEL = path + 'stanford-postagger/models/english-left3words-distsim.tagger'
		self.pos_tagger = StanfordPOSTagger(MODEL, JAR, encoding='utf8')

		#self.LEXICON_FILE1 = os.getcwd() + '/' + 'lexicon/sad_lexicon'
		#self.LEXICON_FILE2 = os.getcwd() + '/' + 'lexicon/pos_lexicon'
		#self.LEXICON_FILE3 = os.getcwd() + '/' + 'lexicon/neg_lexicon'
		arguments = {}
		arguments['--jar']	= path + 'stanford-parser/stanford-parser.jar'
		arguments['--modeljar']	= path + 'stanford-parser/stanford-parser-3.8.0-models.jar'
		arguments['--model'] = 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
		self.parser = StanfordParser(model_path=arguments['--model'], path_to_models_jar=arguments['--modeljar'], path_to_jar=arguments['--jar'])
		self.dependency_parser = StanfordDependencyParser(model_path=arguments['--model'], path_to_models_jar=arguments['--modeljar'], path_to_jar=arguments['--jar'])

		self.tokenizer = WordPunctTokenizer()	
		self.lexicon = self.build_dictionary(LEXICON) 
		print len(self.lexicon)

	def build_dictionary(self, infile):
		with open(infile, 'r') as json_file:
			return json.load(json_file)

	def check_emotion_word(self, w, emotion):
		if w not in self.lexicon or emotion not in self.lexicon[w]\
				 or self.lexicon[w][emotion] == 0:
			return 0
		else:
			return 1

	def get_emotion_score_word(self, w, emotion):
		if not self.check_emotion_word(w, emotion):
			return 0
		return self.lexicon[w][emotion]

	def normalize(self, sentence):
		if not isinstance(sentence, unicode):
			sentence = unicode(sentence, errors='ignore')
		return ' '.join([ str(lemma(x).lower()) for x in self.tokenizer.tokenize(sentence) \
						if isinstance(x, unicode)
						and len(wn.synsets(x)) > 0])

	#emotion feature
	def get_emotion_word_feature(self, sentence, emotion):
		sentence = ' '.join([x for x in  sentence.split(' ') if x not in stopwords.words('english') and len(x) > 1])
		items = word_tokenize(sentence)
		emotion, neg, pos, emotion_score = 0, 0, 0, -10
		#print items
		for word in items:
			if word in self.lexicon:
				emotion = max(self.check_emotion_word(word, emotion), emotion)
				neg = max(self.check_emotion_word(word, 'neg'), neg)
				pos = max(self.check_emotion_word(word, 'pos'), pos)
				emotion_score = max(self.get_emotion_score_word(word, emotion),\
									 emotion_score)
		emotion_score = emotion_score if emotion_score != -10 else 0
		return [emotion, pos, neg, emotion_score]
	
	##POS feature
	def get_pos_feature(self, sentence, emotion):
		items = self.pos_tagger.tag(word_tokenize(sentence))
		flag_adj, flag_adv = 0, 0
		neighbor_pos = []
		for i in range(len(items)):
			word, pos = items[i]
			if self.check_emotion_word(word, emotion):
				flag_adj = 1 if 'JJ' in pos else 0
				flag_adv = 1 if 'RB' in pos else 0
				neighbors = [items[k] if k >= 0 and k < len(items) else [] for k in [i-2, i-1, i+1, i+2] ] 

				neighbor_pos = [x[1] if len(x) > 0 else "" for x in neighbors]
		array = self.pos_to_list(neighbor_pos)
		return [flag_adj, flag_adv] + array

	def negation_detect(self, sentence):
		NEGATION_ADVERBS = ["no", "without", "nil","not", "n't", "never", "none", "neith", "nor", "non"]
		NEGATION_VERBS = ["deny", "reject", "refuse", "subside", "retract", "non"]
		for key_word in NEGATION_ADVERBS + NEGATION_VERBS:	
			if key_word in sentence:
				return [1]
		return [0]

	def pos_mapping(self, pos):
		dict = {}
		s = 'CC,CD,DT,EX,FW,IN,JJ,JJR,JJS,LS,MD,NN,NNS,NNP,NNPS,PDT,POS,PRP,PRP$,RB,RBR,RBS,RP,SYM,TO,UH,VB,VBD,VBG,VBN,VBP,VBZ,WDT,WP,WP$,WRB'
		list = s.split(',')
		for i in range(len(list)):
			dict[list[i]] = i
		return dict.get(pos, -1)

	def pos_to_list(self, neighbor_pos):
		list = [self.pos_mapping(pos) for pos in neighbor_pos]
		array, k = [0] * (36 * 4), 0
		for num in list:
			if num >= 0:
				array[k + num] = 1
			k += 36
		return array

	##sentence features : number of word in the sentence, ignore signle word
	def get_number_word_sentence(self, sentence):
		return len([x for x in sentence.strip().split() if len(x) > 1])
	
	def convert2_adj(self, w):
		possible_adj = []
		for syn in wn.synsets(w):
			for lemma in syn.lemmas(): # all possible lemmas.
				lemma.derivationally_related_forms()
				pertainyms = lemma.pertainyms()
				for ps in pertainyms: # all possible pertainyms.
				    possible_adj += str(ps.name()),
		return possible_adj[0] if len(possible_adj) > 0 else w

	def get_dependency_features(self, sentence, emotion):
		features = [0]  * 13
		try:
			dep = list(self.dependency_parser.raw_parse(sentence).next().triples())
		except:
			dep = []
		for modified, rel, modifying in dep:
			pos1, pos2 = modifying[1], modified[1]
			modifying, modified = modifying[0], modified[0]
			#print rel, modifying, modified, pos1, pos2
			if pos1 == 'RB' or pos1 == 'RBR' or pos1 == 'RBS':
				modifying = self.convert2_adj(modifying)
				
			if pos2 == 'RB' or pos2 == 'RBR' or pos2 == 'RBS':
				modified = self.convert2_adj(modified)	
			#print rel, modifying, modified, pos1, pos2

			if rel == 'neg' and self.check_emotion_word(modified, emotion):
				#print 1, modifying, modified
				features[0] = 1
	
			if rel == 'amod' and self.check_emotion_word(modified, emotion):
				#print 21, modifying, modified	
				features[1] = 1
			 
			if rel == 'amod' and self.check_emotion_word(modifying, emotion):
				#print 22, modifying, modified
				features[2] = 1

			if rel == 'advmod' and self.check_emotion_word(modified, emotion):
				#print 31, modifying, modified	
				features[3] = 1

			if rel == 'advmod' and self.check_emotion_word(modifying, emotion):
				#print 32, modifying, modified	
				features[4] = 1

			#emotion pos
			if rel == 'amod'  and self.check_emotion_word(modifying, emotion)\
						 and self.check_emotion_word(modified, 'pos'):
				#print 41, modifying, modified
				features[5] = 1

			#pos emotion						
			if rel == 'amod'  and self.check_emotion_word(modifying, 'pos') \
						and self.check_emotion_word(modified, emotion):
				#print 42, modifying, modified
				features[6] = 1

			#emotion neg
			if rel == 'amod' and self.check_emotion_word(modifying, emotion) \
						and self.check_emotion_word(modified, 'neg'):
				#print 51, modifying, modified
				features[7] = 1

			#neg emotion
			if rel == 'amod' and self.check_emotion_word(modifying, 'neg') \
						and self.check_emotion_word(modified, emotion):
				#print 52, modifying, modified
				features[8] = 1
			
			#emotion pos
			if rel == 'advmod'  and self.check_emotion_word(modifying, emotion) \
						and self.check_emotion_word(modified, 'pos'):
				#print 61, modifying, modified
				features[9] = 1
			
			#pos emotion
			if rel == 'advmod'  and self.check_emotion_word(modifying, 'pos') \
						and self.check_emotion_word(modified, emotion):
				#print 62, modifying, modified
				features[10] = 1

			#emotion neg
			if rel == 'advmod' and self.check_emotion_word(modifying, emotion) \
						and self.check_emotion_word(modified, 'neg'):
				#print 71, modifying, modified
				features[11] = 1

			#neg emotion
			if rel == 'advmod' and self.check_emotion_word(modifying, 'neg') \
						and self.check_sad(modified, emotion):
				#print 72, modifying, modified
				features[12] = 1

		return features

	def test_lexicon(self):
		for key in self.lexicon:
			print key, len(key), type(key)
			print self.lexicon[key], type(self.lexicon[key])
	
	def get_features1(self, sentence, emotion):
		sentence = self.normalize(sentence)
		#print sentence
		vec =  [1] + self.negation_detect(sentence)  + [self.get_number_word_sentence(sentence)] \
				+ self.get_emotion_word_feature(sentence, emotion) + self.get_pos_feature(sentence, emotion) \
				+ self.get_dependency_features(sentence, emotion)
		#print len(vec)
		return vec
	def get_feature_baseline(self, sentence):
		return [1]

if __name__ == '__main__':	
	wp = word_processor() 
	'''
	s = "Where is the fly?"
	s = "I am sorry"
#	for s in open('train1', 'r').readlines():
#		s, _ = s.split(',')
#		print s
		#print wp.get_features1(s)
		#print wp.get_feature_baseline(s)
	wp.pos_mapping('TO')	
	url = 'https://www.gofundme.com/justin-reed-medical-fun'
	text = get_story(url)
	#for line in text.split('[:!,?\.]+'):
	for line in re.split('[,:!?.]+', text):
		if len(line.split()) >= 2:
			print line
			#print wp.get_features1(line)
	'''

	
	#For testing dependency
	#strs  = ["The dog terribly ran.", "The scold dog ran.",'I have a scold scholar.', 'I have a scold hardness.']
	#strs  = ["The dog terribly ran."]
	#strs  = [""]
	strs = [''.join(x.split(',')[:-1]) for x in open('train1', 'r').readlines()]
	#strs  = ["I hate a dog."]
	for s in strs:
		print '================'
		print s
		#print wp.get_dependency_features(s)
		v = wp.get_features1(s, 'sad')
		print len(v), v
	

