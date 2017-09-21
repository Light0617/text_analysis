# text_analysis
This is an application for recognition emotion from context.

## How to start to build a model
1. Go to https://nlp.stanford.edu/software/tagger.shtml, to download the software. (please install version: 3.8.0) 
2. install nltk
3. install numpy 
4. put the 'stanford-parser' and 'stanford-postagger' in the parent dictionary.
5. python get_features.py  for test
6. run modeling.ipynb 

## How to use, it will output the average sad emotion given a document.
1. story1 is the text you want to analyze
python lib.py story1

2. use API
from lib import *
print predict_emotion_textList(texts)


## INTRODUCTION
### Program 
- get_story.py
This is a python program to get the article from internet and this is what we want to test.

- NaiveBayes.py
There two tow functions for this program. One is for training a lexicon which given a word, it will output the probability of the word means 'sad'.
The other one is for predicting, and it is also my baseline model: unigram, bigram.

- get_features.py 
Precisely, it is getting the features from a sentence. given a sentence, it will get the setence features, word features, POS features.

- modeling.ipynb
I use spark to executing the modeling and prediction we use all features from get_features.py program.

### Data
- Data/
- I collect different size of data to do small data test. There are training data and testing data and there data are setence based.

- trainData and testData are from Semeval_2007 (emotions)
joy, sad, disgust, anger, surprise, fear

# Lexicon
1. lexicon.json
anger,anticipation,disgust,fear,joy,negative,positive,sadness,surprise,trust
 

### Pickles
- trained/
Since getting features is very slow, we store the features in the pickles.

### Result 
- result
It records the performances among different data size and different models.
for all data(training : 8982, testing : 3848), our result is 0.57 accuracy rate over Naive Bayes 0.5

- for Semval data (training : 987, testing : 241)
- joy
our result is 0.614 accuracy rate over Naive Bayes 0.5
- sad
our result is 0.705 accuracy rate over Naive Bayes 0.5
- disgust
our result is 0.639 accuracy rate over Naive Bayes 0.5
- anger
our result is 0.656 accuracy rate over Naive Bayes 0.5
- surprise
our result is 0.573 accuracy rate over Naive Bayes 0.5
- fear
our result is 0.618 accuracy rate over Naive Bayes 0.5





### OTHER
- wordnet-1.6/
- wordnet-domains-sentiwords/
- stanford-postagger-full-2017-06-09/
-external_data/ 
These all are libraries or external original lexicons.


## REFERENCE
### Emotion recogizing
Diman Ghazi , Diana Inkpen , Stan Szpakowicz, Prior and contextual emotion of words in sentential context, Computer Speech and Language, v.28 n.1, p.76-92, January, 2014 

### WordNet
George A. Miller (1995). WordNet: A Lexical Database for English. 
Communications of the ACM Vol. 38, No. 11: 39-41. 

Christiane Fellbaum (1998, ed.) WordNet: An Electronic Lexical Database. Cambridge, MA: MIT Press.

### Polarity
Wilson, T., Wiebe, J., Hoffmann, P.: Recognizing contextual polarity in phrase-level sentiment
analysis. In: Proceedings of the conference on empirical methods in natural language
processing (EMNLP 2005), pp. 347–354. Vancouver, B.C., Canada (2005)

Agarwal, Apoorv, et al. "Sentiment analysis of twitter data." Proceedings of the workshop on languages in social media. Association for Computational Linguistics, 2011.

Wilson, Theresa, Janyce Wiebe, and Paul Hoffmann. "Recognizing contextual polarity: An exploration of features for phrase-level sentiment analysis." Computational linguistics 35.3 (2009): 399-433.

Kiritchenko, Svetlana, Xiaodan Zhu, and Saif M. Mohammad. "Sentiment analysis of short informal texts." Journal of Artificial Intelligence Research 50 (2014): 723-762.

Poria, Soujanya, et al. "Sentic patterns: Dependency-based rules for concept-level sentiment analysis." Knowledge-Based Systems 69 (2014): 45-63.

Medhat, Walaa, Ahmed Hassan, and Hoda Korashy. "Sentiment analysis algorithms and applications: A survey." Ain Shams Engineering Journal 5.4 (2014): 1093-1113.




