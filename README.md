# text_analysis
This is an application for recognition emotion from context.

## How to start
1. Go to https://nlp.stanford.edu/software/tagger.shtml, to download the software. (please install version: 3.8.0) 
2. install nltk
3. install numpy 
5. If you want to train the model, you could request the professor: Saima Aman(http://saimacs.github.io/index.html).


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

### Pickles
- trained/
Since getting features is very slow, we store the features in the pickles.

### Result 
- result
It records the performances among different data size and different models.

### OTHER
- wordnet-1.6/
- wordnet-domains-sentiwords/
- stanford-postagger-full-2017-06-09/
-external_data/ 
These all are libraries or external original lexicons.


## REFERENCE
Diman Ghazi , Diana Inkpen , Stan Szpakowicz, Prior and contextual emotion of words in sentential context, Computer Speech and Language, v.28 n.1, p.76-92, January, 2014 
