import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
# from ass2_data import utils
import string
import time
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import math
import random
from sklearn.metrics import confusion_matrix
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import classification_report

p_stem = PorterStemmer()
stopwords = False
stemming = False

start = time.time()
training_data = pd.read_json("ass2_data/train.json", lines = True)
end = time.time()
print ("Finished loading the train data ", (end - start))

x = training_data['text']
y = training_data['stars']
training_data = None
m = len(y)

if (stopwords):
	analyzer = CountVectorizer(stop_words = "english").build_analyzer()
else:
	analyzer = CountVectorizer(ngram_range=(1,2)).build_analyzer()
vocab = {}
start = time.time()
x = [analyzer(i) for i in x]
def stem(inp):
	return [p_stem.stem(i) for i in inp]
if (stemming):
	x = [stem(i) for i in x]
end = time.time()
print ("Finished the tokenization ",end - start)

start = time.time()
count = 0
for i in x:
	for j in i:
		if j not in vocab:
			vocab[j] = count
			count += 1
end = time.time()
print (end - start)

param = [[1.0 for i in range(count)] for j in range(5)]

start = time.time()
for i in range(len(x)):
	for j in x[i]:
		param[y[i] - 1][vocab[j]] += 1
	if (i % 50000 == 0):
		print (i)
end = time.time()
print (end - start)

start = time.time()
sum_param = [sum(i) for i in param]
for i in range(len(param)):
	for j in range(len(param[i])):
		param[i][j] = math.log(param[i][j]/sum_param[i])
end = time.time()
print (end - start)

y_param = [0.0 for i in range(5)]
for i in y:
	y_param[i-1] += 1
y_param = [math.log(i/m) for i in y_param]
print (y_param)

def predict(inp):
	s = [y_param[i] for i in range(5)]
	for i in inp:
		if i in vocab:
			for j in range(5):
				s[j] += param[j][vocab[i]]
	return (s.index(max(s)) + 1)

def accuracy(inp,y,m):
	cnt = 0
	for i in range(m):
		if (inp[i] == y[i]):
			cnt += 1
	return ((cnt/m)*100)
start = time.time()	
pred_y = [predict(i) for i in x]
end = time.time()
print (end - start)

cnt = 0
for i in range(m):
	if (pred_y[i] == y[i]):
		cnt += 1
print ("Accuracy over training data = ", (cnt/m) * 100)

start = time.time()
test_data = pd.read_json("ass2_data/test.json", lines = True)
end = time.time()
print ("Finished loading the test data ", (end - start))

x = test_data['text']
y = test_data['stars']
test_data = None
m = len(y)
x = [analyzer(i) for i in x]
print ("Finished the tokenization")
pred_y = [predict(i) for i in x]
cnt = 0
for i in range(m):
	if (pred_y[i] == y[i]):
		cnt += 1
print ("Accuracy over test data = ", (cnt/m) * 100)
print ("Classification Report: ")
print (classification_report(y,pred_y))

pred_y = [random.randint(1,5) for i in range(m)]
print ("Accuracy with random guess = ", (accuracy(pred_y,y,m)))

elem = y_param.index(max(y_param)) + 1
pred_y = [elem for i in range(m)]
print ("Accuracy with majority guess = ", (accuracy(pred_y,y,m)))

pred_y = [predict(i) for i in x]
print ("Confusion Matrix")
print (confusion_matrix(y,pred_y))
