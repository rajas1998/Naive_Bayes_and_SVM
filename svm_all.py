import numpy as np
from cvxopt import matrix, solvers
import math
import time
from svmutil import *
from scipy.spatial.distance import pdist, squareform
import scipy
from sklearn.model_selection import train_test_split
import sys

def read_csv(filename):
	f = open(filename,"r")
	x = f.readlines()
	f.close()
	x = [list(map(float, i.split(","))) for i in x]
	y = [int(i[-1]) for i in x]
	x = [np.array(i[:-1])/255 for i in x]
	y = np.array(y)
	x = np.array(x)
	return (x,y)

def dual_class(x,y,a,b):
	x_reduced = []
	y_reduced = []
	m = len(y)
	for i in range(m):
		if (y[i] == a):
			x_reduced.append(x[i])
			y_reduced.append(-1)
		elif (y[i] == b):
			x_reduced.append(x[i])
			y_reduced.append(1)
	return (np.array(x_reduced), np.array(y_reduced))

def fit_linear(x_in, y_in, c): 
	m_in = len(y_in)
	print (x_in.shape)
	K = y_in[:,None] * x_in
	K = np.dot(K, K.T)
	P = matrix(K)
	q = matrix(-np.ones((m_in, 1)))
	G = matrix(np.concatenate((-np.eye(m_in),np.eye(m_in))))
	h = matrix(np.concatenate((np.zeros(m_in),c*np.ones(m_in))))
	A = (y_in.reshape(1, -1))
	A = A.astype(float)
	A = matrix(A)
	b = matrix(np.zeros(1))
	solvers.options['show_progress'] = True
	sol = solvers.qp(P, q, G, h, A, b)
	alphas = np.array(sol['x'])
	w = np.sum(alphas * y_in[:, None] * x_in, axis = 0)
	bias = 0
	for i in range(len(alphas)):
		if (alphas[i] > 1e-4 and alphas[i] < c - 1e-4):
			bias = y_in[i] - np.dot(x_in[i], w)
			break
	return (alphas,w,bias)

def gaussian_multiply(x_in,y_in,alphas,non_zeros,gamma,x):
	ret = 0.0
	# print (x_in.shape)
	m = x_in.shape[0]
	# print (x.reshape(1,len(x)).shape)
	tmp = np.sum((x_in - x)**2, axis = 1)
	tmp = scipy.exp((-1) * gamma * tmp).reshape(m,1) * alphas.reshape(m,1) * y_in.reshape(m,1)
	# print ("Temp ",tmp.shape)
	# sys.exit() \
	# for i in non_zeros:
	# 	# ret += alphas[i]*y_in[i]*math.exp((-1)*gamma*(np.linalg.norm(x_in[i]-x)**2))
	# 	ret += tmp[i]
	# print (ret)
	return np.sum(tmp)
def fit_gaussian(x_in, y_in, c, gamma): 
	m_in = len(y_in)
	# print (x_in.shape)
	start = time.time()
	K = squareform(pdist(x_in, 'euclidean'))
	K = scipy.exp((-1) * (K ** 2) * gamma)
	yv = y_in.reshape(-1, 1)
	ydash = np.matmul(yv,np.transpose(yv))
	K = K * ydash
	# print (ydash.shape)
	# print (K.shape)
	# print (K)
	# K = np.zeros((m_in,m_in))
	# for i in range(m_in):
	# 	# print i
	# 	for j in range(m_in):
	# 		K[i][j] = math.exp((-1)*gamma*(np.linalg.norm(x_in[i]-x_in[j])**2))*y_in[i]*y_in[j]
	# print (K)
	# print (K.shape)
	end = time.time()
	print "The time spent here is",end-start
	P = matrix(K)
	q = matrix(-np.ones((m_in, 1)))
	G = matrix(np.concatenate((-np.eye(m_in),np.eye(m_in))))
	h = matrix(np.concatenate((np.zeros(m_in),c*np.ones(m_in))))
	A = (y_in.reshape(1, -1))
	A = A.astype(float)
	A = matrix(A)
	b = matrix(np.zeros(1))
	solvers.options['show_progress'] = False
	sol = solvers.qp(P, q, G, h, A, b)
	alphas = np.array(sol['x'])
	non_zeros = []
	for i in range(len(alphas)):
		if (alphas[i] > 1e-4):
			non_zeros.append(i)
	for i in range(len(alphas)):
		if (alphas[i] > 1e-4 and alphas[i] < c - 1e-4):
			bias = y_in[i] - gaussian_multiply(x_in,y_in,alphas,non_zeros,gamma,x_in[i])
			break
	return (alphas,non_zeros,bias)

def test_accuracy_linear(x,y,w,b):
	pred_y = []
	for i in range(len(x)):
		if (np.dot(x[i], w) + b >= 0):
			pred_y.append(1)
		else :
			pred_y.append(-1)
	cnt = 0
	for i in range(len(y)):
		if (y[i] == pred_y[i]):
			cnt += 1
	return (float(cnt)/len(y))

def test_accuracy_gaussian(x,y,b,alphas,non_zeros,gamma,x_in,y_in):
	pred_y = []
	for i in range(len(x_in)):
		if (gaussian_multiply(x,y,alphas,non_zeros,gamma,x_in[i]) + b >= 0):
			pred_y.append(1)
		else :
			pred_y.append(-1)
	cnt = 0
	for i in range(len(y_in)):
		if (y_in[i] == pred_y[i]):
			cnt += 1
	return (float(cnt)/len(y_in))

def prediction_gaussian(x,y,b,alphas,non_zeros,gamma,x_in,y_in):
	pred_y = []
	for i in range(len(x_in)):
		if (gaussian_multiply(x,y,alphas,non_zeros,gamma,x_in[i]) + b >= 0):
			pred_y.append(1)
		else :
			pred_y.append(-1)
		# print (i)
	return pred_y

x_train, y_train = read_csv("mnist/train.csv")
x_test, y_test = read_csv("mnist/test.csv")

# First part
# x_a, y_a = dual_class(x_train, y_train, 4, 5)
# start = time.time()
# alpha, w, b = fit_linear(x_a, y_a, 1.0)
# print ("W and b are ",w,b)
# end = time.time()
# print "Time taken for linear is ", (end - start)
# x_test_a, y_test_a = dual_class(x_test, y_test, 4, 5)
# num_sup = 0
# print "The support vectors are:"
# for i in range(len(y_a)):
# 	if ((alpha[i] > 1e-4)):
# 		# print i,
# 		num_sup += 1
# print
# print "The number of support vectors are ",num_sup
# print "The training set accuracy is", test_accuracy_linear(x_a,y_a,w,b)
# print "The test set accuracy is", test_accuracy_linear(x_test_a, y_test_a, w, b)

#Second Part 
# gamma = 0.05
# x_a, y_a = dual_class(x_train, y_train, 5, 6)
# start = time.time()
# alphas, non_zeros, b = fit_gaussian(x_a, y_a, 1.0, gamma)
# end = time.time()
# print "Time taken for gaussian is ", (end - start)
# x_test_a, y_test_a = dual_class(x_test, y_test, 5, 6)
# num_sup = 0
# print "The support vectors are:"
# for i in range(len(y_a)):
# 	if ((alphas[i] > 1e-4)):
# 		# print i,
# 		num_sup += 1
# print
# print "The number of support vectors are ",num_sup
# start = time.time()
# print "The training set accuracy is", test_accuracy_gaussian(x_a,y_a,b,alphas,non_zeros,gamma, x_a, y_a)
# print "The test set accuracy is", test_accuracy_gaussian(x_a, y_a, b, alphas, non_zeros, gamma, x_test_a, y_test_a)
# end = time.time()
# print ("The time taken is ", end - start)

#Third part
# x_a, y_a = dual_class(x_train, y_train, 5, 6)
# x_test_a, y_test_a = dual_class(x_test, y_test, 5, 6)
# prob  = svm_problem(y_a, x_a)
# param = svm_parameter('-t 0 -c 1')
# m = svm_train(prob, param)
# print "Training set: "
# p_label, p_acc, p_val = svm_predict(y_a, x_a, m)
# # print (p_label,p_acc,p_val)
# print "Test set: "
# p_label, p_acc, p_val = svm_predict(y_test_a, x_test_a, m)
# # print (p_label,p_acc,p_val)

#Fourth part
# x_a, y_a = dual_class(x_train, y_train, 5, 6)
# x_test_a, y_test_a = dual_class(x_test, y_test, 5, 6)
# prob  = svm_problem(y_a, x_a)
# param = svm_parameter('-t 2 -c 1 -g 0.05')
# m = svm_train(prob, param)
# print "Training set: "
# p_label, p_acc, p_val = svm_predict(y_a, x_a, m)
# # print (p_label,p_acc,p_val)
# print "Test set: "
# p_label, p_acc, p_val = svm_predict(y_test_a, x_test_a, m)

#Q2a
labels = []
labels2 = []
for i in range(10):
	for j in range(i+1,10):
		print i, ",",j,":- "
		gamma = 0.05
		x_a, y_a = dual_class(x_train, y_train, i, j)
		alphas, non_zeros, b = fit_gaussian(x_a, y_a, 1.0, gamma)
		start = time.time()
		labels.append((prediction_gaussian(x_a,y_a,b,alphas,non_zeros,gamma, x_train, y_train),i,j))
		end = time.time()
		print (end - start)
# for m in models:
# 	p_label, p_acc, p_val = svm_predict(y_train,x_train, m[0])
# 	labels.append((p_label,m[1],m[2]))
print ("Reached here")
pred_y = []
scores = [0 for j in range(10)]
for x in labels:
	if (x[0][i] == 1):
		scores[x[2]] += 1
	else:
		scores[x[1]] += 1
for i in range(len(x_train)):
	w = [0 for j in range(10)]
	for x in labels:
		if (x[0][i] == 1):
			w[x[2]] += 1
		else:
			w[x[1]] += 1
	to_find = max(w)
	matnow = -1
	ind = -1
	for i in range(len(w)):
		if ((w[i] == to_find) and (scores[i] > matnow)):
			matnow = scores[i]
			ind = i
	# pred_y.append(w.index(max(w)))
	pred_y.append(ind)
cnt = 0
for i in range(len(x_train)):
	if (y_train[i] == pred_y[i]):
		cnt += 1
print "The accuracy over the training set is", float(cnt)/len(x_train)
# for m in models:
# 	p_label, p_acc, p_val = svm_predict(y_test,x_test, m[0])
# 	labels2.append((p_label,m[1],m[2]))
pred_y = []
scores = [0 for j in range(10)]
for x in labels2:
	if (x[0][i] == 1):
		scores[x[2]] += 1
	else:
		scores[x[1]] += 1
for i in range(len(x_test)):
	w = [0 for j in range(10)]
	for x in labels2:
		if (x[0][i] == 1):
			w[x[2]] += 1
		else:
			w[x[1]] += 1
	to_find = max(w)
	matnow = -1
	ind = -1
	for i in range(len(w)):
		if ((w[i] == to_find) and (scores[i] > matnow)):
			matnow = scores[i]
			ind = i
	# pred_y.append(w.index(max(w)))
	pred_y.append(ind)
cnt = 0
for i in range(len(x_test)):
	if (y_test[i] == pred_y[i]):
		cnt += 1
print "The accuracy over the test set is", float(cnt)/len(x_test)

#Q2b
# trainingSet_x, validationSet_x, trainingSet_y, validationSet_y = train_test_split(x_train, y_train, test_size=0.1)
# start = time.time()
# prob  = svm_problem(trainingSet_y, trainingSet_x)
# for c in [10**(-5),10**(-3),1,5,10]:
# 	print "Checking for c = "+str(c)
# 	param = svm_parameter('-t 2 -c '+str(c)+' -g 0.05')
# 	m = svm_train(prob, param)
# 	print "Validation set: "
# 	p_label, p_acc, p_val = svm_predict(validationSet_y, validationSet_x, m)
# 	print "Test set: "
# 	p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
# # print "Confusion Matrix:"
# end = time.time()
# print (end - start)
# print (confusion_matrix(y_test,p_label))

