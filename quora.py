import csv
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import pickle
from tqdm import tqdm
import os.path

def create_csv():
	file = open("train.csv","r")

	reader = csv.reader(file)
	neg_c = [row for row in reader if row[5] != 'is_duplicate' and 0 == int(row[5])]

	file.seek(0)
	pos_c = [row for row in reader if row[5] != 'is_duplicate' and 1 == int(row[5])]
	count_pos = len(pos_c)
	count_neg = len(neg_c)

	print " len of pos ex = %s " % len(pos_c)
	print " len of neg ex = %s " % len(neg_c)

	with open('pos_train.csv', 'w') as f:
	    writer = csv.writer(f)
	    writer.writerows(pos_c)

	with open('neg_train.csv', 'w') as f:
	    writer = csv.writer(f)
	    writer.writerows(neg_c)    

def get_feature_from_question(q1 , q2 , feature_size, cosineDist):

	X = np.zeros((feature_size))
	X[0] = len(q1) #len of q1
	X[1] = len(q2) #len of q2

	X[2] = abs(X[0] - X[1]) #difference of len

	X[3] = len(set(q1.replace(' ', '')) )   # unique chars in q1
	X[4] = len(set(q2.replace(' ', '')) )   # unique chars in q2

	X[5] = len( q1.split() )   # no of words in q1
	X[6] = len( q2.split() )   # no of words in q2

	X[7] = len(   set( q1.lower().split() ).intersection( set( q2.lower().split()  ) ) )  # common words
	X[8] = cosineDist
	X[feature_size - 1] = 1
	return X

def getFeature11(question): # "what"
	X = np.zeros(len(question))
	print " size of temp X feature is"
	print X.shape
	for index in range(len(question)):
		if question[index].startswith("What"):
			X[index] = 1
		else :
			X[index] = 0
	return X
def create_features():
	print " Creating Features Start !"
	use_partial = "NO"
	DATA_SIZE = 0
	question1 = []
	question2 = []
	is_duplicate = []
	feature_size = 9
	tokenize = lambda doc: doc.lower().split(" ")
	print " FEature size = %s" % feature_size
	
	print "Reading Negative Train File"
	if use_partial == "YES":
		file = open("partial_neg_train.csv","r")
	else:
		file = open("neg_train.csv","r")

	reader = csv.reader(file)

	for row in reader:
		str1 = unicode(str(row[3]), errors='ignore')
		question1.append(str1)
		str2 = unicode(str(row[4]), errors='ignore')
		question2.append(str2)
		is_duplicate.append(row[5])
		DATA_SIZE += 1
	print "Reading Positive Train File"
	if use_partial == "YES":
		file1 = open("partial_pos_train.csv","r")
	else:
		file1 = open("pos_train.csv","r")
	reader1 = csv.reader(file1)

	for row in reader1:
		str1 = unicode(str(row[3]), errors='ignore')
		question1.append(str1)
		str2 = unicode(str(row[4]), errors='ignore')
		question2.append(str2)
		is_duplicate.append(row[5])
		DATA_SIZE += 1

	#for index in range( DATA_SIZE):
		# print "q1: %s\nq2: %s \n%s \n\n" % (question1[index] , question2[index] , is_duplicate[index] )

		# starting vectorizer
	if not os.path.exists("X_cosine_scipy.pickle") or use_partial == "YES":
		print "Picle file does not exist !"
		print "Running Tf-Idf Vectorizer"

		tfidvec = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
		tflist = list(question1)
		tflist.extend(question2)
		tfidvec.fit(tflist)

		tfid_q1 = tfidvec.transform(question1)
		
		tfid_q2 = tfidvec.transform(question2)
		
		print " question 1 vector size = %d X %d " % (tfid_q1.shape[0] , tfid_q1.shape[1])
		print " question 2 vector size = %d X %d" % (tfid_q2.shape[0], tfid_q2.shape[1])
		print " DATA SIZE  = %d " % DATA_SIZE
		feature_size += 1
		print " new feature size is %d " % feature_size 

		# print " tf id = %s " % tfid_q1[0]
		# print tfid_q1[0,:]*tfid_q2[0,:]
		eX=scipy.spatial.distance.cosine(tfid_q1[0].toarray(),tfid_q2[0].toarray())
		# print eX
		
		# q1_tfarray = tfid_q1.toarray()
		# q2_tfarray = tfid_q2.toarray()

		# print " question 1 array size = %d X %d " % (q1_tfarray.shape[0] , q1_tfarray.shape[1])
		# print " question 2 array size = %d X %d" % (q2_tfarray.shape[0], q2_tfarray.shape[1])

		X = np.zeros((DATA_SIZE , feature_size))

		for index in tqdm(range(DATA_SIZE)):
			ques1 = question1[index]
			ques2 = question2[index]
			eX = scipy.spatial.distance.cosine(tfid_q1[index].toarray(),tfid_q2[index].toarray())
			#print " ex = %s and y = %d " % (eX, Y[index])
			X[index] = get_feature_from_question(ques1,ques2, feature_size, eX)
			# if index % 100 == 0:
			# 	print " index  = %s " % index
		
		# pickleing X
		with open('X_piclePartial.pickle', 'wb') as handle:
			pickle.dump(X, handle , protocol = pickle.HIGHEST_PROTOCOL)
	else:
		print "Pickle file exist."
		with open('X_cosine_scipy.pickle' , 'rb') as handle:
			X = pickle.load(handle)
		print " loaded pickle file size is"
		feature_size = X.shape[1]
		print X.shape

	Y = np.zeros(DATA_SIZE)
	#W = np.zeros(feature_size)
	
	for index in tqdm(range(DATA_SIZE)):
		if is_duplicate[index] == '0':
			Y[index] = 0
		else :
			Y[index] = 1

	print " X.shape = "
	print (X.shape)
	print "feature size = %s " % feature_size

	# adding 11th feature what
	if not os.path.exists("trainFeature11.pickle"):
		print " adding feature 11!"
		X11Q1 = getFeature11(question1)
		X = np.insert( X , feature_size , X11Q1 , axis = 1)
		X11Q2 = getFeature11(question2)
		X = np.insert( X , feature_size + 1 , X11Q1 , axis = 1)
		feature_size += 2
		write = np.zeros((len(question1) , 0))
		write = np.insert(write , 0 , X11Q1 , axis = 1)
		write = np.insert(write , 1 , X11Q2 , axis = 1)
		
		with open('trainFeature11.pickle', 'wb') as handle:
			pickle.dump(write, handle , protocol = pickle.HIGHEST_PROTOCOL)

	else:
		print " feature 11 file exist "
		with open('trainFeature11.pickle', 'rb') as handle:
			f11 = pickle.load(handle)
		print f11.shape
		X = np.insert( X , feature_size , values = 0 , axis = 1)
		X = np.insert( X , feature_size + 1 , values = 0 , axis = 1)
		for i in range(len(X)):
			X[i][feature_size] = f11[i][0] 
			X[i][feature_size+1] = f11[i][1]
		feature_size += 2


	print X.shape
	print "feature size = %s " % feature_size
	# print tfid_q1.shape
	# print tfid_q2.shape
	
	# from sklearn.metrics.pairwise import cosine_similarity
	# cs = cosine_similarity(tfid_q1,tfid_q2)
	# for x,y in zip(list(tfid_q1[0].toarray()),list(tfid_q2[0].toarray())):
	# 	print "%s\t%s" % (x,y)
	# exit(9)
	# diag = [ row[-i-1] for i,row in enumerate(cs) ]
	# print diag
	
	# exit(14)
	# print np.sum(tfid_q1[0].toarray())
	# print tfid_q2[0].toarray()
	# dsp = tfid_q1.multiply(tfid_q2)
	# # print dsp[0].toarray()
	# mod_tfid_q1 = tfid_q1.multiply(tfid_q1)
	# # .sum(axis=1)
	# print mod_tfid_q1
	# print "Haha %s" %np.sum(mod_tfid_q1[0].toarray())
	# mod_tfid_q2 = tfid_q2.multiply(tfid_q2)
	# .sum(axis=1)
	# print mod_tfid_q2
	# print dsp.shape
	# exit(12)

	#randon permutation
	# print "Y = %s" %(Y)
	print "permutating data features"
	np.random.seed(1)
	permutation = np.random.permutation(DATA_SIZE)
	X = X[permutation,]
	Y = Y[permutation,]
	lr = LogisticRegression(verbose=2)
	print "Training model !"
	lr.fit(X,Y)
	print lr.score(X,Y)

	print "Opening test.csv !"
	file3 = open("test.csv","r")
	# file3 = open("partial_neg_train.csv","r")
	reader3 = csv.reader(file3)

	test_q1 = []
	test_q2 = []
	test_qid = []
	DATA_SIZE = 0

	for row in reader3:
		if DATA_SIZE == 0:
			DATA_SIZE = 1
		else :
			test_qid.append(row[0])
			test_q1.append(row[1])
			test_q2.append(row[2])
			DATA_SIZE += 1
	DATA_SIZE -= 1
	feature_size = 10
	if not os.path.exists("testFeatures.pickle"):
		tfidvec1 = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
		tflist1 = list(test_q1)
		tflist1.extend(test_q2)
		tfidvec1.fit(tflist1)

		tf_q1 = tfidvec1.transform(test_q1)
		
		tf_q2 = tfidvec1.transform(test_q2)


		test_X = np.zeros((DATA_SIZE , feature_size))

		for index in tqdm(range(DATA_SIZE)):
			ques1 = test_q1[index]
			ques2 = test_q2[index]
			eX = scipy.spatial.distance.cosine(tf_q1[index].toarray(),tf_q2[index].toarray())
			test_X[index] = get_feature_from_question(ques1,ques2, feature_size,eX)
		
		with open('testFeatures.pickle', 'wb') as handle:
			pickle.dump(test_X, handle , protocol = pickle.HIGHEST_PROTOCOL)
	else:
		print "testFeature Pickle file exist."
		with open('testFeatures.pickle' , 'rb') as handle:
			test_X = pickle.load(handle)
		print " loaded pickle file size is"
		feature_size = test_X.shape[1]
		print test_X.shape

	print " adding 11 feature for test data "
	print "feature size = %d " % feature_size
	# adding 11th feature what
	if not os.path.exists("testFeature11.pickle"):
		print " adding feature 11!"
		X11Q1 = getFeature11(test_q1)
		test_X = np.insert( test_X , feature_size , X11Q1 , axis = 1)
		X11Q2 = getFeature11(test_q2)
		test_X = np.insert( test_X , feature_size + 1 , X11Q1 , axis = 1)
		feature_size += 2
		write = np.zeros((len(test_q1) , 0))
		write = np.insert(write , 0 , X11Q1 , axis = 1)
		write = np.insert(write , 1 , X11Q2 , axis = 1)
		
		with open('testFeature11.pickle', 'wb') as handle:
			pickle.dump(write, handle , protocol = pickle.HIGHEST_PROTOCOL)

	else:
		print " feature 11 file exist "
		with open('testFeature11.pickle', 'rb') as handle:
			f11 = pickle.load(handle)
		print f11.shape
		test_X = np.insert( test_X , feature_size , values = 0 , axis = 1)
		test_X = np.insert( test_X , feature_size + 1 , values = 0 , axis = 1)
		for i in range(len(test_X)):
			test_X[i][feature_size] = f11[i][0] 
			test_X[i][feature_size+1] = f11[i][1]
		feature_size += 2




	print "\n *** * predicting on testX  * * * * \n"

	y_test = lr.predict(test_X)
	# y_test = int(y_test)
	print " y ans = %s and shape = %s " % (y_test,y_test.shape)

	with open('results1.csv',"wb") as outob:
		outob.write("test_id," +"is_duplicate\n")
		for i in range(DATA_SIZE):
			outob.write(str(test_qid[i])+","+str(int(y_test[i]))+"\n" ) 
	
	# print "X = %s" %(X)
	#print "Accuracy (Logistic Loss):\t%s" % crossValidation(X, Y, SgdLogistic, maxIter=100, lmda=.3, learningRate=0.0001, sample=range(200))


def main():
	#create_csv()
	create_features()


if __name__ == "__main__":
	main()