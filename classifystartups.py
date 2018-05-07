import numpy as np
import sklearn
import csv 


categories_file = open("label_codes.csv", 'r')
categories_list = categories_file.readlines()[0].split('\r')

categories = {}
category_list = []
i = 0
for c in categories_list[1:]:
	# print(c.split(','))
	cat = c.split(',')[-1].lower().strip()
	# print cat
	categories[cat] = i;
	categories_list.append(cat)
	i = i+1
	pass
# print(categories)

dataset_filename = "tags_dataset.csv"
# data.readline()
# data = data.readline()
data_list = []
inputs = []
targets = []
i = 0

with open(dataset_filename, 'rU') as f:
    f.readline()
    for line in f:
        arr = line.rsplit(",",2)
        data_list.append(arr)
        inputs.append(arr[0])
        # print arr
        targets.append(arr[1].lower().strip())
        pass
	pass      

# print inputs

train_num = 2000

from sklearn.pipeline import Pipeline
# text_clf = Pipeline([('vect', CountVectorizer()),
# 					('tfidf', TfidfTransformer()),
# 					('clf', MultinomialNB())
# 					])


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(inputs[:train_num])

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)

# print count_vect.vocabulary_.get(u'algorithm')
from sklearn.naive_bayes import MultinomialNB
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, targets[:train_num])


docs_new = inputs[train_num:]
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
# print docs_new
# print predicted
for doc, category in zip(docs_new, predicted):
	print('%r => %s' % (doc, category))
	pass

docs_test = inputs[train_num:]
docs_test_target = targets[train_num:]

X_testinput_counts = count_vect.transform(docs_test)
X_testinput_tfidf = tfidf_transformer.transform(X_testinput_counts)

X_testtarget_counts = count_vect.transform(docs_test)
X_test_tfidf = tfidf_transformer.transform(X_testtarget_counts)

predicted = clf.predict(X_new_tfidf)
print (np.mean(predicted == X_test_tfidf))            
# extract data as list [Description, Name, Number]
# for row in data[1:]:
# 	arr = row[0].rsplit(",",2)
# 	print arr
# 	data_list.append(arr)
# 	inputs = arr[0]
# 	targets = arr[2]
# 	print i
# 	i = i + 1

# from sklearn.naive_bayes import MultinomialNB





# data_list = data.readline().strip().split('\r')
# inputs = []
# targets = []
# for row in data_list[1:20]:
# 	arr = row.split(',')
# 	print arr
# 	# inputs.append(arr[0])
# 	# targets.append(arr[2])
# 	pass

# print inputs[1:20]
