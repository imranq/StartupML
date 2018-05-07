import numpy as np
import sklearn
import csv 
import string
import nn


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

# unique list from targets
categories = sorted(list(set(targets)))
# print categories
# print inputs

train_num = 2000

# get a list of tokenized words, returns an array with unique values
# ignore capitalized words
# remove 
def tokenizeWords (str_desc):
	exclude = set(string.punctuation)
	token = str_desc.lower().strip()
	token = ''.join(ch for ch in token if ch not in exclude) #string punctuation

	token_list = list(set(token.split(" ")))
	# for word in token_list:
		
	# 	pass
	return token_list

def getNN_input(str_input, nn_cat):
	nn_input = np.zeros([len(nn_cat)])+0.01
	# returns false if there are not words found
	tokens = tokenizeWords(str_input)
	exists = False
	for t in tokens:
		index = nn_cat.index(t) if t in nn_cat else -1
		if index != -1:
			nn_input[index]=0.99
			exists = True
		pass
	if exists==True:
		return np.asfarray(nn_input)
	else:
		return False
	pass

def getNN_output(category, cat_list):
	nn_output = np.zeros([len(cat_list)])+0.01
	# print category
	index = cat_list.index(category)

	nn_output[index]=0.99

	return np.asfarray(nn_output)
	pass


nn_cat = []
train_num = 2000

corpus = "".join(inputs[0:train_num])

nn_cat = sorted(list(set(tokenizeWords(corpus))))

# print np.where(getNN_input(tokenizeWords(inputs[2]), nn_cat)==0.99)
# print nn_cat
# print tokenizeWords(inputs[2])
# print getNN_input(tokenizeWords(inputs[2]), nn_cat)

inputnodes = len(nn_cat)
hiddennodes = int(1/3.0*inputnodes)
outputnodes = len(categories)
learningrate = 0.3

# print inputnodes
# print outputnodes
# print categories.index('workforce of the future')

nn = nn.neuralnetwork(inputnodes,hiddennodes,outputnodes,learningrate)



for x in range(0,train_num):
	nn_input = getNN_input(inputs[x], nn_cat)
	nn_output =  getNN_output(targets[x],categories)
	# print (nn_input.shape)
	# print (nn_output.shape)

	nn.train(nn_input,nn_output)
	pass

for test_num in range(3000, 3200):
	test = getNN_input(inputs[test_num], nn_cat)
	actual_result = targets[test_num]
	result_matrix = nn.query(test)
	predicted_result = categories[np.argsort(result_matrix)[0]]
	print "%s > %s > %s > %s" % (inputs[test_num], categories[np.argsort(result_matrix)[-1]],categories[np.argsort(result_matrix)[-2]], categories[np.argsort(result_matrix)[-3]]) 
	# print "%s was tested with result %s, with the actual label being %s" % (inputs[test_num], actual_result, predicted_result)
	pass



# print result_matrix

# print ["yo", "my", "bro", "is"].index("b")
# need a neural network that accepts all tokens and returns the values of the category labels


