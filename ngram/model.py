
from nltk import word_tokenize
from collections import Counter
import operator
import pandas as pd
import re
from numpy.random import choice
import numpy as np
import sys

class sentence:

    def __init__(self):
        self.unigrams = []
        self.bigrams = []
        self.trigrams = []
        
    def createNgrams(self, sent):

        self.unigrams = sent.split()
        
        for i in range(len(self.unigrams)-1):

            if self.unigrams[i] != '</s>':
                self.bigrams.append(self.unigrams[i] + ' ' + self.unigrams[i+1])
            
        for i in range(len(self.unigrams)-2):

            if self.unigrams[i] != '</s>':

                if self.unigrams[i+1] != '</s>':

                    self.trigrams.append(self.unigrams[i] + ' ' + self.unigrams[i+1] + ' ' + self.unigrams[i+2])


def get_bigrams(sent):

    a = sentence()
    a.createNgrams(sent)

    return(a.bigrams)
    

def get_trigrams(sent):

    a = sentence()
    a.createNgrams(sent)

    return(a.trigrams)


def predict_next_word_interpolation(bigram):
    
    elements = []
    weights = []

    for i in generate_sent_map_interpolation[bigram]:

        elements.append(i)
        weights.append(prob_interpolation[i + '|' + bigram])

    
    weights = np.array(weights)
    weights /= weights.sum()
    nextword = choice(elements, p = weights)

    return(nextword)

    
def generate_sent_interpolation():
    
    bigram = '<s> <s>'
    final_sent = '<s> '
    next_word = predict_next_word_interpolation(bigram)
    bigram = bigram.split()[1] + ' '+next_word
    final_sent += next_word

    while bigram.split()[1] != '</s>':

        next_word = predict_next_word_interpolation(bigram)
        bigram = bigram.split()[1] + ' '+next_word
        final_sent += ' ' + next_word
    
    return(final_sent)


def predict_next_word_add1(bigram):
    
    elements = []
    weights = []
    for i in generate_sent_map_trigram[bigram]:
        elements.append(i)
        weights.append(prob_add1_trigram[i+'|'+bigram])

    
    weights=np.array(weights)
    weights /= weights.sum()
    nextword = choice(elements, p = weights)

    return(nextword)

    
def generate_sent_add1():
    
    bigram = '<s> <s>'
    final_sent = '<s> '
    next_word = predict_next_word_add1(bigram)
    bigram = bigram.split()[1]+' '+next_word
    final_sent+=next_word

    while bigram.split()[1] != '</s>':
        next_word = predict_next_word_add1(bigram)
        bigram = bigram.split()[1]+' '+next_word
        final_sent+=' '+next_word
    
    return(final_sent)


def calc_perplexity_interpolation(testsent, l1, l2, l3):

    global prob_add1_trigram
    global prob_add1_bigram
    global prob_add1_unigram
    global trigrams_count
    global bigrams_count
    global vocab
    global prob_interpolation
    
    trigrams_list = get_trigrams(testsent)
    
    temp = 1

    for i in range(len(trigrams_list)):

        b = trigrams_list[i].split()
        key = b[2] + '|' + b[0] + ' ' + b[1]

        if key not in prob_add1_trigram:
            val1 = (0+1)/(bigrams_count[b[0] + ' ' + b[1]] + vocab)
        else:
            val1 = prob_add1_trigram[key]
	
        key = b[2] + '|'+b[1]
		
        if key not in prob_add1_bigram:
           val2 = (0+1)/(unigrams_count[b[1]]+vocab)
        else:
           val2 = prob_add1_bigram[key]
			
        key = b[2]
		
        if key not in prob_add1_unigram:
           val3 = (0+1)/(2*vocab)
        else:
           val3 = prob_add1_unigram[key]
        
        val = l1*val1 + l2*val2 + l3*val3
        prob_interpolation[b[2] + '|' + b[0] + ' ' + b[1]] = val

        temp8 = b[0] + ' ' + b[1]

        if temp8 in generate_sent_map_interpolation.keys():
            generate_sent_map_interpolation[temp8].add(b[2])
        else:
            generate_sent_map_interpolation[temp8] = set({b[2]})

        temp = temp / val

    perplexity = temp**(1/len(trigrams_list))

    return(perplexity)
    

def tot_perplexity(test_data_v2, l1, l2, l3):

    sum_perplexity = 0

    for i in range(len(test_data_v2)):

        val = calc_perplexity_interpolation(test_data_v2[i], l1, l2, l3)
        sum_perplexity += val
        #print(val)
    
    return([sum_perplexity/len(test_data_v2), l1, l2, l3])
    
    
def best_lambda(dev_data_v2):
    allperplexity=[]
       
    # Perplexity of dev data
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.1, 0.8))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.8, 0.1))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.7, 0.2))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.2, 0.7))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.6, 0.3))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.3, 0.6))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.4, 0.5))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.5, 0.4))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.2, 0.1, 0.7))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.2, 0.7, 0.1))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.2, 0.2, 0.6))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.2, 0.6, 0.2))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.2, 0.3, 0.5))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.2, 0.5, 0.3))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.3, 0.1, 0.6))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.3, 0.6, 0.1))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.3, 0.2, 0.5))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.3, 0.5, 0.2))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.3, 0.3, 0.4))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.3, 0.4, 0.3))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.4, 0.1, 0.5))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.4, 0.5, 0.1))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.4, 0.2, 0.4))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.4, 0.4, 0.2))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.4, 0.3, 0.3))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.5, 0.1, 0.4))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.5, 0.4, 0.1))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.5, 0.2, 0.3))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.5, 0.3, 0.2))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.6, 0.1, 0.3))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.6, 0.3, 0.1))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.6, 0.2, 0.2))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.7, 0.1, 0.2))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.7, 0.2, 0.1))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.8, 0.1, 0.1))

    allperplexity.sort(key = operator.itemgetter(0), reverse = False)

    return(allperplexity[0])


def calc_perplexity_add1(testsent):

    global prob_add1_trigram
    global trigrams_count
    global bigrams_count
    global vocab
    
    trigrams_list = get_trigrams(testsent)
    
    temp = 1

    for i in range(len(trigrams_list)):

        b = trigrams_list[i].split()
        key = b[2] + '|' + b[0] + ' ' + b[1]

        if key not in prob_add1_trigram:
            val = (0+1)/(bigrams_count[b[0] + ' '+b[1]] + vocab)
        else:
            val = prob_add1_trigram[key]
        
        temp = temp / val

    perplexity = temp**(1/len(trigrams_list))

    return(perplexity)


def calc_add1_smoothing_unigram(testsent):

    global prob_add1_unigram
    global unigrams_count
    global vocab
    global c
    
    unigrams_list = testsent.split()

    for i in range(len(unigrams_list)):
        
        key = unigrams_list[i]
        value = (unigrams_count[unigrams_list[i]] + 1)/(c + vocab)
        prob_add1_unigram[key] = value


def calc_add1_smoothing_bigram(testsent):

    global prob_add1_bigram
    global unigrams_count
    global bigrams_count
    global vocab
    global generate_sent_map_bigram
    
    bigrams_list = get_bigrams(testsent)

    for i in range(len(bigrams_list)):
        
        b = bigrams_list[i].split()
        key = b[1] + '|' + b[0]

        value = (bigrams_count[bigrams_list[i]] + 1)/(unigrams_count[b[0]]+vocab)
        
        prob_add1_bigram[key] = value
        
        temp = b[0]

        if temp in generate_sent_map_bigram.keys():
            generate_sent_map_bigram[temp].add(b[1])

        else:
            generate_sent_map_bigram[temp] = set({b[1]})


def calc_add1_smoothing_trigram(testsent):

    global prob_add1_trigram
    global trigrams_count
    global bigrams_count
    global vocab
    global generate_sent_map_trigram
    
    trigrams_list = get_trigrams(testsent)
    
    for i in range(len(trigrams_list)):
        
        b = trigrams_list[i].split()
        key = b[2] + '|' + b[0] + ' ' + b[1]
        value = (trigrams_count[trigrams_list[i]] + 1)/(bigrams_count[b[0] + ' ' + b[1]] + vocab)
        
        prob_add1_trigram[key] = value
        
        temp = b[0] + ' ' + b[1]

        if temp in generate_sent_map_trigram.keys():
            generate_sent_map_trigram[temp].add(b[2])

        else:
            generate_sent_map_trigram[temp] = set({b[2]})


def replace_with_unk(train_data, unknown_set): 

    train_data_v2 = []

    for i in range(len(train_data)):

        k = train_data[i].split()
        sent = ''

        for j in range(len(k)):

            if k[j] in unknown_set:
                sent += '<UNK>'+' '

            else:
                sent += k[j]+' '

        train_data_v2.append(sent)

    return(train_data_v2)


def get_unk_set(train_data):

    corpus = ''
    unknown_set = set()
    
    for i in range(len(train_data)):
        corpus += train_data[i] + ' '
    
    a = sentence()
    a.createNgrams(corpus)
    
    del corpus
    unigrams_count = Counter(a.unigrams)
    
    for k,v in unigrams_count.items():

        if v < 5:
            unknown_set.add(k)
        
    return(unknown_set)


def preprocess_data(df):
    train_data = []
    
    for i in range(len(df)):
        
        if df.text[i] != 'text':
            sent = '<s> <s>'
            temp = df.text[i].lower() #lower
            temp = re.sub(r'\<.*\>', '', temp)
            temp = temp.replace('/', '').replace('{a', '').replace('{c', '').replace('{d', '').replace('{e', '').replace('{f', '').replace('}', '').replace('[', '').replace(']', '').replace('+', '').replace('#', '').replace('(', '').replace(')', '').replace('--', '').replace('-','')
            tokentemp = word_tokenize(temp)

            for i in range(len(tokentemp)):
                sent += ' ' + tokentemp[i]

            sent += ' </s>'

        train_data.append(sent)

    return(train_data)
    



unigrams_count = dict()
bigrams_count = dict()
trigrams_count = dict()
prob_add1_trigram = dict()
prob_add1_bigram = dict()
prob_add1_unigram = dict()
prob_interpolation = dict()
generate_sent_map_trigram = dict()
generate_sent_map_bigram = dict()
generate_sent_map_interpolation = dict() 

train = './data/train_set.csv'
dev = './data/dev_set.csv'
test = './data/test_set.csv'

print("Loading data..")

df = pd.read_csv(train)

print("Pre-processing data..")

train_data = preprocess_data(df)

print("Replacing words with freq less than 5 as <UNK>..")

train_unk_set = get_unk_set(train_data)
train_data_v2 = replace_with_unk(train_data, train_unk_set)

corpus = ''

for i in range(len(train_data_v2)):
    corpus += train_data_v2[i] + ' '

a = sentence()
a.createNgrams(corpus)

del corpus

unigrams_count = Counter(a.unigrams)
bigrams_count = Counter(a.bigrams)
trigrams_count = Counter(a.trigrams)

vocab = len(unigrams_count)
c = sum(unigrams_count.values())

print("Calculate add1 smoothing for unigrams, bigrams and trigrams..")


for i in range(len(train_data_v2)):

    calc_add1_smoothing_trigram(train_data_v2[i])
    calc_add1_smoothing_bigram(train_data_v2[i])
    calc_add1_smoothing_unigram(train_data_v2[i])


print("\n")
print("Running on test data..")

df2 = pd.read_csv(test)
test_data = preprocess_data(df2)
test_unk_set = get_unk_set(test_data)

full_unk_set = train_unk_set.union(test_unk_set)
test_data_v2 = replace_with_unk(test_data, full_unk_set)

print("Calculating perplexity for add1 smoothing..")

sum_perplexity = 0

for i in range(len(test_data_v2)):

    val = calc_perplexity_add1(test_data_v2[i])
    sum_perplexity += val
    
print("Perplexity of add1 smoothing: ", sum_perplexity/len(test_data_v2))

print("\n")
print("Calculating lambdas for simple interpolation on dev set..")

df3 = pd.read_csv(dev)
dev_data = preprocess_data(df3)
dev_data_v2 = replace_with_unk(dev_data, full_unk_set)

best_result = best_lambda(dev_data_v2)
l1 = best_result[1]
l2 = best_result[2]
l3 = best_result[3]

print("Best lambda1: ", best_result[1])
print("Best lambda2: ", best_result[2])
print("Best lambda3: ", best_result[3])

print("\n")
print("Calculating perplexity for simple interpolation on test set..")

sum_perplexity = 0

for i in range(len(test_data_v2)):

    val = calc_perplexity_interpolation(test_data_v2[i], l1, l2, l3)
    sum_perplexity += val

print("Perplexity of interpolation smoothing: ", sum_perplexity/len(test_data_v2))

print("\n")
print("Generating sentences using add1 smoothing:")

for i in range(20):

    k = generate_sent_add1()
    print (str(i) + '.', k)

print("\n")
print("Generating sentences using simple interpolation:")

for i in range(20):

    k = generate_sent_interpolation()
    print (str(i) + '.', k)
