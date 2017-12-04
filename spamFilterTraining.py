# algorithm for serializing and de-serializing a Python object structure. “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, and “unpickling” is the inverse operation, whereby a byte stream is converted back into an object hierarchy. Pickling (and unpickling) is alternatively known as “serialization”, “marshalling,” [1] or “flattening”, however, to avoid confusion, the terms used here are “pickling” and “unpickling”..

# Warning The pickle module is not intended to be secure against erroneous or maliciously constructed data. Never unpickle data received from an untrusted or unauthenticated source.

# Latest spam collection- intially used enron data.. but found it too old to work in the present scenario's. Got the
# latest data from the spam data available online suggested by ohan,, then stated training the data .. it took almost
#  2 days for the system to get trained. approximately 300K files were used for training.. that is 300K spam mails
# mails ...with a lesser quantities of HAM mail..which is not available to that extent. so future would be adding
# that mail.


import time
import pickle
import os
import nltk as nl
import random
rootdir = "C:\\Users\\Mounish\\Documents\\books\\Information Retrieval\\Spam filter\\Enron Spam"


ham_list = []
spam_list = []

def create_word_features(words):
    my_dict = dict( [ (word, True) for word in words] )
    return my_dict

for directories, subdirs, files in os.walk(rootdir):
    if ('ham' in os.path.split(directories)[1]):
        for filename in files:
            # print(filename )
            with open(os.path.join(directories, filename),encoding="latin-1") as f:
                data = f.read()
                words = nl.word_tokenize(data)
                ham_list.append((create_word_features(words), "ham"))
    if ('spam' in os.path.split(directories)[1] ):
            for filename in files:
                # print(filename)
                with open(os.path.join(directories, filename),encoding="latin-1") as f:
                    data = f.read()
                    words = nl.word_tokenize(data)
                    spam_list.append((create_word_features(words), "spam"))

combined_list = ham_list + spam_list

random.shuffle(combined_list)

training_part = int(len(combined_list) * .7)

print("len(combined_list) " + str(len(combined_list)))
training_set = combined_list[:training_part]

# test_set = combined_list[training_part:]

print("len(training_set) "+str(len(training_set)))
# print(len(test_set))

classifier = nl.NaiveBayesClassifier.train(training_set)
print(classifier.show_most_informative_features())
f = open('my_classifier.p', 'wb')
pickle.dump(classifier, f)
f.close()

# Find the accuracy, using the test data
# print(classifier.show_most_informative_features())
# accuracy = nl.classify.util.accuracy(classifier, test_set)
#
# f = open('my_classifier.p', 'rb')
# classifier = pickle.load(f)
# f.close()
# print("Model Trained and pickled..!")
#
# # test_set = combined_list[training_part:]
# print(classifier.show_most_informative_features())
# # accuracy = nl.classify.util.accuracy(classifier, test_set)
# # print("Accuracy is: ", accuracy * 100)
#
# # print("Accuracy is: ", accuracy * 100)
# t0 = time.time()
# print(time.time() - t0, "Time")
