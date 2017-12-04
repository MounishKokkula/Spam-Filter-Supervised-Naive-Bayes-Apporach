
import time
import pickle
import os
import nltk as nl
import random

f = open('my_classifier.p', 'rb')
classifier = pickle.load(f)
f.close()

def create_word_features(words):
    my_dict = dict( [ (word, True) for word in words] )
    return my_dict

# test_set = combined_list[training_part:]
print(classifier.show_most_informative_features())
# accuracy = nl.classify.util.accuracy(classifier, test_set)
# print("Accuracy is: ", accuracy * 100)

# print("Accuracy is: ", accuracy * 100)
# t0 = time.time()
# print(time.time() - t0, "Time")

msg1 = '''Hello th̓ere seُx master :-)
i need c0ck ri͏ght noِw ..͏. don't tell my hǔbbٚy.ٚ. ))
My sc͕rٞeٚe̻nname is Dorry.
My accֺo֔unt is h֯ere: http:nxusxbnd.GirlsBadoo.ru
C u late٘r!'''

msg2 = '''Hi Mounish, 

Are you deciding which city is best for your summer internship? Today is the final day to save up to $1,000 on any Dream Careers Internship Program!

Choose from our most popular destinations like New York City, Los Angeles, London, Barcelona, Hong Kong, or Paris. The Dream Careers Internship Programs offer the most beneficial way to intern in an exciting city as we take care of all of the details including a guaranteed internship placement, group housing, and exciting group activities on weekends! 

Interested? Getting started is easy! You can apply for the program at Summerinternships.com or talk to an Advisor. Plus, if this is the right program for you, if you sign up before November 30th, you can save up to $1,000. 

Call us at 800-251-2933 with any questions and prepare for a summer experience filled with travel, work experience, and new friendships.

 

ps. You can view our 2017 Program adventures through the Dream Careers FB or Instagram accounts.

 '''

msg3='''Hi how are you'''

words = nl.word_tokenize(msg1)
features = create_word_features(words)

print("Message 1 is :" ,classifier.classify(features))

words = nl.word_tokenize(msg2)
features = create_word_features(words)

print("Message 2 is :" ,classifier.classify(features))

words = nl.word_tokenize(msg3)
features = create_word_features(words)

print("Message 3 is :" ,classifier.classify(features))
