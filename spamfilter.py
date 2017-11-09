import os
import nltk as nl
import random
from pprint import pprint

rootdir = "C:\\Users\\Mounish\\Documents\\books\\Information Retrieval\\Spam filter\\Enron Spam"


ham_list = []
spam_list = []

def create_word_features(words):
    my_dict = dict( [ (word, True) for word in words] )
    return my_dict

for directories, subdirs, files in os.walk(rootdir):
    if (os.path.split(directories)[1] == 'ham'):
        for filename in files:
            with open(os.path.join(directories, filename),encoding="latin-1") as f:
                data = f.read()
                words = nl.word_tokenize(data)
                ham_list.append((create_word_features(words), "ham"))
                # pprint(ham_list)
    if (os.path.split(directories)[1] == 'spam'):
        if (os.path.split(directories)[1] == 'spam'):
            for filename in files:
                with open(os.path.join(directories, filename),encoding="latin-1") as f:
                    data = f.read()
                    words = nl.word_tokenize(data)
                    spam_list.append((create_word_features(words), "spam"))

combined_list = ham_list + spam_list
# pprint(combined_list[10:])
# pprint(ham_list[10:])
# pprint(spam_list[10:])
random.shuffle(combined_list)

training_part = int(len(combined_list) * .7)

# print("len(combined_list) " + str(len(combined_list)))
training_set = combined_list[:training_part]

test_set = combined_list[training_part:]

# print("len(training_set) "+str(len(training_set)))
# print(len(test_set))

classifier = nl.NaiveBayesClassifier.train(training_set)
# Find the accuracy, using the test data

accuracy = nl.classify.util.accuracy(classifier, test_set)
print("Accuracy is: ", accuracy * 100)
# print(classifier.show_most_informative_features(20))

# add precision and recall

msg1 = '''RSVP for our online information session!

 

Join us on Wednesday, November 15th at 6:00pm for an online information session highlighting our graduate programs and application requirements. Admissions counselors will be available to answer your questions about our university in order to help you start your graduate education with us. Please RSVP to join and learn more about UT.

Learn more about our graduate programs and apply. We are currently enrolling for Spring and Summer of 2018.

Business
Executive MBA
Professional MBA 
MBA (evening/part-time and day/full-time)
MS in Accounting
MS in Finance
MS in Marketing
MS in Entrepreneurship
MS in Cybersecurity
Certificate in Cybersecurity
Certificate in Nonprofit Management
Master of Fine Arts in Creative Writing
Master of Education in Educational Leadership
Master of Education in Curriculum and Instruction
Master of Science in Criminology and Criminal Justice
Master of Science in Cybersecurity
Master of Science in Exercise and Nutrition Science
Master of Science in Instructional Design and Technology
Master of Science in Nursing


RSVP Now!
 
If you have any questions, please feel free to contact us at (813) 258-7409 or utgrad@ut.edu.
'''

msg2 = '''​ECOLOGICALTEMPLATE FOR YOU
Etiam ex mi, viverra quis aliquam ut, tempus bibendum orci. Integer el.
LOREMIPSUM

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqu.
Obraz
LOREM IPSUM
​LOREMIPSUM​
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
LOREM IPSUM
LOREM IPSUM
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

Obraz
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmodtempor incididunt ut labore et dolore magna aliqua.
Obraz
'''

msg3 = '''Yes this is a good one'''

words = nl.word_tokenize(msg1)
features = create_word_features(words)
print("Message 1 is :", classifier.classify(features))

words = nl.word_tokenize(msg2)
features = create_word_features(words)
print("Message 2 is :", classifier.classify(features))

words = nl.word_tokenize(msg3)
features = create_word_features(words)
print("Message 3 is :", classifier.classify(features))