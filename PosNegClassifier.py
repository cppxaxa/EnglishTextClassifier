import pickle

from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

train = [
    ("What is the time now", "N"),
    ("When is the new year", "N"),
    ("What the age of my brother", "N"),
    ("What is succumb", "D"),
    ("How you do it", "D"),
    ("What is a cat", "D"),
    ("Where is NY", "L"),
    ("Restaurants nearby me", "L"),
    ("Where I play games", "L")
]
test = [
    ("Where you live", "L"),
    ("What is your age", "L"),
    ("When do you go school", "N"),
    ("What is dog in animals", "D")
]

#cl = NaiveBayesClassifier(train)
with open("classifier.pickle", "rb") as f:
    cl = pickle.load(f)

# Classify some text
print(cl.classify("What is a burger"))  # "pos"
print(cl.classify("Where can I get pizza"))   # "neg"

# Classify a TextBlob
blob = TextBlob("The beer was amazing. But the hangover was horrible. "
                "My boss was not pleased.", classifier=cl)
print(blob)
print(blob.classify())

for sentence in blob.sentences:
    print(sentence)
    print(sentence.classify())

# Compute accuracy
print("Accuracy: {0}".format(cl.accuracy(test)))

pickle.dump(cl, open("classifier.pickle", "wb"))

# Show 5 most informative features
cl.show_informative_features(5)