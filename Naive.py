from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import math
import nltk
nltk.download('punkt')
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from itertools import islice

with open("UpdatedReviewDataset.txt", 'r', encoding='utf-8') as f:
    content = (f.read())

print('Raw Data: ')
print(content)

content = content.replace('।' , ' ')
content = content.replace(',' , ' ')
content = content.replace('!' , ' ')

x = content.split("\n")

commentList = []

for i in range(len(x)):
    commentList.append(x[i])

commentList.pop(0)

for i in range(len(commentList)):
    print('Comment ', i + 1, ': ', commentList[i])

commentType = []
comment = []

for i in range(len(commentList)):
    commentType.append(commentList[i][0])
    comment.append(commentList[i][3:])

for i in range(len(commentType)):
    print('Comment Class ', i + 1, ': ', commentType[i])

for i in range(len(comment)):
    print('Comment ', i + 1, ': ', comment[i])

commentToken = []

for i in range(len(comment)):
   Token = word_tokenize(comment[i])
   commentToken.append(Token)

for i in range(len(commentToken)):
    print('Comment ', i + 1, ': ', commentToken[i])
print("\n\n")

#######STOP_WORD_REMOVAL

with open("stopWordList.txt", 'r', encoding='utf-8') as f:
    stopWord = (f.read())

stopWordList = []

stopWordList = stopWord.split("\n")

commonWord = []

for i in range(len(commentToken)):
    for j in range(len(commentToken[i])):
        for k in range(len(stopWordList)):
            if (commentToken[i][j] == stopWordList[k]):
                commonWord.append(commentToken[i][j])

for i in range(len(commonWord)):
    print('Common Word ', i + 1, ': ', commonWord[i])

for i in range(len(commonWord)):
    for j in range(len(commentToken)):
        if commonWord[i] in commentToken[j]:
            commentToken[j].remove(commonWord[i])

print('\n\n After Stop Word Remove:\n')

for i in range(len(commentToken)):
    print('Comment ', i + 1, ': ', commentToken[i])

f = open("pComment.txt", "w+", encoding='utf-8')

for i in range(len(commentToken)):
    for j in range(len(commentToken[i])):
        a = commentToken[i][j]
        f.write(str(a)+" ")
    #f.write("\n")
f.close()

with open("pComment.txt", 'r', encoding='utf-8') as f:
    content = (f.read())

x = content.split("\n")
print(len(x))

vectorizer = CountVectorizer(max_features=2500, min_df=0.05, max_df=0.95)
X = vectorizer.fit_transform([content]).toarray()
print(X)
X=X.transpose()
y = commentType
#for i in range(len(X)):
#    print('Comment: ', i+1, X)

print(len(X))
print(len(y))

X_train, X_test, y_train, y_test = train_test_split(X, commentType, test_size=0.2, random_state=0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("For MinMaxScalar:\n")
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))