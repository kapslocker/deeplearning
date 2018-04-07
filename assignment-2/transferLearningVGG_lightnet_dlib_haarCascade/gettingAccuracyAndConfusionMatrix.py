import pandas as pd
import numpy as np
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')







dflabels=pd.read_csv('labels_2.csv',header=None)
labels= dflabels.values
labelsList=[currlabel[0]+currlabel[1] for currlabel in labels]

frameNumber=len(labelsList)

personList=['atulFace','fluteFace','pantFace','sadguruFace','sandeepFace','shailendraFace']

dfprediction=pd.read_csv('video2_bias32003200.csv',header=None)

predictionList=dfprediction.values
predictionList.shape


frameface= [curr[0].split('_') for curr in predictionList]

framefaceInt = [(int(x[0]), int(x[1])) for x in frameface]

# type(framefaceInt[0])
# type(predictionList[0])

# temp=predictionList[0]

# np.append(np.append(temp,framefaceInt[0][0]),framefaceInt[0][1])

mylist = []
for t in range(len(predictionList)):
    temp=np.append(np.append(predictionList[t],framefaceInt[t][0]),framefaceInt[t][1])
    mylist.append(temp)

mat = np.array(mylist)
# mat has ('fram_face','label',prob,frame,face)
# mat[0]

# bool_arr=np.array([row[3]==0 for row in mat])
# currFaces = mat[bool_arr]
# len(currFaces)
# currFaces
# bool_arr_new=np.array([(row[1] in personList) for row in currFaces])
# filteredRows=currFaces[bool_arr_new]
# type(filteredRows)
# filteredRows.shape
# filteredRows[0]
# maxProb=max([x[2] for x in filteredRows])
# maxProb
# bool_arr_new1=np.array([(row[2]==maxProb) for row in filteredRows])
# filteredRows[bool_arr_new1][0][1]

finalAns=[]
for frameNum in range(frameNumber):
    bool_arr = np.array([row[3]==frameNum for row in mat])
    currFaces = mat[bool_arr]
    # or currFaces.shape[0]>1
    if (currFaces.shape[0]==0 ) :
        finalAns.append('mixedFace')
    else:
        bool_arr_new=np.array([(row[1] in personList) for row in currFaces])
        filteredRows=currFaces[bool_arr_new]
        if filteredRows.shape[0]==0:
            finalAns.append('mixedFace')
        else:
            maxProb=max([x[2] for x in filteredRows])
            bool_arr_new1=np.array([(row[2]==maxProb) for row in filteredRows])
            finalAns.append(filteredRows[bool_arr_new1][0][1])
            
        
dict={}

# personList=['atulFace','fluteFace','pantFace','sadguruFace','sandeepFace','shailendraFace']
dict['atulFace']='AK'
dict['fluteFace']='FR'
dict['pantFace']='SP'
dict['sadguruFace']='SG'
dict['sandeepFace']='SM'
dict['shailendraFace']='SK'
dict['mixedFace']='jk'


predictionLabelFinal=[dict[x] for x in finalAns]
# labelsList[0]
# predictionLabelFinal[0]
sir=np.asarray(labelsList)
mine=np.asarray(predictionLabelFinal)
print ("correctAns"+str(np.sum(sir==mine)))
type(predictionLabelFinal)
type(labelsList)



cnf_matrix = confusion_matrix(sir, mine)
cnf_matrix
class_names=['AK||' 'FR||' 'SG||' 'SK||' 'SM||' 'SP||' 'jk']
unique_elements, counts_elements = np.unique(sir, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# val=3169
# labelsList[val]
# predictionLabelFinal[val]


