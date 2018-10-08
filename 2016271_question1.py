#Surabhi S Nath
#2016271

import numpy
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import math
from sklearn.metrics import classification_report
import sys
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

#Function which returns Sensitivity, Specificity, Accuracy and MCC
def calc_performance(actual, probab, thresh):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    #Count TP, TN, FP, FN based on threshold
    for i in range(len(actual)):
        if(actual[i] == 1 and probab[i] >= thresh):
            TP += 1
        elif(actual[i] == -1 and probab[i] < thresh):
            TN += 1
        elif(actual[i] == -1 and probab[i] >= thresh):
            FP += 1
        elif(actual[i] == 1 and probab[i] < thresh):
            FN +=1
    
    #Calculate Accuracy
    if((TP + TN + FP + FN) == 0):
        accuracy = 0
    else:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    #Calculate Sensitivty
    if((TP + FN) == 0):
        sensitivity = 0
    else:
        sensitivity = TP / (TP + FN) 
    #Calculate Specificity
    if((TN + FP) == 0):
        specificity = 0
    else:
        specificity = TN / (TN + FP)
    #Calculate MCC
    if(((TP + FP) * (TP + FN) * (TN + FN) * (TN + FP)) == 0):
        mcc = 0
    else:
        mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FN) * (TN + FP))

    return sensitivity, specificity, accuracy, mcc
    
def main(argv):

    infile1 = argv[0]   #Neg file
    infile2 = argv[1]   #Pos file
    outfile1 = argv[2]  #TrainX
    outfile2 = argv[3]  #TestX
    outfile3 = argv[4]  #TrainY
    outfile4 = argv[5]  #TestY
    
    neg = []
    #Open file having negative examples
    with open(infile1) as f:
        next(f) #Ignore header

        #Read line by line
        for line in f:
            line = line.replace('\n','')

            #Make list of the float values
            expressions = list(map(float,line.split(',')))
            #Append label
            expressions.append(-1)

            #Make array of negative cases
            neg.append(expressions)


    pos = []
    #Open file having positive examples
    with open(infile2) as f:
        next(f) #Ignore header

        #Read line by line
        for line in f:
            line = line.replace('\n','')

            #Make list of the float values
            expressions = list(map(float,line.split(',')))
            #Append label
            expressions.append(1)

            #Make array of positive cases
            pos.append(expressions)


    length = len(pos[0])
    l_pos = len(pos)
    l_neg = len(neg)

    neg = numpy.array(neg)
    pos = numpy.array(pos)

    #---------------------------------------------------------------
    #Split 80:20

    Xtrain_neg = neg[0:math.ceil(0.8*l_neg),0:length-1]
    Ytrain_neg = neg[0:math.ceil(0.8*l_neg),length-1]

    Xtest_neg = neg[math.ceil(0.8*l_neg):l_neg,0:length-1]
    Ytest_neg = neg[math.ceil(0.8*l_neg):l_neg,length-1]

    Xtrain_pos = pos[0:math.ceil(0.8*l_pos),0:length-1]
    Ytrain_pos = pos[0:math.ceil(0.8*l_pos),length-1]

    Xtest_pos = pos[math.ceil(0.8*l_pos):length,0:length-1]
    Ytest_pos = pos[math.ceil(0.8*l_pos):length,length-1]

    Xtrain_set = numpy.concatenate((Xtrain_neg, Xtrain_pos))
    Ytrain_set = numpy.concatenate((Ytrain_neg, Ytrain_pos))
    Xtest_set = numpy.concatenate((Xtest_neg, Xtest_pos))
    Ytest_set = numpy.concatenate((Ytest_neg, Ytest_pos))

    #----------------------------------------------------------------
    
    #Write to files
    trainX = open(outfile1,'w')
    trainX.write("[\n")
    for i in range(0, len(Xtrain_set)):
        trainX.write(numpy.array2string(Xtrain_set[i])+"\n")
    trainX.write("]")
    trainX.close()

    trainY = open(outfile2,'w')
    trainY.write(numpy.array2string(Ytrain_set))
    trainY.close()

    testX = open(outfile3,'w')
    testX.write("[\n")
    for i in range(0, len(Xtest_set)):
        testX.write(numpy.array2string(Xtest_set[i])+"\n")
    testX.write("]")
    testX.close()

    testY = open(outfile4,'w')
    testY.write(numpy.array2string(Ytest_set))
    testY.close()

    #----------------------------------------------------------------
    # Training

    # SVM
    param = [{'C': [0.1,0.3,1,2.5,5,10,50,100,500,1000], 'kernel': ['linear']},{'C': [0.1,0.3,1,2.5,5,10,50,100,500,1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
    svc = svm.SVC(probability=True)

    #Find best parameters using Grid Search
    classifier_SVM = GridSearchCV(svc, param)
    #Fit model on data
    classifier_SVM.fit(Xtrain_set, Ytrain_set)
    #Make predictions
    predictions_SVM = classifier_SVM.predict(Xtest_set)
    #Get scores
    Ypred_scores = classifier_SVM.predict_proba(Xtest_set)
    Ypred_scores = [i[1] for i in Ypred_scores]

    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    savesens,savespec,saveacc,savemcc = [0,0,0,0]   #Initalize to 0
    min_diff = sys.maxsize
    #Iterate over all thresholds
    for i in thresholds:
        a,b,c,d = calc_performance(Ytest_set, Ypred_scores, i)
        #Find best performing model (Sens>Spec, Diff between Sens and SPec min)
        if((a-b)>0 and (a-b)<min_diff):
            savesens = a
            savespec = b
            saveacc = c
            savemcc = d

    #Find AUC
    saveauc = roc_auc_score(Ytest_set, Ypred_scores)

    print("Performance of Best SVM: ")
    print("Sensitivity: ",savesens, "\nSpecificity: ", savespec, "\nAccuracy: ", saveacc, "\nMCC: ", savemcc, "\nAUCROC: ", saveauc)
    print()
    
    
    #-----------------------------------------------------------------

    # ANN

    param = [{'solver': ['lbfgs','adam','sgd'], 'alpha': [0.1,0.01,0.001], 'learning_rate' : ['constant', 'adaptive'], 'hidden_layer_sizes': [(5,2)], 'activation' : ['identity']}]

    classifier_ANN = GridSearchCV(MLPClassifier(), param)

    #Fit model on data
    classifier_ANN.fit(Xtrain_set, Ytrain_set)

    #Make predictions
    predictions_ANN = classifier_ANN.predict(Xtest_set)

    #Get scores
    Ypred_scores2 = classifier_ANN.predict_proba(Xtest_set)
    Ypred_scores2 = [i[1] for i in Ypred_scores2]

    savesens,savespec,saveacc,savemcc = [0,0,0,0]   #Initalize to 0
    min_diff = sys.maxsize

    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    
    #Iterate over all thresholds
    for i in thresholds:
        a,b,c,d = calc_performance(Ytest_set, Ypred_scores2, i)
        if((a-b)>0 and (a-b)<min_diff):
            savesens = a
            savespec = b
            saveacc = c
            savemcc = d

    #Find AUC
    saveauc = roc_auc_score(Ytest_set, Ypred_scores2)

    print("Performance of Best ANN: ")
    print("Sensitivity: ",savesens, "\nSpecificity: ", savespec, "\nAccuracy: ", saveacc, "\nMCC: ", savemcc, "\nAUCROC: ", saveauc)

if __name__ == "__main__":
   main(sys.argv[1:])
