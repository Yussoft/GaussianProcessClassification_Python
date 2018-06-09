# -*- coding: utf-8 -*-
"""
Created on Thu Jun 5 12:33:23 2018

@author: Jesús Sánchez de Castro
"""
import pandas as pd
import numpy as np
import GPy as gpy 
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
#------------------------------------------------------------------------------
# Load the folds of data
data_dir = "C://Users//Yus//Desktop//PracticaGPC//"
healthy_folds = []
malign_folds = []

# Create different dataframes for each fold and add the labels
for i in range(1,6):
    h = pd.read_csv(data_dir+"healthy_fold"+str(i)+".csv", sep=",", 
                    header = None)
    # Add healthy label
    h['label']=1
    healthy_folds.append(h)
    
    m = pd.read_csv(data_dir+"malign_fold"+str(i)+".csv", sep=",",
                    header = None)
    # Add malign label
    m['label']=-1
    malign_folds.append(m)
#------------------------------------------------------------------------------
# Create test and train folds
test_folds = []
train_folds = []
for i in range(0, len(healthy_folds)):
    test_folds.append(pd.concat([healthy_folds[i],
                                 malign_folds[i]]).reset_index(drop=True))
    aux_h = healthy_folds[:i] + healthy_folds[i+1:]
    aux_m = malign_folds[:i] + malign_folds[i+1:]
    aux = aux_h + aux_m
    train_folds.append(pd.concat(aux).reset_index(drop=True))
    
print("Dimensiones de los folds de test:")
for elem in test_folds:
    print(elem[elem.label == 1].shape,"+",elem[elem.label == -1].shape,"=",
          elem.shape)
    
print("Dimensiones de los folds de train:")
for elem in train_folds:
    print(elem.shape)
    
#------------------------------------------------------------------------------
# Normalizar
for i in range(0,5):
    n_test = test_folds[i].shape[0]
    n_train = train_folds[i].shape[0]
#    print("Test",test_folds[i].shape, "train",train_folds[i].shape)
    
    fold = pd.concat([test_folds[i].iloc[:,:-1], train_folds[i].iloc[:,:-1]])
    fold = pd.DataFrame(preprocessing.scale(fold))
    
    aux_test = pd.DataFrame(fold.iloc[0:n_test,:])
    aux_train = pd.DataFrame(fold.iloc[n_test:n_train,:])
    aux_test["label"]=test_folds[i].iloc[0:n_test,-1]
    aux_train["label"]=train_folds[i].iloc[n_test:n_train,-1]
    


#------------------------------------------------------------------------------
# Crear los folds de entrenamiento   
balanced_train_folds = []

for i in range(0, len(train_folds)):
    aux = []
    # Split train from ith fold in healthy and maling by labels
    healthy_train = train_folds[i][train_folds[i].label == 1]
    malign_train = train_folds[i][train_folds[i].label == -1]
    
    # Number of samples of each healthy train subset (4 of them of equal size)
    n_sample = len(healthy_train)//4

    # Shuffle the IDs
    healthy_train.reset_index(drop=True)
    np.random.seed(77183)
    shuffled_healthy_train = healthy_train.loc[np.random.permutation(len(healthy_train))]
    
    aux.append(pd.concat([shuffled_healthy_train.iloc[0:n_sample,:], malign_train]))
    aux.append(pd.concat([shuffled_healthy_train.iloc[n_sample:(n_sample*2),:], malign_train]))
    aux.append(pd.concat([shuffled_healthy_train.iloc[n_sample*2:(n_sample*3),:], malign_train]))
    aux.append(pd.concat([shuffled_healthy_train.iloc[n_sample*3:(n_sample*4),:], malign_train]))
  
    balanced_train_folds.append(aux)
    # Check disjoints groups
#    print("Check if index of the splitted folds are repited:")
#    for elem in aux:
#        i = elem[0].index
#        ii = elem[1].index
#        iii = elem[2].index
#        iiii = elem[3].index
#        
#        print(set(i.isin(ii)))
#        print(set(i.isin(iii)))
#        print(set(i.isin(iiii)))
#        print(set(ii.isin(iii)))
#        print(set(ii.isin(iiii)))
#        print(set(iii.isin(iiii)))
    
#-----------------------------------------------------------------------------
# Entrenamiento
conf_matrix_RBF = []
conf_matrix_linear = []
probs_RBF_folds = []

probs = 0
# Train and test for each test folds
for fold in range(0, len(test_folds)):
    fold_test = test_folds[fold]
    fold_X_test = fold_test.loc[:, fold_test.columns != "label"]
    fold_y_test = fold_test.loc[:, fold_test.columns == "label"]

    print("##################################################################")
    print("Test para fold",fold+1,", dimension:",fold_test.shape)
    print(" ")

    # Auxiliar confusion matrix
    aux_cm_RBF = []
    aux_cm_linear = []
    sum_RBF = 0
    sum_linear = 0
    
    # For each train set in fold x
    probs_RBF_ = []
    for i in range(0, 4):
        
        data = balanced_train_folds[fold][i]
        X = data.loc[:, data.columns != "label"]
        y = data.loc[:, data.columns == 'label']
        
#        print("_____________________________________________________")
#        print("Fold",fold+1,", modelo:",i+1)
#        print("train:",data.shape)
#        print("_____________________________________________________")
        
#        RBF_kernel = gpy.kern.RBF(input_dim = X.shape[1], variance=1.0,
#                                  lengthscale=1.9)
        
        RBF_kernel = gpy.kern.Linear(input_dim = X.shape[1])

        model_RBF = gpy.models.GPClassification(X,y,
                                                kernel = RBF_kernel, 
                                                mean_function = None)
        # Optimize 
        model_RBF.optimize(messages=0)

        # Get the probabilities
        probs_RBF = model_RBF.predict(fold_X_test.as_matrix())[0]
        probs_RBF_.append(probs_RBF)
    
    # Add the probs of the fold
    probs_RBF_folds.append(probs_RBF_)
    
    # Calculate the mean probability of 
    fold_prob = sum(probs_RBF_)/4
    print("RBF CONFUSION MATRIX FOLD",fold+1)
    cm_RBF = gpy.util.classification.conf_matrix(fold_prob,
                                                 fold_y_test.as_matrix(),
                                                 threshold  = 0.25,
                                                 names = ['1','-1'])
    
    TP = cm_RBF[1]
    FN = cm_RBF[4]
    FP = cm_RBF[2]
    TN = cm_RBF[3]
    
    accuracy=(TP+TN)/(TP+FN+FP+TN)
    specificity=TN/(TN+FP)
    sensitivity=TP/(TP+FN)
    precision=TP/(TP+FP)
    F_score=(2*precision*sensitivity)/(precision+sensitivity)
    
    print("Accuracy:",accuracy)
    print("Specificity:",specificity)
    print("Sensitivity:",sensitivity)
    print("Precision:",precision)
    print("FScore:",F_score)
    print("AUC:",roc_auc_score(y_true=fold_y_test.as_matrix(), y_score=fold_prob))
    
    conf_matrix_RBF.append(cm_RBF)
        
#Lineral_kernel = gpy.kern.Linear(input_dim=10)
#model_linear = gpy.models.GPClassification(X,y,kernel = Lineral_kernel,
#                                                   mean_function=None)
#    probs_linear = model_linear.predict(fold_X_test.as_matrix())[0]
#        print("Linear CONFUSION MATRIX")
#        cm_linear = gpy.util.classification.conf_matrix(probs_linear, 
#                                                     fold_y_test.as_matrix())
#    aux_cm_linear.append(cm_linear)
#        sum_linear = sum_linear + (1-cm_linear[0])
#            print("ACCURACY MEDIA Kernel RBF:",sum_linear/4)