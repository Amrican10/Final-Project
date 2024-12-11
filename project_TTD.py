from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from time import time
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import sys,argparse


    
    
def parsePoint(line):
    #values = [float(x) for x in line.split(',')]
    values=[]
    for x in line.split(','):
        f=0
        try:
            f=eval(x)
        except: 
            f=-1
        values.append(f)
    #values = [float(x) for x in line.split(',')]
    y=0
    try:
        y=values[0]
        values.pop(0)
    except: 
        y=-1
    return LabeledPoint(y,values)
    

def readfold(f,sparkContext):
    data = sc.textFile("./final_train.csv")
    parsedData = data.map(parsePoint).repartition(28).cache()
    return parsedData
    
def crossval(folds,hp):
    (nt,md,mb)=hp
    crossvalacc=[]   
    for k in folds:
        train_folds = [folds[j] for j in folds if j is not k]
        train = train_folds[0]
        for fold in  train_folds[1:]:
            train=train.union(fold)
        train.repartition(28).cache()
        test = folds[k].repartition(28).cache()
        Mtrain = train.count()
        Mtest = test.count()
        
        print("Initiating fold %d with %d train samples and %d test samples" % (k,Mtrain,Mtest))
        startvf = time()
        
        model2 = RandomForest.trainClassifier(train, numClasses=2, categoricalFeaturesInfo={},
                                             numTrees=nt, featureSubsetStrategy="auto",
                                             impurity='gini', maxDepth=md, maxBins=mb)
                                             
    
        predictions = model2.predict(test.map(lambda x: x.features))
        labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
        testErr = labelsAndPredictions.filter(
            lambda lp: lp[0] != lp[1]).count() / float(test.count())
        print('validation ACC = ' + str(1-testErr))
        crossvalacc.append(1-testErr)
        print("validation fold time= ",time()-startvf)
    return np.mean(crossvalacc)

def tune(fn,sc):
    folds = {}
    hps=[(1,1,8),(2,2,16),(4,8,32),(8,16,64)]
    hp=0
    for k in range(fn):
        folds[k] = readfold("./fold"+str(k),sc)
    
    for hp in hps:
        print("|||||||||||||||||||||||||") 
        print("cross validation for hp ",hp)
        print("mean accuracy= ",crossval(folds,hp))
    
    
if __name__ == "__main__":
    #parallel
    # Load and parse the data
    
    sc = SparkContext(appName='project')
    startrd = time()
    data = sc.textFile("./final_train.csv")
#####   
    tune(4,sc)
#####    
    parsedData = data.map(parsePoint).repartition(28).cache()
    print("PROJECT") 
    
    print("train dataset size: ",parsedData.count())
    print("----------------------")                                        
    print("reading train time")
    print(time()-startrd)
    print("----------------------")
    
    startrt = time()
    test= sc.textFile("./final_test.csv")
    parsedtest = test.map(parsePoint).cache()
    print("test dataset size: ",parsedtest.count())
    print("----------------------")                                        
    print("reading test time")
    print(time()-startrt)
    print("----------------------")

#####    
    
    print("\n\n\n")
    print("+++++++++++++++++++++") 
    print("random forest")
    start_rf = time()
    model2 = RandomForest.trainClassifier(parsedData, numClasses=2, categoricalFeaturesInfo={},
                                         numTrees=8, featureSubsetStrategy="auto",
                                         impurity='gini', maxDepth=16, maxBins=64)
                                         
    print("----------------------")                                        
    print("training time")
    print(time()-start_rf)
    print("----------------------")  
    # Evaluate model on test instances and compute test error
    startt = time()
    
    # Evaluate model on test instances and compute test error
    predictions = model2.predict(parsedtest.map(lambda x: x.features))
    labelsAndPredictions = parsedtest.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(
        lambda lp: lp[0] != lp[1]).count() / float(parsedtest.count())
    metrics = BinaryClassificationMetrics(labelsAndPredictions.map(lambda lp: (lp[1],lp[0])))
    print('Test ACC = ' + str(1-testErr))
    # Area under precision-recall curve
    print("Area under PR = %s" % metrics.areaUnderPR)
    
    # Area under ROC curve
    print("Area under ROC = %s" % metrics.areaUnderROC)
    print("----------------------")                                        
    print("testing time")
    print(time()-startt)
    print("----------------------")
    
    
    
    
    
    
#####    
    
    

    
    
    # Build the model
    print("\n\n\n")
    print("+++++++++++++++++++++") 
    print("logistic regression") 
    start = time()
    model = LogisticRegressionWithLBFGS.train(parsedData)
    print("----------------------")                                        
    print("training time")
    print(time()-start)
    print("----------------------")  
    
    #testing
    # Compute raw scores on the test set
    startt = time()
    predictionAndLabels = parsedtest.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    
    # Instantiate metrics object
    metrics = BinaryClassificationMetrics(predictionAndLabels)
    
    testErr = predictionAndLabels.filter(
        lambda lp: lp[0] != lp[1]).count() / float(parsedtest.count())
    print('Test ACC = ' + str(1-testErr))
    
    # Area under precision-recall curve
    print("Area under PR = %s" % metrics.areaUnderPR)
    
    # Area under ROC curve
    print("Area under ROC = %s" % metrics.areaUnderROC)
    print("----------------------")                                        
    print("testing time")
    print(time()-startt)
    print("----------------------") 
    # $example off$
    
    
#####    
    
    
    print("\n\n\n")
    print("+++++++++++++++++++++") 
    print("random forest")
    start_rf = time()
    model2 = RandomForest.trainClassifier(parsedData, numClasses=2, categoricalFeaturesInfo={},
                                         numTrees=3, featureSubsetStrategy="auto",
                                         impurity='gini', maxDepth=4, maxBins=32)
                                         
    print("----------------------")                                        
    print("training time")
    print(time()-start_rf)
    print("----------------------")  
    # Evaluate model on test instances and compute test error
    startt = time()
    
    # Evaluate model on test instances and compute test error
    predictions = model2.predict(parsedtest.map(lambda x: x.features))
    labelsAndPredictions = parsedtest.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(
        lambda lp: lp[0] != lp[1]).count() / float(parsedtest.count())
    metrics = BinaryClassificationMetrics(labelsAndPredictions.map(lambda lp: (lp[1],lp[0])))
    print('Test ACC = ' + str(1-testErr))
    # Area under precision-recall curve
    print("Area under PR = %s" % metrics.areaUnderPR)
    
    # Area under ROC curve
    print("Area under ROC = %s" % metrics.areaUnderROC)
    print("----------------------")                                        
    print("testing time")
    print(time()-startt)
    print("----------------------")

#####
    
    print("\n\n\n")
    print("+++++++++++++++++++++") 
    print("random forest")
    start_rf = time()
    model2 = RandomForest.trainClassifier(parsedData, numClasses=2, categoricalFeaturesInfo={},
                                         numTrees=5, featureSubsetStrategy="auto",
                                         impurity='gini', maxDepth=7, maxBins=32)
                                         
    print("----------------------")                                        
    print("training time")
    print(time()-start_rf)
    print("----------------------")  
    # Evaluate model on test instances and compute test error
    startt = time()
    
    # Evaluate model on test instances and compute test error
    predictions = model2.predict(parsedtest.map(lambda x: x.features))
    labelsAndPredictions = parsedtest.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(
        lambda lp: lp[0] != lp[1]).count() / float(parsedtest.count())
    metrics = BinaryClassificationMetrics(labelsAndPredictions.map(lambda lp: (lp[1],lp[0])))
    print('Test ACC = ' + str(1-testErr))
    # Area under precision-recall curve
    print("Area under PR = %s" % metrics.areaUnderPR)
    
    # Area under ROC curve
    print("Area under ROC = %s" % metrics.areaUnderROC)
    print("----------------------")                                        
    print("testing time")
    print(time()-startt)
    print("----------------------")
    
#####
    
     # Build the model
    print("\n\n\n")
    print("+++++++++++++++++++++") 
    print("SVM") 
    start = time()
    # Build the model
    model = SVMWithSGD.train(parsedData, iterations=100)
    print("----------------------")                                        
    print("training time")
    print(time()-start)
    print("----------------------")  
    
    
    #testing
    # Compute raw scores on the test set
    startt = time()
    predictionAndLabels = parsedtest.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    
    # Instantiate metrics object
    metrics = BinaryClassificationMetrics(predictionAndLabels)
    
    testErr = predictionAndLabels.filter(
        lambda lp: lp[0] != lp[1]).count() / float(parsedtest.count())
    print('Test ACC = ' + str(1-testErr))
    
    # Area under precision-recall curve
    print("Area under PR = %s" % metrics.areaUnderPR)
    
    # Area under ROC curve
    print("Area under ROC = %s" % metrics.areaUnderROC)
    print("----------------------")                                        
    print("testing time")
    print(time()-startt)
    print("----------------------") 
    # $example off$
    
    
    
    
    
                                             
                                           
                                
       
#spark-submit --master local[28] --executor-memory 100G --driver-memory 100G project_TTD.py > p28

#spark-submit --master local[1] --executor-memory 100G --driver-memory 100G project_TTD.py > s1


#cd /courses/EECE5645.202510/students/davarakis.t/project

#srun --pty --export=ALL --partition courses --tasks-per-node 1 --nodes 1 --mem=10Gb --time=01:30:00 /bin/bash



#%%
