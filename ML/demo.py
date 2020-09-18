import mytool
import importlib
import pandas as pd
importlib.reload(mytool)


#模型快速试跑与调参
modelset = mytool.ModelClassifier('./model_para.json')
modelset.model_dict.keys()
sample = pd.read_csv('./SampleClassification.csv')
X = sample.iloc[:,0:19]
y = sample.iloc[:,20]

for i in modelset.model_dict.keys():
    modelset.fit(model_nm=i,X_train=X,y_train=y,eval_set=[(X,y)])
    y_hat = modelset.predict_prob(model_nm=i,X = X)
    aftpro = mytool.afterprocessing()
    aftpro.eval_binary(eval_list=[(y,y_hat)],legend=['train'])

gs = modelset.grid_search('XGBClassifier',X_train=X,y_train=y,parameters={'max_depth':[1,2,3,4,5]})
gs = modelset.fit(model_nm='XGBClassifier',X_train=X,y_train=y,eval_set=[(X,y)])

