import mytool
import importlib
import pandas as pd
importlib.reload(mytool)


# mytool test
if __name__ == '__main__':

    # ###########
    # # 二分类
    # ###########
    modelset = mytool.Modelset('./Classifier_para.json',type='classifier')
    modelset.model_dict.keys()
    sample = pd.read_csv('./SampleClassification.csv')
    X = sample.iloc[:,0:19]
    y = sample.iloc[:,20]

    for i in modelset.model_dict.keys():
        modelset.fit(model_nm=i,X_train=X,y_train=y,eval_set=[(X,y)])
    for i in modelset.model_dict.keys():
        y_hat = modelset.predict(model_nm=i,X = X)
        aftpro = mytool.AfterProcessing()
        aftpro.eval_binary(eval_list=[(y,y_hat)],legend=['train'])

    gs = modelset.grid_search('XGBClassifier',X_train=X,y_train=y,parameters={'max_depth':[1,2,3,4,5]},cv=3)
    y_hat = modelset.predict(model_nm='XGBClassifier',X = X)
    # aftpro.eval_binary(eval_list=[(y,y_hat)],legend=['train'])


    # ###########
    # # 回归
    # ###########
    # modelset = mytool.Modelset('./Regressor_para.json',type='regressor')
    # modelset.model_dict.keys()
    # sample = pd.read_csv('./SampleRegression.csv')
    # X = sample.iloc[:,0:19]
    # y = sample.iloc[:,20]

    # for i in modelset.model_dict.keys():
    #     modelset.fit(model_nm=i,X_train=X,y_train=y,eval_set=[(X,y)])
    # for i in modelset.model_dict.keys():
    #     y_hat = modelset.predict(model_nm=i,X = X)
    #     aftpro = mytool.AfterProcessing()
    #     aftpro.eval_regression(eval_list=[(y,y_hat)],legend=['train'])

    # gs = modelset.grid_search('XGBRegressor',X_train=X,y_train=y,parameters={'max_depth':[1,2,3,4,5]},cv=3)
    # gs = modelset.fit(model_nm='XGBRegressor',X_train=X,y_train=y,eval_set=[(X,y)])
    # y_hat = modelset.predict(model_nm='XGBRegressor',X = X)
    # aftpro.eval_regression(eval_list=[(y,y_hat)],legend=['train'])


    ###########
    # 随机森林/决策树路径解析
    ###########
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.ensemble import RandomForestClassifier
    # sample = pd.read_csv('./SampleClassification.csv')
    # X = sample.iloc[:,0:19]
    # y = sample.iloc[:,20]
    # decision_tree = DecisionTreeClassifier(max_depth=5)
    # decision_tree.fit(X,y)
    # aftmethod = mytool.Afterprocessing()
    # df = aftmethod.tree_export(decision_tree,max_depth = 5)
    # tree = aftmethod.tree_print(decision_tree,max_depth = 5)
    # print(tree)


    #############
    # BetterTreeClassifier test
    #############
    # sample = pd.read_csv('./SampleClassification.csv')
    # X = sample.iloc[:,0:19]
    # y = sample.iloc[:,20]
    # BT = mytool.BetterTreeClassifier(max_depth=2,fit_rounds = 2)
    # BT.fit(X,y)

    # BT.rules

    # BT.predict(X)



from sklearn2pmml import sklearn2pmml, PMMLPipeline,make_pmml_pipeline
import joblib
from xgboost import XGBClassifier
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.8,
              colsample_bynode=0.8, colsample_bytree=0.8, 
              importance_type='gain', 
              learning_rate=0.1, max_depth=5,
              n_estimators=100, n_jobs=-1)
xgb.fit(X,y)              
pipeline = PMMLPipeline([("classifier", xgb)])
pipeline.fit(X,y)
joblib.dump(pipeline,'/Users/yu/Downloads/xgb.pkl',compress=0)

pipeline.predict_proba(X)
sklearn2pmml(pipeline, "./xgb_v2.pmml", with_repr = True)
tmp = make_pmml_pipeline(xgb)
sklearn2pmml(tmp, "./xgb_v2.pmml", with_repr = True)



premethod = mytool.PreProcessing()
import numpy as np
premethod.chi2_test(np.array([[3713,30990-3713],[4386,33449-4386]]))
premethod.chi2_test(np.array([[463,4922],[501,4930]]))

premethod.chi2_test(np.array([[1040,68378],[1961,209621]]))
premethod.chi2_test(np.array([[871,68377],[1412,209620]]))

