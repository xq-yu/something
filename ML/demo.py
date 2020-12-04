import mytool
import importlib
import pandas as pd
importlib.reload(mytool)


#模型快速试跑与调参
if __name__ == '__main__':

    # ###########
    # # 二分类
    # ###########
    # modelset = mytool.Modelset('./Classifier_para.json',type='classifier')
    # modelset.model_dict.keys()
    # sample = pd.read_csv('./SampleClassification.csv')
    # X = sample.iloc[:,0:19]
    # y = sample.iloc[:,20]

    # for i in modelset.model_dict.keys():
    #     modelset.fit(model_nm=i,X_train=X,y_train=y,eval_set=[(X,y)])
    # for i in modelset.model_dict.keys():
    #     y_hat = modelset.predict(model_nm=i,X = X)
    #     aftpro = mytool.afterprocessing()
    #     aftpro.eval_binary(eval_list=[(y,y_hat)],legend=['train'])

    # gs = modelset.grid_search('XGBClassifier',X_train=X,y_train=y,parameters={'max_depth':[1,2,3,4,5]},cv=3)
    # gs = modelset.fit(model_nm='XGBClassifier',X_train=X,y_train=y,eval_set=[(X,y)])
    # y_hat = modelset.predict(model_nm='XGBClassifier',X = X)
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
    #     aftpro = mytool.afterprocessing()
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
    # aftmethod = mytool.afterprocessing()
    # df = aftmethod.TreeExport(decision_tree,max_depth = 5)
    # tree = aftmethod.TreePrint(decision_tree,max_depth = 5)
    # print(tree)


    #############
    # BetterTreeClassifier test
    #############
    import mytool
    import importlib
    import pandas as pd
    importlib.reload(mytool)
    sample = pd.read_csv('./SampleClassification.csv')
    X = sample.iloc[:,0:19]
    y = sample.iloc[:,20]
    BT = mytool.BetterTreeClassifier(max_depth=5,fit_rounds = 3)
    BT.fit(X,y)

    BT.rules

    BT.predict(X)
