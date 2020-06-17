class FeatureEngeering:
    import pandas as pd 
    import numpy as np 
    def missing_cal(self, df):
        """
        purpose：
            计算特征数据缺失占比
        input：
            df：输入数据集
        output:
            返回一个dataframe：col，missing_pct
        """
        missing_series = df.isnull().sum() / df.shape[0]
        missing_df = pd.DataFrame(missing_series).reset_index()
        missing_df = missing_df.rename(columns={'index':'col',0:'missing_pct'})
        missing_df = missing_df.sort_values('missing_pct',ascending=False).reset_index(drop = True)
        return missing_df

    def missing_filter(self, df, threshold):
        """
        purpose：
            删除缺失率过高的字段
        input：
            df：输入的数据集
            threshold: 缺失率阈值，删除缺失率大于等于该值的字段
        output：
            过滤后的dataframe
        """
        df_new = df.copy()
        missing_df = self.missing_cal(df)
        col_drop = list(missing_df.loc[missing_df.missing_pct>=threshold,'col'])
        df_new = df_new.drop(col_drop)
        return df_new
    
    def const_filter(self, df, threshold, subset=None):
        """
        purpose:
            删除某个常值占比过大的字段
        input:
            df:数据集
            threshold: 单值占比阈值
            subset:考虑的部分字段，默认为None
        output:
            过滤后的数据集
        """
        df_new = df.copy()
        col_drop = []
        if not subset:
            subset = df.columns
        for col in subset:
            const_pct = df_new[col].value_counts().iloc[0] / sum(df_new[col].notnull())
            if const_pct>=threshold:
                col_drop.append(col)
        df_new = df_new.drop(col_drop,axis = 1)
        return df_new

    def var_filter(self,df,threshold,subset=None):
        """
        purpose:
            删除某个方差过小的字段
        input:
            df:数据集
            threshold: 标准差阈值
            subset:考虑的部分字段，默认为None
        output:
            过滤后的数据集  
        """
        df_new = df.copy()
        if not subset:
            subset = list(df.columns)
        var = df_new[subset].std().drop_index()
        var.columns = ['col','std']
        col_drop = list(var.loc[var.std<=threshold,'col'])
        df_new = df_new.drop(col_drop,axis = 1)
        return df_new
    
    def feature_class(self, df, threshold, subset=None):
        """
        purpose:
            根据数值个数区分连续变量和离散变量
        input:
            df:数据集
            threshold: 标准差阈值
            subset:考虑的部分字段，默认为None
        output:
            过滤后的数据集  
        """        
        return




import pandas as pd
tmp = pd.DataFrame({'a':[1,2,3],'b':[2,3,4]})

print(FeatureEngeering().missing_cal(tmp))
