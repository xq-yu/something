import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from scipy.stats import chi2,chi2_contingency
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import json

class PreProcessing:
    def missing_cal(self, df):
        """
        purpose：
            计算特征数据缺失占比
        input：
            df:pd.DataFrame  输入数据集
        output:
            missing_df:pd.DataFrame  [col，missing_pct]
        """
        missing_series = df.isnull().sum() / df.shape[0]
        missing_df = pd.DataFrame(missing_series).reset_index()
        missing_df = missing_df.rename(
            columns={'index': 'col', 0: 'missing_pct'})
        missing_df = missing_df.sort_values(
            'missing_pct', ascending=False).reset_index(drop=True)
        return missing_df

    def missing_filter(self, df, threshold, subset = None):
        """
        purpose：
            删除缺失率过高的字段
        input：
            df：pd.DataFrame输入的数据集
            threshold: float 缺失率阈值，删除缺失率大于等于该值的字段
        output：
            df_new: pd.DataFrame 过滤后的dataframe
        """
        if not subset:
            subset = df.columns
            
        df_new = df.copy()
        missing_df = self.missing_cal(df[subset])
        col_drop = list(
            missing_df.loc[missing_df.missing_pct >= threshold, 'col'])
        df_new = df_new.drop(col_drop,axis=1)
        return df_new

    def const_filter(self, df, threshold, subset=None):
        """
        purpose:
            删除某个常值占比过大的字段
        input:
            df:pd.DataFrame数据集
            threshold: float 单值占比阈值
            subset:list 考虑的部分字段，默认为None
        output:
            df_new: pd.DataFrame 过滤后的数据集
        """
        df_new = df.copy()
        col_drop = []
        if not subset:
            subset = df.columns
        for col in subset:
            const_pct = df_new[col].value_counts(
            ).iloc[0] / sum(df_new[col].notnull())
            if const_pct >= threshold:
                col_drop.append(col)
        df_new = df_new.drop(col_drop, axis=1)
        return df_new

    def var_filter(self, df, threshold, subset=None):
        """
        purpose:
            删除某个方差过小的字段
        input:
            df:pd.DataFrame 数据集
            threshold: float标准差阈值
            subset:list 考虑的部分字段，默认为None
        output:
            df_new: pd.DataFrame 过滤后的数据集  
        """
        df_new = df.copy()
        if not subset:
            subset = list(df.columns)
        var = df_new[subset].std().drop_index()
        var.columns = ['col', 'std']
        col_drop = list(var.loc[var.std <= threshold, 'col'])
        df_new = df_new.drop(col_drop, axis=1)
        return df_new

    def datatype_clean(self, df):
        """
        purpose:
            对数据类型进行规整，将可以转化为数值型的字段转换为数值类型
        input:
            df:pd.DataFrame 数据集
        output:
            df_new:pd.DataFrame 转换后的数据集
        """
        df_new = df.copy()
        col_type = pd.DataFrame(df_new.dtypes).reset_index()
        col_type.columns = ['col', 'type']
        col_object = list(col_type.loc[col_type.type == 'object', 'col'])
        for col in col_object:
            ser_tmp = df_new[col].copy()
            ser_tmp.loc[ser_tmp.isna()] = None
            ser_tmp = ser_tmp.astype('str').str.strip()
            try:
                ser_tmp = pd.to_numeric(ser_tmp)
            except Exception as e:
                pass
            df_new[col] = ser_tmp
        return df_new

    def feature_class(self, df, threshold, subset=None):
        """
        purpose:
            根据数值个数区分连续变量和离散变量
        input:  
            df:pd.DataFrame 数据集
            threshold: float 取值小于该阈值的字段为类别型变量，大于该值为连续型变量
            subset:list 考虑的部分字段，默认为None
        output:
            feature_cat: list 类别型变量（取值数量小于等于阈值的字段）
            feature_num: list 数值型变量，类别为数值型，且取值个数大于阈值
            feature_res: list 存在矛盾的变量或其他类型字段，需进一步判断别（1.非object类型字段和数值型字段；2.object型字段但是取值数量大于阈值）
        """
        feature_cat = []
        feature_num = []
        feature_res = []
        for col in df.columns:
            if df[col].dtype() == 'object':
                if len(df[col].drop_duplicates()) <= threshold:
                    feature_cat.append(col)
                else:
                    feature_res.append(col)
            elif df[col].dtype() in ('int64', 'float64'):
                if len(df[col].drop_duplicates()) <= threshold:
                    feature_cat.append(col)
                else:
                    feature_num.append(col)
            else:
                feature_res.append(col)
        return feature_cat, feature_num, feature_res


    def _categary_merge(self,col,threshold,tag):
        """
        purpose:
            合并样本数较少的类别
        input:  
            trainset: pd.Series 数据列1
            threshold：float 将样本数小于 threshold 的类别合并
            tag: 合并类别的标签
        output:
            col: pd.Series 合并后的新列
            merge_cats: list 合并的类别列表
        """
        tmp = col.value_counts()
        merge_cats = list(tmp.loc[tmp<threshold].index)
        col = col.map(lambda x: tag if (x in merge_cats) else x)
        return col,merge_cats
        
        
    def _psi_one_col(self, trainset, testset, category_type=False):
        """
        purpose:
            计算一个变量的psi稳定性，对连续性变量默认等频分10个桶进行计算
        input:  
            trainset: pd.Series 数据列1
            testset: pd.Series 数据列2
            category_type:bool 是否为类别型变量
        output:
            psi: float psi值
        """
        if category_type == False:
            # 分桶
            quantiles = list(np.arange(0.1, 1, 0.1))
            cut_points = list(trainset.quantile(quantiles))
            cut_points = [-np.inf]+cut_points+[np.inf]
            trainset = pd.cut(trainset, bins=cut_points, duplicates='drop')
            testset = pd.cut(testset, bins=cut_points, duplicates='drop')
        else:
            # 合并样本数较少的类别
            trainset,_ = self._categary_merge(trainset,36,'OTHER')
            testset,_ = self._categary_merge(testset,36,'OTHER')

        # 计算每个类别样本数
        trainset_df = pd.DataFrame(trainset.value_counts())
        trainset_df.columns = ['trainset']
        testset_df = pd.DataFrame(testset.value_counts())
        testset_df.columns = ['testset']
        psi_df = pd.merge(trainset_df, testset_df, right_index=True,
                          left_index=True, how='outer').fillna(0)

        # 计算psi
        psi_df['trainset_rate'] = (
            psi_df['trainset']+1)/psi_df['trainset'].sum()        
        psi_df['testset_rate'] = (psi_df['testset']+1)/psi_df['testset'].sum()

        psi_df['psi'] = (psi_df['trainset_rate']-psi_df['testset_rate']) * \
            np.log(psi_df['trainset_rate']/psi_df['testset_rate'])
        psi = psi_df['psi'].sum()    
        return psi
        

    def psi_calculation(self, trainset, testset, cat_fea=[], num_fea=[]):
        """
        purpose:
            计算一个df指定字段的psi稳定性，对连续性变量默认等频分10个桶进行计算
            spi大于 0.25 说明极不稳定
        input:  
            trainset: pd.DataFrame 数据集1
            testset: pd.DataFrame 数据集2
            cat_fea: list 需要分析的类别型变量
            num_fea: list 需要分析的数值型变量
        output:
            psi: pd.DataFame [col,psi]
        """

        psi = []
        fea_all =  list(cat_fea)+list(num_fea)
        for col in fea_all:
            if col in cat_fea:
                tmp = self._psi_one_col(trainset[col], testset[col], category_type=True)
            else:
                tmp = self._psi_one_col(trainset[col], testset[col], category_type=False)
            psi.append(tmp)
        psi_df = pd.DataFrame({'col': fea_all, 'psi': psi})
        return psi_df

    def chi2_test(self, arr):
        '''
        purpose:
            计算卡方值
        input:
            arr:np.array 二维混淆矩阵（一般为2X2）
        output:
            x2:卡方值
            df: 自由度
            p: p值
        '''
        # R_N = np.array([arr.sum(axis=1)])
        # C_N = np.array([arr.sum(axis=0)])
        # N = arr.sum()
        # E = np.dot(R_N.T, C_N)/N  #理论混淆矩阵
        # square = (arr-E)**2/E
        # square[E == 0] = 0
        # x2 = square.sum() #卡方值
        # df = (arr.shape[0]-1)*(arr.shape[1]-1)  #自由度
        # p = chi2.cdf(x2, df=df)
        x2,p,df,_ = chi2_contingency(arr)  
        return x2,df,p

    def buckets_cut_unsupervised(self, col, method, bins_num=None, cut_points=None):
        """
        purpose:
            对某个字段进行分析分箱
        input:  
            col: pd.Series
            method: string 分箱方法 e.g {'等频':'qcut','等宽':'wcut','自定义':'cust'}
            bins_num: int 分箱数
            cut_point: list 自定义分割点
        output:
            col_new: pd.Series 分箱结果,按照0～n建立索引
            cut_points: list 分割点
        Notes:
        """
        # 无监督分箱方法
        if method == 'qcut':
            quantiles = list(np.arange(1/bins_num, 1, 1/bins_num))
            cut_points = list(set(col.quantile(quantiles)))
            cut_points = [-np.inf]+cut_points+[np.inf]
            cut_points.sort()
        elif method == 'wcut':
            step = (max(col)-min(col))/bins_num
            cut_points = list(np.arange(min(col)+step, max(col), step))
            cut_points = [-np.inf]+cut_points+[np.inf]
        elif method == 'cust':
            cut_points = list(set(cut_points))
            cut_points.sort()
        labels = list(range(len(cut_points)-1))
        col_new = pd.cut(col, bins=cut_points, labels=labels).astype('int64')
        return col_new, cut_points

    def buckets_cut_chimerge(self, col, target, category_type=False, bins_num=None, threshold=None):
        """
        purpose:
            对某个字段进行有监督分箱
        input:  
            col: pd.Series
            category_type: bool 是否离散型变量
            bins_num: int 最大分箱数
            threshole: float 卡方阈值，如果未指定bins_num,默认使用置信度95%设置threshold;threshold 越大，分箱数越少
            target: pd.Series 有监督分箱目标
        output:
            col_new: pd.Series 分箱结果
            cut_points: list 分割点（连续性变量）
            buckets_dict：dict 字典{'原始类别'：新类别}（离散型变量）
        Notes:
            如果bins_num存在，则threshold 不起作用，强制分成bins_num个箱
            如果bins_num不存在，根据threshold进行分箱
            如果bins_num 和 threshold 都不存在，则根据根据95%置信度设置threshold 进行分箱
        """
        # 连续性变量卡方分箱
        if category_type == False:
            # 分箱并计算每个箱中每一类的数量
            col_new, cut_points = self.buckets_cut_unsupervised(
                col, method='qcut', bins_num=100)
            freq = pd.crosstab(col_new, target).values

            # 根据置信度treshold计算卡方阈值
            cls_num = freq.shape[-1]
            if threshold is None:
                threshold = chi2.isf(0.05, df=cls_num-1)
            else:
                threshold = chi2.isf(1-threshold, df=cls_num-1)
            

            # 卡方分箱合并
            while len(freq) > 2:
                minvalue = None
                minidx = None

                #寻找卡方最小的两个区间    
                for i in range(len(freq)-1):
                    v,_,_ = self.chi2_test(freq[i:i+2])
                    if minvalue is None or v < minvalue:
                        minvalue = v
                        minidx = i

                # 当 bins_num 存在时判断当前桶数，当 bins_num 不存在则判断最小卡方是否小于阈值
                # 合并分桶
                if (bins_num is not None and bins_num < len(freq)) or (bins_num is None and minvalue < threshold):
                    tmp = freq[minidx]+freq[minidx+1]
                    freq[minidx] = tmp
                    freq = np.delete(freq, minidx+1, 0)
                    cut_points = cut_points[0:minidx+1]+cut_points[minidx+2:]
                else:
                    break

            col_new,cut_points = self.buckets_cut_unsupervised(
                col, method='cust', cut_points=cut_points)
            return col_new,cut_points

        # 离散变量卡方分箱
        else:
            # 分箱并计算每个箱中每一类的数量
            col_new,merge_cats = self._categary_merge(col,36,'OTHER')
            freq = pd.crosstab(col_new, target)
            tmp = list(freq.index)
            freq = freq.values
            buckets = []
            for i,cat in enumerate(tmp):
                if cat!='OTHER':
                    buckets.append([cat])
                else:
                    buckets.append(merge_cats)

            # 根据置信度treshold计算卡方阈值
            cls_num = freq.shape[-1]
            if threshold is None:
                threshold = chi2.isf(0.05, df=cls_num-1)
            else:
                threshold = chi2.isf(1-threshold, df=cls_num-1)
            
            # 卡方分箱合并
            while len(freq) > 2:
                minvalue = None
                minidx = None

                #寻找卡方最小的两个类别    
                for i in range(len(freq)-1):
                    for j in range(i+1,len(freq)):
                        v,_,_ = self.chi2_test(freq[[i,j]])
                        if minvalue is None or v < minvalue:
                            minvalue = v
                            minidx_1 = i
                            minidx_2 = j

                # 当 bins_num 存在时判断当前桶数，当 bins_num 不存在则判断最小卡方是否小于阈值
                # 合并分桶
                if (bins_num is not None and bins_num < len(freq)) or (bins_num is None and minvalue < threshold):
                    tmp = freq[minidx_1]+freq[minidx_2]
                    freq[minidx_1] = tmp
                    freq = np.delete(freq, minidx_2, 0)
                    buckets[minidx_1] = buckets[minidx_1]+buckets[minidx_2]
                    buckets = buckets[0:minidx_2]+buckets[minidx_2+1:]
                else:
                    break
            
            buckets_dict = {}
            for index,buc in enumerate(buckets):
                for j in buc:
                    buckets_dict[j] = index
            col_new = col.map(buckets_dict)            
            return col_new,buckets_dict
        


    def woe_encoding(self,col,target,pos_label):
        """
        purpose:
            对离散变量已分好桶的连续型变量进行woe编码
        input:  
            col: pd.Series
            target: pd.Series 目标
            pos_label: target中正样本标签
        output:
            col_new: pd.Series woe编码后的新列
            woe: dict woe映射关系
        Notes:
        """
        tmp = list(target.unique())
        tmp.remove(pos_label)
        neg_label = tmp[0]
        
        # 每个类别/分桶的正负样本数
        gb_i = pd.crosstab(col,target)+1

        # 全局正负样本数
        gb_pos = sum(target == pos_label)
        gb_neg = sum(target == neg_label)

        #woe计算 log（（g_i/b_i）/（g/b））
        gb_i['woe'] = np.log( ( gb_i[pos_label] / gb_i[neg_label] ) / (gb_pos / gb_neg) )
        woe = gb_i.woe.to_dict()
        col_new = col.copy()
        col_new = col_new.map(woe)
        return col_new, woe

    def iv_calculation(self,col,target,pos_label):
        """
        purpose:
            对离散变量/已分好桶的连续型变量计算iv值
        input:  
            col: pd.Series
            target: pd.Series 目标
            pos_label: target中正样本标签
        output:
            iv: float 该字段iv值
        Notes:
        """
        eps = 1
        tmp = list(target.unique())
        tmp.remove(pos_label)
        neg_label = tmp[0]


        # 合并样本数较少的类别
        col,_ = self._categary_merge(col,36,'OTHER')
        
        # 每个类别/分桶的正负样本数
        gb_i = pd.crosstab(col,target)+eps
        gb_i['num'] = gb_i.sum(axis=1)
    
        # 全局正负样本数
        gb_pos = sum(target == pos_label)+eps
        gb_neg = sum(target == neg_label)+eps

        #woe计算 log（（g_i/b_i）/（g/b））
        gb_i['iv'] = ( gb_i[neg_label] / gb_neg - gb_i[pos_label] / gb_pos )* \
                        np.log((gb_i[neg_label] / gb_i[pos_label]) / ( gb_neg / gb_pos ))
        return gb_i.iv.sum()

    def vif_calculation(self,df,subset):
        """
        purpose:
            计算某个字段的vif值
        input:  
            df: pd.DataFrame特征宽表 
            subset: list 目标字段列表
        output:
            vif: float vif值
        Notes:
        """        
        vif = []
        tmp = df.values
        for col in subset:
            idx = list(df.columns).index(col)
            vif.append(variance_inflation_factor(tmp,idx))
        return vif


    def vif_filer(self, df, threshold = 10):
        """
        purpose:
            根据VIF对特征过滤，降低共线性
        input:  
            df: pd.DataFrame 特征宽表 
            threshold: float vif阈值
        output:
            df_new: pd.DataFrame vif过滤之后的df
        Notes:
        """
        dropped = True
        df_new = df.copy()

        while dropped:
            dropped = False
            #找出vif最大的字段，如果vif超过阈值则直接删除
            vif_ls = self.vif_calculation(df = df_new,subset = df_new.columns)
            maxvif = max(vif_ls)
            maxidx = vif_ls.index(maxvif)
            maxcol = df_new.columns[maxidx]
            if maxvif>threshold:
                df_new = df_new.drop(maxcol,axis = 1)
                dropped = True
        return df_new

    def corr_filter(self, df,threhold,iv_dict,subset=None):
        """
        purpose:
            根据相关系数剔除变量
        input:  
            df: pd.DataFrame 特征宽表 
            threshold: float 相关系数阈值
            iv_dict: dict 变量iv字典
            subset: list 变量子集
        output:
            fea_tmp: list 过滤之后的变量列表
        Notes:
        """
        if subset is None:
            subset = list(df.columns)
        fea_tmp = subset.copy()

        corr_matrix = abs(df[fea_tmp].corr().values)
        for i in range(len(corr_matrix)):
            corr_matrix[i][i]=0
        index = np.where(corr_matrix>=threhold)


        # 如果某个变量与不止一个其他变量的相关系数大于阈值，则该变量暂时不删除
        print('开始剔除高相关性非耦合变量...')
        col_keep = []
        col_ls = list(index[0])
        for col in set(col_ls):
            col_keep.append(col) if col_ls.count(col)>1 else None  
        col_ls = list(index[1])
        for col in set(col_ls):
            col_keep.append(col) if col_ls.count(col)>1 else None
        col_keep = set(col_keep)

        col_remove = []
        for i in range(len(index[0])):
            col0 = fea_tmp[index[0][i]]
            col1 = fea_tmp[index[1][i]]
            if index[1][i] not in col_keep:
                col_remove.append(col1) if iv_dict[col0]>=iv_dict[col1] else col_remove.append(col0)
        col_remove = set(col_remove)

        # 删除相关系数大于阈值，并且没有耦合的column
        fea_tmp = [x for x in fea_tmp if x not in col_remove]

        # 从高到底剔除相关系数最大的columns，每次剔除后重新计算相关系数矩阵，防止过度删除变量
        print('开始剔除高相关性耦合变量...')
        while True:
            corr_matrix = abs(df[fea_tmp].corr().values)
            for i in range(len(corr_matrix)):
                corr_matrix[i][i]=0
            corr_max = np.max(corr_matrix)
            if corr_max<threhold:
                break
            index = np.where(corr_matrix==corr_max)
            print('%s feature left; max corration %s'%(len(fea_tmp),corr_max))
            if iv_dict[fea_tmp[index[0][0]]]>iv_dict[fea_tmp[index[0][1]]]:
                fea_tmp.remove(fea_tmp[index[0][1]])
            else:
                fea_tmp.remove(fea_tmp[index[0][0]])

        print('相关系数过滤完成...')
        return fea_tmp

    def ttest_1sample(sample,u):
        """
        purpose:
            单样本t检验
        input:  
            sample: list 采样样本
            u: 整体样本均值
        output:
            df_res: pd.DatFrame t检验结果
        Notes:
        """        
        lst=sample.copy()
        n=len(lst)
        s=np.std(lst)*(n**0.5)/(n-1)**0.5
        t=(np.mean(lst)-u)/(s/(n)**0.5)
        sig=2*stats.t.sf(abs(t),n-1)
        dic_res=[{'t值':t,'自由度':n-1,'Sig.':sig,'平均值差值':np.mean(lst)-u}]
        df_res=pd.DataFrame(dic_res,columns=['t值','自由度','Sig.','平均值差值'])
        return df_res

    

class AfterProcessing:
    def eval_binary(self,eval_list = [],legend = [],pos_label=1,is_sampled=False,real_pos_neg_ratio=[]):
        """
        purpose:
            对二分类问题的结果进行检验
        input:  
            eval_list: list with tuples like （y,y_hat） 一组或多组 （y,y_hat）的元组列表 
            legend: list with string每组列表的含义
            pos_label: 正样本值
            is_sampled: bool 样本集是否经过采样
            real_pos_neg_ratio: list 每个数据集真实正负样本比
        Notes:
        """
        plt.figure(figsize=(12,12))
        # ROC_curve
        plt.subplot(3,2,1)
        for i, eval_set in enumerate(eval_list):
            fpr,tpr,threhold = metrics.roc_curve(eval_set[0],eval_set[1],pos_label=pos_label)
            auc_score = metrics.auc(fpr,tpr)
            label = "{} (AUC = {:0.2f})".format(legend[i], auc_score)
            plt.plot(fpr,tpr,label = label)
        plt.legend(loc='lower right')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.grid()
        plt.title('ROC_curve')
        
        # KS
        plt.subplot(3,2,2)
        for i, eval_set in enumerate(eval_list):
            fpr,tpr,threhold = metrics.roc_curve(eval_set[0],eval_set[1],pos_label=pos_label)
            ks_score = max(tpr-fpr)
            label = "{} (KS = {:0.2f})".format(legend[i], ks_score)
            plt.plot(threhold,tpr-fpr,label = label)
        plt.legend()
        plt.xlabel('Threshold')
        plt.ylabel('tpr-fpr')
        #plt.grid()
        plt.title('KS_curve')
       
        # PR
        plt.subplot(3,2,3)
        for i,eval_set in enumerate(eval_list):
            precision,recall,threhold = metrics.precision_recall_curve(eval_set[0],eval_set[1],pos_label=pos_label)
            F1_score = max(2*precision*recall/(precision+recall))
            label = "{} (F1 = {:0.2f})".format(legend[i], F1_score)
            plt.plot(recall,precision,label = label)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        #plt.grid()
        plt.legend()
        plt.title('PR_curve')

        # Top-Percent
        plt.subplot(3,2,4)
        for i,eval_set in enumerate(eval_list):
            tmp = pd.DataFrame({'y':eval_set[0],'y_hat':eval_set[1]}).sort_values('y_hat',ascending=False)
            top = [(x+1)/100 for x in range(100)]
            cnt = len(tmp)
            precision = [sum(tmp.iloc[0:int(x*cnt),0]== pos_label)/int(x*cnt) for x in top]
            plt.plot(top,precision,label = legend[i])
        plt.xlabel('Top-Percent')
        plt.ylabel('Precision')
        #plt.grid()
        plt.legend()
        plt.title('Top-Percent')

        
        
        # 对采样前的P-R曲线进行评估
        if is_sampled:
            for i,eval_set in enumerate(eval_list):    
                plt.subplot(3,2,5)
                pos_num_sample = sum(eval_set[0]==pos_label)  #采样集正样本数量
                neg_num_sample = sum(eval_set[0]!=pos_label)  #采样集负样本数量
                #采样集pr曲线
                precision_pos,recall_pos,threhold = metrics.precision_recall_curve(eval_set[0],eval_set[1],pos_label=pos_label)
                #采样集对应负样本recall
                recall_neg = ((recall_pos * pos_num_sample) / precision_pos)*(1-precision_pos)/neg_num_sample
                
                precision_pos_real = recall_pos*real_pos_neg_ratio[i]/(recall_neg+recall_pos*real_pos_neg_ratio[i])
                F1_score = max(2*precision_pos_real*recall_pos/(precision_pos_real+recall_pos))
                label = "{} (F1 = {:0.2f})".format(legend[i], F1_score)

                plt.plot(recall_pos,precision_pos_real,label=label)

            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            #plt.grid()
            plt.legend()
            plt.title('PR_curve_real')

        plt.show()


    def eval_regression(self,eval_list = [],legend = []):
        """
        purpose:
            对回归问题的结果进行检验
        input:  
            eval_list: list of tuples like （y,y_hat） 一组或多组 （y,y_hat）的元组列表 
            legend: list of string 每组列表的含义 
        Notes:
        """
        plt.figure()
        for i,eval_set in enumerate(eval_list):
            #print(legend[i]+' corr: '+str(eval_set[0].corr(eval_set[1])))
            print(legend[i]+' r2: '+str(metrics.r2_score(eval_set[0],eval_set[1])))
            print(legend[i]+' mae: '+str(metrics.mean_absolute_error(eval_set[0],eval_set[1])))
            print(legend[i]+' mse: '+str(metrics.mean_squared_error(eval_set[0],eval_set[1])))
            plt.plot(eval_set[0],eval_set[1],'.',label = legend[i])
            plt.xlabel('y')
            plt.ylabel('y_hat')
            plt.grid()
            plt.legend()
            plt.show()        
    
    def cols_compare(self,col,label,category_type=False,bins = 50,x_label = None,y_label = None,ran=None,figsize=None,save_as=None,ifprint=True):
        """
        purpose:
            对比不同标签下特征的分布差异
        input:  
            col: pd.Series 特征
            label: pd.Series 标签
            category_type: bool 是否为类别型变量
            x_label: string x轴标签
            y_label: string y轴标签
            bins: int 分箱数
            merge: bool 是否显示在同一张图中
            ran: list/tuple 数据显示范围
            figsize: list 图像大小
            save_as: string 保存图片的地址
            ifprint: bool 是否打印
        Notes:
        """

        # 对于数值型特征，限制左右边界，去掉异常值的影响
        if (not category_type) and (ran is None):
            ran_left = []
            ran_right=  []
            for i,tag in enumerate(label.unique()):
                col_tmp = col.loc[label==tag]
                median = col.quantile(q = 0.5)
                p75 = col_tmp.quantile(q = 0.75)
                p25 = col_tmp.quantile(q = 0.25)
                IQR = p75-p25
                ran_left.append(max((p25-3*IQR),min(col)))
                ran_right.append(min((p75+3*IQR),max(col)))
            ran = (min(ran_left),max(ran_right))
        
        plt.figure(figsize=figsize)
        label_set = label.unique()
        label_set.sort()
        if not category_type:
            value_cnt = len(col.loc[(col>=ran[0]) & (col<=ran[1]) ].unique())
            #如果值个数小于bins
            if value_cnt<bins:
                for i,tag in enumerate(label_set):
                    col_tmp = col.loc[label==tag]
                    tmp = []
                    for i in col_tmp:
                        if i<ran[0]:
                            tmp.append(-np.inf)
                        elif i>ran[1]:
                            tmp.append(np.inf)
                        else:
                            tmp.append(i)
                    col_tmp = pd.Series(tmp)            
                    hist_counts = col_tmp.value_counts().sort_index()/len(col_tmp)
                    bar_nm = []
                    for i in hist_counts.index:
                        if i==np.inf:
                            bar_nm.append('(%s,inf)'%(ran[1]))
                        elif i==-np.inf:
                            bar_nm.append('(-inf,%s)'%(ran[0]))
                        else:
                            bar_nm.append(str(i))
                    plt.bar(bar_nm,hist_counts,alpha = 0.5,label = tag)
            else:
                for i,tag in enumerate(label_set):
                    col_tmp = col.loc[label==tag]
                    width = (ran[1]-ran[0])/(bins-1)
                    cut_points = [-np.inf]+[ran[0]+(x+1)*width for x in range(bins-1)]+[np.inf]
                    col_tmp = pd.cut(col_tmp, bins=cut_points)              
                    hist_counts = col_tmp.value_counts().sort_index()/len(col_tmp)
                    bar_nm = ['(%s,%s]'%(round(cut_points[i],2),round(cut_points[i+1],2)) for i in range(len(cut_points)-1)]
                    plt.bar(bar_nm,hist_counts,alpha = 0.5,label = tag)  
            plt.xticks(rotation=90)
            


        else:
            for i,tag in enumerate(label_set):
                col_tmp = col.loc[label==tag]
                tmp = col_tmp.groupby(col_tmp).count()/len(col_tmp)
                plt.bar(tmp.index,tmp,label = tag)
        
        plt.legend()
        plt.grid()
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.ylim(0,1)
        
        if save_as:
            plt.subplots_adjust(bottom=0.5)
            plt.savefig(save_as)
        if ifprint:
            plt.show()
        plt.close()

    def tree_export(self,decision_tree,max_depth,feature_names=None):
        """
        purpose:
            单棵决策树规则提取
        input：
            decision_tree: sklearn.tree.DecisionTreeClassifier 决策树模型
            max_depth: 规则提取深度
            feature_names: 特征名称
        output: 
            res:pd.DataFrame,{'路径规则'，'训练集样本','精度'，'召回率'}
        """
        
        from sklearn.tree import export_text
        import re

        tree = export_text(decision_tree=decision_tree,
                        max_depth=decision_tree.max_depth,
                        feature_names=feature_names,
                        show_weights=True,
                        decimals=2).split('\n')
        tree = [x for x in tree if len(x)>0]
        
        def getrules(tree,con_list,con_cur,depth_cur,max_depth):
            if len(tree)==1:
                samplecnt = eval(re.findall(r"\[.+?\]",tree[0][4:])[0])
                con_list.append((con_cur,samplecnt))
                return
            elif depth_cur==max_depth:
                leafs = [x for x in tree if 'weights' in x]
                leafs = [eval(re.findall(r"\[.+?\]",x)[0]) for x in leafs]
                samplecnt = list(sum(np.array(leafs)))
                con_list.append((con_cur,samplecnt))
                return
            
            root_index = [i for i in range(len(tree)) if tree[i][0:4]=='|---']
            root_index.append(len(tree))
            
            for i in range(len(root_index)-1):
                subtree = [row[4:] for row in tree[root_index[i]+1:root_index[i+1]]]
                getrules(subtree,con_list,con_cur+('|'+tree[root_index[i]][4:]),depth_cur+1,max_depth)

        con_list = []
        con_cur = ''
        depth_cur = 0
        getrules(tree,con_list,con_cur,depth_cur,max_depth)

        res = pd.DataFrame({'condition':[x[0] for x in con_list],
                            'samplecnt':[x[1] for x in con_list]})
        res['precision'] = res.samplecnt.map(lambda x: list(np.array(x)/sum(x)))
        sample_sum = sum(np.array(list(res.samplecnt)))
        res['recall'] = res.samplecnt.map(lambda x: list(x/sample_sum))
        return res

    def forest_export(self,random_forest,max_depth,feature_names=None):
        """
        purpose:
            随机森林规则提取
        input：
            random_forest: sklearn.ensemble.RandomForestClassifier 决策树模型
            max_depth: 规则提取深度
            feature_names: 特征名称
        output: 
            res:pd.DataFrame,{'路径规则'，'训练集样本','精度'，'召回率'}
        """
        for i,decision_tree in enumerate(random_forest.estimators_):
            if i==0:
                res = self.tree_export(decision_tree=decision_tree,max_depth=max_depth,feature_names=feature_names)
            else:
                res = pd.concat([res,self.tree_export(decision_tree=decision_tree,max_depth=max_depth,feature_names=feature_names)])
        
        res = res.drop_duplicates(subset = 'condition')

        return res


    def tree_print(self,decision_tree,max_depth,feature_names=None):
        """
        purpose:
            单棵决策树结构打印
        input：
            decision_tree: sklearn.tree.DecisionTreeClassifier 决策树模型
            max_depth: 规则提取深度
            feature_names: 特征名称
        output: 
            tree_struct:string,树结构
        """
        from sklearn.tree import export_text
        import re

        tree = export_text(decision_tree=decision_tree,
                        max_depth=decision_tree.max_depth,
                        feature_names=feature_names,
                        show_weights=True,
                        decimals=2).split('\n')
        tree = [x for x in tree if len(x)>0]


        def _tree_print(tree,judge,depth_cur,max_depth):
            if len(tree)==1:  #如果到达叶节点
                leaf = str(list(np.array(eval(re.findall(r"\[.+?\]",tree[0][4:])[0].replace(' ',''))).astype(np.int)))
                lineformat = "{:^%s}"%(len(leaf))
                line = lineformat.format('|')
                return [line,line,lineformat.format(judge),leaf]
            elif depth_cur==max_depth:  #如果达到最大深度
                # 结算子树所有叶节点样本总和
                leafs = [x for x in tree if 'weights' in x]
                leafs = [eval(re.findall(r"\[.+?\]",x)[0]) for x in leafs]
                leaf = str(list(sum(np.array(leafs)).astype(np.int)))
                lineformat = "{:^%s}"%(len(leaf))
                line = lineformat.format('|')
                return [line,line,lineformat.format(judge),leaf]
            
            #抓取根节点索引
            root_index = [i for i in range(len(tree)) if tree[i][0:4]=='|---']
            root_index.append(len(tree))
            
            #获取根节点
            root = '{'+tree[root_index[0]][4:].replace(' ','')+'}'
            
            #左树
            subtree_left = [row[4:] for row in tree[root_index[0]+1:root_index[1]]]
            subtopo_left = _tree_print(subtree_left,'YES',depth_cur+1,max_depth=max_depth)

            #右树
            subtree_right = [row[4:] for row in tree[root_index[1]+1:root_index[2]]]
            subtopo_right= _tree_print(subtree_right,'NO',depth_cur+1,max_depth=max_depth)

            #左右子树合并
            subtopo = []
            diff = len(subtopo_left)-len(subtopo_right)
            if diff<0:
                tmp = max([len(x) for x in subtopo_left])
                subtopo_left = subtopo_left+[' '*tmp]*(-diff)
            elif diff>0:
                tmp = max([len(x) for x in subtopo_right])
                subtopo_right = subtopo_right+[' '*tmp]*(diff)            
            for i in range(len(subtopo_left)):
                subtopo.append(subtopo_left[i]+' '+subtopo_right[i])

            samplecnt = str(list(sum(np.array([eval(x) for x in re.findall("\[.+?\]",subtopo[3])])))) # 根节点样本数
            L = max([len(subtopo[0].strip()),len(root),len(samplecnt)])
            
            nodeformat = "{:-^%s}"%(L)
            lineformat = "{:^%s}"%(L)

            gap = (subtopo[0].find('|')-(L-len(subtopo[0].strip()))//2)*' '
            line =  gap+lineformat.format('|')
            judge = gap+lineformat.format(judge)
            samplecnt = gap+lineformat.format(samplecnt)
            root = gap+nodeformat.format(root)
            res = [line,line,judge,samplecnt,line,line,root]+subtopo
            L = max([len(x) for x in res])
            tmp = "{:<%s}"%(L)
            res = [tmp.format(x) for x in res]
            return res

        tree_struct = '\n'.join(_tree_print(tree,judge='|',depth_cur=0,max_depth=max_depth))
        tree_struct = tree_struct.replace('[','(').replace(']',')')
        return tree_struct

    ############################
    #通过解析pmml文件方式提取规则路径
    ############################
    # def PmmlTreeDraw(self,node_list,con,con_list):
    #     """
    #     purpose:
    #         pmml单棵决策树规则提取
    #     input：
    #         node_list: 某一层的节点列表
    #         con：通往该层的前置条件
    #         con_list: 路径存储列表
    #     """
    #     #如果没有节点
    #     if len(node_list)==0:
    #         con_list.append(con)
    #     #如果存在节点
    #     else:
    #         #取出第一个节点
    #         node_0 = node_list[0] 
    #         child_nodes = [k for k in node_0.childNodes if k.localName=='Node']
    #         con_0 = ''
    #         if [k for k in node_0.childNodes if k.localName=='SimplePredicate']!=[]:
    #             con_0=node_list[0].getElementsByTagName('SimplePredicate')[0].getAttribute('field')+' '+ \
    #                         node_list[0].getElementsByTagName('SimplePredicate')[0].getAttribute('operator') + ' ' +\
    #                         node_list[0].getElementsByTagName('SimplePredicate')[0].getAttribute('value')
    #             con_1=node_list[0].getElementsByTagName('SimplePredicate')[0].getAttribute('field')+' '+ \
    #                         'not '+\
    #                         node_list[0].getElementsByTagName('SimplePredicate')[0].getAttribute('operator') + ' ' +\
    #                         node_list[0].getElementsByTagName('SimplePredicate')[0].getAttribute('value')
    #         else:
    #             con_0 = ''
    #         # 判断该节点是否是叶节点
    #         if child_nodes==[]:
    #             #计算样本分布
    #             cnt = {}
    #             for i in [k for k in node_0.childNodes if k.localName=='ScoreDistribution']:
    #                 cnt[i.getAttribute('value')] = i.getAttribute('recordCount')
    #             con_0 = con_0+'|'+str(cnt)
    #         self.PmmlTreeDraw([],con+'|'+con_0,con_list)
    #         # 处理剩余节点
    #         if len(node_list)>1:
    #             self.PmmlTreeDraw(node_list[1:],con+'|'+con_1,con_list)

    # def PmmlRandomForestDraw(self,pmml_file):
    #     """
    #     purpose:
    #         pmml随机森林规则提取
    #     input：
    #         pmml_file: string 随机森林pmml文件路径
    #     output:
    #         res: pd.DataFrame 叶节点的样本数与通往该叶节点的规则路径
    #     """
    #     import xml.dom.minidom as xmldom
    #     # 一个segment对应一棵树
    #     domobj=xmldom.parse(pmml_file)
    #     elementobj=domobj.documentElement
    #     segments=elementobj.getElementsByTagName('Segment')

    #     con_list = []
    #     for tree in  segments:
    #         con_list_tmp = []
    #         #获取第一层节点列表
    #         node_list = [k for k in tree.getElementsByTagName('Node')[0].childNodes if k.localName=='Node' ]
    #         self.PmmlTreeDraw(node_list,'',con_list_tmp)
    #         con_list = con_list+con_list_tmp

    #     # pmml 中判断关键字
    #     operator = {
    #     'not lessOrEqual':'>',
    #     'not greaterThan':'<=',
    #     'not equal':'!=',
    #     'lessOrEqual':'<=',
    #     'greaterThan':'>',
    #     'equal':'=='
    #     }

    #     # 决策路径整理
    #     condition = ['|'.join(con.split('|')[0:-1]) for con in con_list]
    #     res = pd.DataFrame([eval(con.split('|')[-1]) for con in con_list ])
    #     res['condition'] = condition
    #     res = res.drop_duplicates(subset='condition')
    #     for key in operator.keys():
    #         res['condition'] = res['condition'].str.replace(key,operator[key])
    #     return res


class Modelset:
    def __init__(self,config_file,type):
        """
        purpose:
            建立通用模型训练代码，快速构建demo
        input:
            config_file: string 模型参数文件
            type: string 模型类型，分类or回归 ('classifier','regressor')
        """
        self.parameters = self._paramread(config_file)
        self.type = type
        if type == 'classifier':        
            self.model_dict = self._classifierdefine()
        elif type == 'regressor':
            self.model_dict = self._regressordefine()


    def _paramread(self,config_file):
        """
        purpose:
            读取参数文件
        input:
            config_file: string 参数文件路径
        output:
            parameters:dict 参数输出
        """
        with open(config_file, encoding="utf-8") as f:
            tmp = f.readlines()
            tmp = [x.split("//")[0] for x in tmp]
            parameters = json.loads(''.join(tmp))
        return parameters


    def _classifierdefine(self):
        """
        purpose:
            根据配置文件定义分类模型对象
        input:
        output:
            model_dict: dict 模型对象字典
        """
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        from sklearn.ensemble import RandomForestClassifier
        model_dict = {}
        model_dict['LogisticRegression'] = LogisticRegression(**self.parameters['LogisticRegression']['model_para'])
        model_dict['XGBClassifier'] = XGBClassifier(**self.parameters['XGBClassifier']['model_para'])
        model_dict['RandomForestClassifier'] = RandomForestClassifier(**self.parameters['RandomForestClassifier']['model_para'])
        return model_dict


    def _regressordefine(self):
        """
        purpose:
            根据配置文件定义回归模型对象
        input:
        output:
            model_dict: dict 模型对象字典
        """
        from sklearn.linear_model import Lasso,Ridge
        from xgboost import XGBRegressor
        from sklearn.ensemble import RandomForestRegressor
        model_dict = {}    
        model_dict['Lasso'] = Lasso(**self.parameters['Lasso']['model_para'])
        model_dict['Ridge'] = Ridge(**self.parameters['Ridge']['model_para'])
        model_dict['XGBRegressor'] = XGBRegressor(**self.parameters['XGBRegressor']['model_para'])
        model_dict['RandomForestRegressor'] = RandomForestRegressor(**self.parameters['RandomForestRegressor']['model_para'])
        return model_dict


    def fit(self,model_nm,X_train,y_train,eval_set = []):
        """
        purpose:
            模型训练
        input:
            model_nm: string 模型名字
            X_train: pd.DataFrame 用于模型训练的特征
            y_train: pd.Series 模型训练标签
            eval_set: list of tuples like (y,y_hat) 验证数据集
        output:
            
        """
        model = self.model_dict[model_nm]
        print('start %s fitting ....'%(model_nm))


        if self.type == 'classifier':
            if model_nm in ('LogisticRegression','RandomForestClassifier'):        
                model.fit(X = X_train,y=y_train,**self.parameters[model_nm]['fit_para'])
            elif  model_nm =='XGBClassifier':
                model.fit(X = X_train,y=y_train,eval_set=eval_set,**self.parameters[model_nm]['fit_para'])
            print('%s fitting finished '%(model_nm))
        elif self.type == 'regressor':
            if model_nm in ('Lasso', 'Ridge', 'RandomForestRegressor'):        
                model.fit(X = X_train,y=y_train,**self.parameters[model_nm]['fit_para'])
            elif  model_nm =='XGBRegressor':
                model.fit(X = X_train,y=y_train,eval_set=eval_set,**self.parameters[model_nm]['fit_para'])
            print('%s fitting finished '%(model_nm))


    def grid_search(self,model_nm,X_train,y_train,parameters,cv):
        """
        purpose:
            模型参数网格搜索
        input:
            model_nm: string 模型名字
            X_train: pd.DataFrame 用于模型训练的特征
            y_train: pd.Series 模型训练标签
            parameters: dict of parameter list 用于网格搜索的参数列表
            cv: int k-fold number
        output:
            gs:GridSearchCV 网格搜索后重新fit的最优模型
        """
        model = self.model_dict[model_nm]    
        print('start %s grid searching ....'%(model_nm))
        gs = GridSearchCV(model,parameters,cv=cv)
        gs.fit(X_train,y_train)
        print('%s grid searching finished '%(model_nm))
        print('best params are %s:' %(gs.best_params_))
        self.model_dict[model_nm] = gs.best_estimator_
        print('model_dict is updated')
        return gs
    

    def predict(self,model_nm,X):
        """
        purpose:
            模型预测
        input:
            model_nm: string 模型名字
            X: pd.DataFrame/np.array 特征
        output:
            result: np.array 模型预测结果
        """
        model = self.model_dict[model_nm]
        if self.type=='classifier':
            result = model.predict_proba(X)[:,1]
            return result
        elif self.type=='regressor':
            result = model.predict(X)
            return result



class BetterTreeClassifier:
    """
    每次选取最优路径，对剩余路径下的样本重新建模，通过多次迭代生成分类树模型
    input:
        max_depth:int 树深度
        fit_rounds:int 迭代次数
    """
    def __init__(self,max_depth,fit_rounds):
        from sklearn.tree import DecisionTreeClassifier
        self.max_depth = max_depth
        self.fit_rounds = fit_rounds
        self.tree = DecisionTreeClassifier(max_depth=max_depth)

    def _getrules(self,X,y):
        self.tree.fit(X,y)

        aftmethod = AfterProcessing()
        res = aftmethod.tree_export(self.tree,max_depth=self.tree.max_depth)
        
        return res[['condition','samplecnt','precision','recall']]

    def _exec_rule(self,rule,X):
        X.columns = ['feature_'+str(i) for i in range(X.shape[1])]
        tmp = '&'.join(['(X.%s)'%(x.replace(' ','')) for x in rule.split('|') if len(x)>0])
        return eval(tmp)

    def fit(self,X,y):    
        rule_final = []
        precision_final = []
        samplecnt_final = []
        data_tmp = pd.concat([X,y],axis=1)
        for i in range(self.fit_rounds):    
            X_tmp = data_tmp.iloc[:,0:-1]
            y_tmp = pd.Series(list(data_tmp.iloc[:,-1]))
            res = self._getrules(X_tmp,y_tmp)
            res['pos_score'] = res.precision.map(lambda x:x[1])
            res = res.sort_values('pos_score',ascending=False)
            rule_final.append(res.iloc[0,0])
            samplecnt_final.append(res.iloc[0,1])
            precision_final.append(res.iloc[0,2])
            data_tmp = data_tmp[(~self._exec_rule(res.iloc[0,0],data_tmp))]

        rule_final = rule_final+list(res.iloc[1:,0])
        samplecnt_final = samplecnt_final+list(res.iloc[1:,1])
        precision_final = precision_final+list(res.iloc[1:,2])
        self.rules = pd.DataFrame({'condition':rule_final,'samplecnt':samplecnt_final,'precision':precision_final})

        tmp  = sum(np.array(list(self.rules.samplecnt)))
        self.rules['recall'] = self.rules.samplecnt.map(lambda x: list(np.array(x)/tmp))

    def _predict_row(self,row):
        i=0
        while self._exec_rule(self.rules.iloc[i,0],row)==False:
            i+=1
        return self.rules.iloc[i,1]

    def predict(self,X):
        res = np.array([[-1.0,-1.0]]*len(X))
        for i in self.rules.index:
            rule = self.rules.loc[i,'condition']
            score = self.rules.loc[i,'precision']
            res[(res[:,0]==-1)&self._exec_rule(rule,X),:]=score
        return res