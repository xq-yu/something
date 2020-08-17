import pandas as pd
import numpy as np
class FeatureEngeering:


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
        missing_df = missing_df.rename(
            columns={'index': 'col', 0: 'missing_pct'})
        missing_df = missing_df.sort_values(
            'missing_pct', ascending=False).reset_index(drop=True)
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
        col_drop = list(
            missing_df.loc[missing_df.missing_pct >= threshold, 'col'])
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
        var.columns = ['col', 'std']
        col_drop = list(var.loc[var.std <= threshold, 'col'])
        df_new = df_new.drop(col_drop, axis=1)
        return df_new

    def datatype_clean(self, df):
        """
        purpose:
            对数据类型进行规整，将可以转化为数值型的字段转换为数值类型
        input:
            df:数据集
        output:
            df_new:转换后的数据集
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
            df:数据集
            threshold: 取值小于该阈值的字段为类别型变量，大于该值为连续型变量
            subset:考虑的部分字段，默认为None
        output:
            feature_cat: 类别型变量（取值数量小于等于阈值的字段）
            feature_num: 数值型变量，类别为数值型，且取值个数大于阈值
            feature_res: 存在矛盾的变量或其他类型字段，需进一步判断别（1.非object类型字段和数值型字段；2.object型字段但是取值数量大于阈值）
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

    def __PsiOneCol__(self, trainset, testset, category_type=False):
        """
        purpose:
            计算一个变量的psi稳定性，对连续性变量默认等频分10个桶进行计算
        input:  
            trainset: pandas Series 数据列1
            testset: pandas Series 数据列2
            category_type:是否为类别型变量
        output:
            psi: psi值
        """
        import numpy as np
        import pandas as pd
        if category_type == False:
            # 分桶
            quantiles = list(np.arange(0.1, 1, 0.1))
            cut_points = list(trainset.quantile(quantiles))
            cut_points = [-np.inf]+cut_points+[np.inf]
            trainset = pd.cut(trainset, bins=cut_points, duplicates='drop')
            testset = pd.cut(testset, bins=cut_points, duplicates='drop')

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

    def PsiCalculation(self, trainset, testset, cat_fea=[], num_fea=[]):
        """
        purpose:
            计算一个df指定字段的psi稳定性，对连续性变量默认等频分10个桶进行计算
            spi大于 0.25 说明极不稳定
        input:  
            trainset: pandas df 数据集1
            testset: pandas df 数据集2
            cat_fea: 需要分析的类别型变量
            num_fea: 需要分析的数值型变量
        output:
            psi: pd df
        """
        import pandas as pd
        psi = []
        for col in cat_fea:
            psi.append(self.__PsiOneCol__(
                trainset[col], testset[col], category_type=True))
        for col in num_fea:
            psi.append(self.__PsiOneCol__(
                trainset[col], testset[col], category_type=False))

        psi_df = pd.DataFrame({'columns': cat_fea+num_fea, 'psi': psi})
        return psi_df

    def __chi3__(self, arr):
        '''
        purpose:
            计算卡方值
        input:
            arr:二维混淆矩阵（2*N）
        '''
        import numpy as np
        R_N = np.array([arr.sum(axis=1)])
        C_N = np.array([arr.sum(axis=0)])
        N = arr.sum()
        E = np.dot(R_N.T, C_N)/N
        square = (arr-E)**2/E
        square[E == 0] = 0
        v = square.sum()
        return v

    def BucketsCutUnsupervised(self, col, method, bins_num=None, cut_points=None):
        """
        purpose:
            对某个字段进行分析分箱
        input:  
            col: pandas series
            method: 分箱方法 e.g {'等频':'qcut','等宽':'wcut','自定义':'cust'}
            bins_num: 分箱数
            cut_point: 自定义分割点
        output:
            col_new: pandas Series 分箱结果,按照0～n建立索引
            cut_points: list，分割点
        Notes:
        """
        import numpy as np
        import pandas as pd
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
        else:
            cut_points = list(set(cut_points))
            cut_points.sort()
        labels = list(range(len(cut_points)-1))
        col_new = pd.cut(col, bins=cut_points, labels=labels)
        return col_new, cut_points

    def BucketsCutSupervised(self, col, target, method, bins_num=None, threshold=None):
        """
        purpose:
            对某个字段进行有监督分箱
        input:  
            col: pandas series
            method: 分箱方法 e.g {'卡方分箱':'ChiMerge'}
            bins_num: 最大分箱数
            threshole: 卡方阈值，如果未指定bins_num,默认使用置信度95%设置threshold
            target: 有监督分箱目标
        output:
            col_new: pandas Series 分箱结果
            cut_points: list，分割点
        Notes:
            如果bins_num存在，则threshold 不起作用，强制分成bins_num个箱
            如果bins_num不存在，根据threshold进行分箱
            如果bins_num 和 threshold 都不存在，则根据根据95%置信度设置threshold 进行分箱
        """
        from scipy.stats import chi2
        import numpy as np
        import pandas as pd
        if method == 'ChiMerge':
            # 分箱并计算每个箱中每一类的数量
            col_new, cut_points = self.BucketsCutUnsupervised(
                col, method='qcut', bins_num=100)
            freq = pd.crosstab(col_new, target).values

            # 95%置信度trehold
            if bins_num is None and threshold is None:
                cls_num = freq.shape[-1]
                threshold = chi2.isf(0.05, df=cls_num-1)

            # 卡方分箱合并
            while len(freq) > 2:
                minvalue = None
                minidx = None

                #寻找卡方最小的两个区间    
                for i in range(len(freq)-1):
                    v = self.__chi3__(freq[i:i+2])
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

            col_new = self.BucketsCutUnsupervised(
                col, method='cust', cut_points=cut_points)
        return col_new,cut_points


    def WoeEncoding(self,col,target,pos_label):
        """
        purpose:
            对离散变量已分好桶的连续型变量进行woe编码
        input:  
            col: pandas series
            target: 目标
            pos_label: target中正样本标签
        output:
            col_new: woe编码后的新列
            woe: woe映射关系
            iv: 该字段iv值
        Notes:
        """
        import pandas as pd
        import numpy as np
        eps = 0.00001
        tmp = list(target.unique())
        tmp.remove(pos_label)
        neg_label = tmp[0]
        
        # 每个类别/分桶的正负样本数
        gb_i = pd.crosstab(col,target)+eps

        # 全局正负样本数
        gb_pos = sum(target == pos_label)
        gb_neg = sum(target == neg_label)

        #woe计算 log（（g_i/b_i）/（g/b））
        gb_i['woe'] = np.log( ( gb_i[pos_label] / gb_i[neg_label] ) / (gb_pos / gb_neg) )
        woe = gb_i.woe.to_dict()
        col_new = col.copy()
        col_new = col_new.map(woe)
        return col_new, woe

    def IVCalculation(self,col,target,pos_label):
        """
        purpose:
            对离散变量已分好桶的连续型变量计算iv值
        input:  
            col: pandas series
            target: 目标
            pos_label: target中正样本标签
        output:
            iv: 该字段iv值
        Notes:
        """
        import pandas as pd
        import numpy as np
        eps = 0.00001
        tmp = list(target.unique())
        tmp.remove(pos_label)
        neg_label = tmp[0]
        
        # 每个类别/分桶的正负样本数
        gb_i = pd.crosstab(col,target)+eps

        # 全局正负样本数
        gb_pos = sum(target == pos_label)+eps
        gb_neg = sum(target == neg_label)+eps

        #woe计算 log（（g_i/b_i）/（g/b））
        gb_i['iv'] = ( gb_i[neg_label] / gb_neg - gb_i[pos_label] / gb_pos )* \
                        np.log((gb_i[neg_label] / gb_i[pos_label]) / ( gb_neg / gb_pos ))
        return gb_i.iv.sum()

    def VIF_calculation(self,df,col):
        """
        purpose:
            计算某个字段的vif值
        input:  
            df: 特征宽表 
            col: 目标字段
        output:
            vif: vif值
        Notes:
        """        
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        idx = list(df.columns).index(col)
        vif = variance_inflation_factor(df.values,idx)
        return vif


    def VIF_filer(self, df, threshold = 10):
        """
        purpose:
            根据VIF对特征过滤，降低共线性
        input:  
            df: 特征宽表 
            threshold: vif阈值
        output:
            df_new: vif过滤之后的df
        Notes:
        """
        dropped = True
        df_new = df.copy()

        while dropped:
            dropped = False
            #找出vif最大的字段，如果vif超过阈值则直接删除
            vif_ls = [self.VIF_calculation(df_new,col) for col in df_new.columns]
            maxvif = max(vif_ls)
            maxidx = vif_ls.index(maxvif)
            maxcol = df_new.columns[maxidx]
            if maxvif>threshold:
                df_new = df_new.drop(maxcol,axis = 1)
                dropped = True
        return df_new

