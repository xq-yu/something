import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import json
class preprocessing:
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


	def _CategaryMerge_(self,col,threshold,tag):
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
		
		
	def __PsiOneCol__(self, trainset, testset, category_type=False):
		"""
		purpose:
			计算一个变量的psi稳定性，对连续性变量默认等频分10个桶进行计算
		input:  
			trainset: pd.Series 数据列1
			testset: pd.Series 数据列2
			category_type:bool 是否为类别型变量
		output:
			psi: flaot psi值
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
			trainset,_ = self._CategaryMerge_(trainset,36,'OTHER')
			testset,_ = self._CategaryMerge_(testset,36,'OTHER')

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
			trainset: pd.DataFrame 数据集1
			testset: pd.DataFrame 数据集2
			cat_fea: list 需要分析的类别型变量
			num_fea: list 需要分析的数值型变量
		output:
			psi: pd.DataFame   [columns,psi]
		"""
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
			arr:np.array 二维混淆矩阵（2*N）
		'''
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

	def BucketsCutChiMerge(self, col, target, category_type=False, bins_num=None, threshold=None):
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
			col_new, cut_points = self.BucketsCutUnsupervised(
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

			col_new,cut_points = self.BucketsCutUnsupervised(
				col, method='cust', cut_points=cut_points)
			return col_new,cut_points

		# 离散变量卡方分箱
		else:
			# 分箱并计算每个箱中每一类的数量
			col_new,merge_cats = self._CategaryMerge_(col,36,'OTHER')
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
						v = self.__chi3__(freq[[i,j]])
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
		


	def WoeEncoding(self,col,target,pos_label):
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
			对离散变量/已分好桶的连续型变量计算iv值
		input:  
			col: pd.Series
			target: pd.Series 目标
			pos_label: target中正样本标签
		output:
			iv: float 该字段iv值
		Notes:
		"""
		eps = 0.00001
		tmp = list(target.unique())
		tmp.remove(pos_label)
		neg_label = tmp[0]


		# 合并样本数较少的类别
		col,_ = self._CategaryMerge_(col,36,'OTHER')
		
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

	def VIF_calculation(self,df,subset):
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


	def VIF_filer(self, df, threshold = 10):
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
			vif_ls = self.VIF_calculation(df = df_new,subset = df_new.columns)
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
			col1 = fea_tmp[index[0][i]]
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


class afterprocessing:
	def eval_binary(self,eval_list = [],legend = [],pos_label=1):
		"""
		purpose:
			对二分类问题的结果进行检验
		input:  
			eval_list: list with tuples like （y,y_hat） 一组或多组 （y,y_hat）的元组列表 
			legend: list with string每组列表的含义 
		Notes:
		"""
		plt.figure(figsize=(12,12))
		# ROC_curve
		plt.subplot(2,2,1)
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
		plt.subplot(2,2,2)
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
		plt.subplot(2,2,3)
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
		plt.subplot(2,2,4)
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
			print(legend[i]+' corr: '+str(eval_set[0].corr(eval_set[1])))
			print(legend[i]+' r2: '+str(metrics.r2_score(eval_set[0],eval_set[1])))
			print(legend[i]+' mae: '+str(metrics.mean_absolute_error(eval_set[0],eval_set[1])))
			print(legend[i]+' mse: '+str(metrics.mean_squared_error(eval_set[0],eval_set[1])))
			plt.plot(eval_set[0],eval_set[1],label = legend[i])
			plt.xlabel('y')
			plt.ylabel('y_hat')
			plt.grid()
			plt.legend()
			plt.show()		
	
	def cols_compare(self,col,label,x_label = None,y_label = None, bins = 50,merge=True,ran=None):
		"""
		purpose:
			对比不同标签下特征的分布差异
		input:  
			col: pd.Series 特征
			label: pd.Series 标签
			x_label: string x轴标签
			y_label: string y轴标签
			bins: int 分箱数
			merge: bool 是否显示在同一张图中
			ran：list/tuple 数据显示范围
		Notes:
		"""
		if ran is None:
			median = col.quantile(q = 0.5)
			p75 = col.quantile(q = 0.75)
			p25 = col.quantile(q = 0.25)
			IQR = p75-p25
			ran_left = max((p25-3*IQR),min(col))
			ran_right = min((p75+3*IQR),max(col))
			ran = (ran_left,ran_right)
		
		tmp = (ran[1]-ran[0])/bins
		cutpoint = [ran[0]+x*tmp for x in range(bins+1)]
		col = col.loc[(col>=ran[0]) & (col<=ran[1])]
		if merge:
			for tag in label.unique():
				sns.distplot(col.loc[label==tag],bins = cutpoint,kde_kws={'label':tag})
			plt.legend()
			plt.xlabel(x_label) 
			plt.ylabel(y_label)
			plt.show()
		else:
			for i,tag in enumerate(label.unique()):
				plt.subplot(len(label.unique()),1,i+1)
				plt.hist(col.loc[label==tag],range = ran, density = False,bins = bins,alpha = 0.5,label = tag)
				plt.legend()
				plt.ylabel(y_label)
			plt.xlabel(x_label) 
			plt.show()
	
	
class ModelClassifier:
	def __init__(self,config_file):
		"""
		purpose:
			建立通用模型训练代码，快速构建demo
		input:
			config_file: string 模型参数文件
		"""
		self.parameters = self._para_read_(config_file)
		self.model_dict = self._model_define_()

	def _para_read_(self,config_file):
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

	def _model_define_(self):
		"""
		purpose:
			根据配置文件定义模型对象
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
		if model_nm in ('LogisticRegression','RandomForestClassifier'):		
			model.fit(X = X_train,y=y_train,**self.parameters[model_nm]['fit_para'])
		elif  model_nm =='XGBClassifier':
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
	
	def predict_prob(self,model_nm,X):
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
		result = model.predict_proba(X)[:,1]
		return result

	
