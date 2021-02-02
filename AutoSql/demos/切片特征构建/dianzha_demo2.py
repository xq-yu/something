import AutoSql
import importlib
importlib.reload(AutoSql)



#%% 
############################################################
#
# 1.基础流水表参数
#
############################################################
basic_param = {
        'database': 'bi_ysfdb',  #数据库名
        'table': ('yuanchengwithlabel','远程转账交易自相交表'),  #表名
        'entitys':[('prikey','string','交易唯一识别号')],  #实体字段
        
        #维度字段
        'dimensions':[
                         ('tsdate','int','本次交易日期yyyymmdd')
                        ,('tsyyyymm','int',"本次交易月份 年月")
                        ,('tsmon','int',"本次交易月份 [1,12]",)
                        ,('tsday','int',"本次交易日期 [1,31]",)
                        ,('tshour','int',"本次交易时间小时 [0,23]")
                        ,('timestamp_s','int',"本次交易uninx 时间戳，单位为秒" )
                        ,('transamt','double','本次交易金额 ')
                        ,('outbankcard','string','本次转出卡 ')
                        ,('inbankcard','string','本次转入卡 ')
                        ,('inbankcardattr','string','转入卡性质')
                        ,('usrid','string','转出用户id')
                        ,('histransamt','double','历史交易金额 ')
                        ,('histimestamp_s','int','历史交易uninx 时间戳 ')
                        ,('hisoutbankcard','string','历史交易付款卡')
                        ,('hisinbankcard','string','历史交易收款卡')
                    ],       
        #窗口相对位置标记字段
        'winpos_col':[('tsdaydiff','int',"历史交易相对时间")],
        #窗口名称
        'win_nm':['win1'],

        'extra_col':[],  #其他字段
        'partitioned_by':[],  #分区字段
        'clustered_by':[],   #分桶字段
        'bucket_num':135,    # 分桶数量
        'field_sep':',',    # 字段分隔符
        'stored_as':'orc',   #表格式
        'location':'hdfs://master:9000/user/hive/warehouse/mydatabase.db/' #表路径
}




#%%
############################################################
#
# 2.函数/组合函数别名
#
############################################################
fun_dict = {
                'sum':      'sum              (  COL1  ) ',
                'avg':      'avg              (  COL1  ) ',
                'max':      'max              (  COL1  ) ',
                'mid':      'percentile_approx(  COL1  , 0.5) ',
                'min':      'min              (  COL1  ) ',
                'std':      'std              (  COL1  ) ',
                'count':    'count            (  COL1  ) ',
                'cntdist':  'count(distinct   (  COL1  ))',
                'sumabs':   'sum(abs          (  COL1  ))',
                'maxabs':   'max(abs          (  COL1  ))',
                'minabs':   'min(abs          (  COL1  ))',
                'avgabs':   'avg(abs          (  COL1  ))',
                'pctsum':   'sum(             (  COL1  )/sum(COL2)',
                'pctcnt':   'count(           (  COL1  )/count(COL2)'
          }


#%%
############################################################
#
# 3.基础特征  
#
############################################################
config0 = {
    'objects':['transamt'],
    'function':['COUNT','PCTSUM'],
    'condition_cols':[('tsdaydiff',[ ("COL>0 and COL<=1",'最近1天','a') 
                                    ,("COL>0 and COL<=3",'最近3天','b')
                                    ,("COL>0 and COL<=30",'最近30天','c')]),

                    ('transamt',[("COL>0 and COL<=100",'交易金额0~100','d')
                                ,("COL>10000 and COL<=30000",'交易金额10000~30000','e')
                                ,("COL>30000 and COL<=100000",'交易金额30000~100000','f')
                                ,("COL>100000 and COL<=300000",'交易金额100000~300000','g')
                                ,("COL>300000 and COL<=500000",'交易金额300000~500000','h')
                                ,("COL>500000 ",'交易金额大于500000','i')])
    ],
    'tag':'set0',
    'comment':'交易金额transamt - |近1/3/7/30天|成功/失败|'
}

#%%
autosql = AutoSql.AutosqlFlowTable('./config2.xlsx',fun_dict)
basetable_ddl = autosql.base_table_create()
#base_feature_sql,features = autosql.BaseFeatureCreate([config0],['PRI_ACCT_NO_SM3','trans_dt_yuancheng'],window='win1')
#base_feature_sql,features = autosql.BaseFeatureCreate('/Users/yu/Desktop/coding/github/something/AutoSql/config1.xlsx',['PRI_ACCT_NO_SM3','trans_dt_yuancheng','label'],window='win1')

base_feature_sql,features = autosql.base_feature_create('./config2.xlsx',['prikey'],window='d20201201')


#percent_feature_sql = autosql.PercentFeatureCreate([config1,config2,config3,config4,config5,config7],features)
with open('./sql_command2.sql','w') as f:
    f.write('--基础流水表ddl\n')
    f.write(basetable_ddl)
    f.write('--基本特征select语句\n')
    f.write(base_feature_sql)

