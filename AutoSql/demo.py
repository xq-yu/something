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
        'database': 'mydatabase',  #数据库名
        'table': ('tmp','表注释'),  #表名
        'entitys':[('entityid','string','客户id')],  #实体字段
        
        #维度字段
        'dimensions':[
                         ('tsdate','int','yyyymmdd')
                        ,('tsyyyymm','int',"年月")
                        ,('tsyear','int',"年_yyyy",)
                        ,('tsmon','int',"月 [1,12]",)
                        ,('tsday','int',"日 [1,31]",)
                        ,('tshour','int',"小时 [0,23]")
                        ,('tsmin','int',"分 [0,59]")
                        ,('tssec','int',"秒 [0,59]")
                        ,('tsdayofweek','int',"星期 [0,6]")
                        ,('tsweekofyear','int',"一年中的第几个星期 [1,53]")
                        ,('tsdayofyear','int',"一年中的第几天 [1,365]")
                        ,('tsunixtime','int',"uninx 时间戳，单位为秒" )  
                        ,('transtype','string',"交易类型 ")
                        ,('transamt','double','交易金额 ')
                        ,('counterpart','string','交易对手 ')
                        ,('cdflg','string','借代标志 (in,out)')
                        ,('cardattri','string','卡类型 (credit,debit)')
                        ,('cardno','string','卡号')
                    ],
        
        #窗口相对位置标记字段
        'winpos_col':[ ('tsmondiff','int',"相对月份 (1,2,3,4...)")
                      ,('tsdaydiff','int',"相对日期 (1,2,3,4...)")],
        #窗口名称
        'win_nm':['win1','win2'],

        'extra_col':[],  #其他字段
        'partitioned_by':[('tsdate','int','yyyymmdd')],  #分区字段
        'clustered_by':['entityid','tsyyyymm'],   #分桶字段
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
                'sum':      'sum           (  COL  ) ',
                'avg':      'avg           (  COL  ) ',
                'max':      'max           (  COL  ) ',
                'min':      'min           (  COL  ) ',
                'std':      'std           (  COL  ) ',
                'count':    'count         (  COL  ) ',
                'cntdist':  'count(distinct(  COL  ))',
                'sumabs':   'sum(abs       (  COL  ))',
                'maxabs':   'max(abs       (  COL  ))',
                'minabs':   'min(abs       (  COL  ))',
                'avgabs':   'avg(abs       (  COL  ))',
          }


#%%
############################################################
#
# 3.基础特征  
#
############################################################
# 3.1 交易金额transamt -｜近1/3/6个月｜转入/转出| 
config1 = {
    'objects':['transamt'],
    'function':['sum','avg','max','min','std','count'],
    'condition_cols':[('tsmondiff',[ "COL>0 and COL<=1" 
                                    ,"COL>0 and COL<=3"
                                    ,"COL>0 and COL<=6"]),

                      ('cdflg',[ "COL='in'"
                                ,"COL='out'"])
    ],
    'tag':'set1',
    'comment':'交易金额transamt -｜近1/3/6个月｜转入/转出| '
}

# 3.2 交易金额transamt - |近1/3/6个月|借记卡/贷记卡|转出|
config2 = {
    'objects':['transamt'],
    'function':['sum','avg','max','min','std','count'],
    'condition_cols':[('tsmondiff',[ "COL>0 and COL<=1"
                                    ,"COL>0 and COL<=3"
                                    ,"COL>0 and COL<=6"]),

                      ('cdflg',["COL='out'"]),

                      ('cardattri',[ "COL='credict'"
                                    ,"COL='debit'"])
    ],
    'tag':'set2',
    'comment':'交易金额transamt - |近1/3/6个月|转出|借记卡/贷记卡|'
}
# 3.3 交易金额transamt - |近1/3/6个月|交易类型|
config3 = {
    'objects':['transamt'],
    'function':['sum','avg','max','min','std','count'],
    'condition_cols':[('tsmondiff',[ "COL>0 and COL<=1"
                                    ,"COL>0 and COL<=3"
                                    ,"COL>0 and COL<=6"]),

                      ('transtype',[ "transtype='type1'"
                                    ,"transtype='type2'"
                                    ,"transtype='type3'"])
    ],
    'tag':'set3',
    'comment':'交易金额transamt - |近1/3/6个月|交易类型|'
}
# 3.4 交易金额transamt - ｜近6个月｜转入/转出｜周末/工作日｜
config4 = {
    'objects':['transamt'],
    'function':['sum','avg','max','min','std','count'],
    'condition_cols':[('tsmondiff',["COL>0 and COL<=6"]),

                      ('cdflg',[ "COL='in'"
                                ,"COL='out'"]),

                      ('tsdayofweek',[ "COL in (0,6)"
                                      ,"COL not in (0,6)"])


    ],
    'tag':'set4',
    'comment':'交易金额transamt - ｜近6个月｜周末/工作日｜转入/转出｜'
}
# 3.5 交易金额transamt - ｜近1/3/6个月｜转入/转出｜一天中的某个时段|
config5 = {
    'objects':['transamt'],
    'function':['sum','avg','max','min','std','count'],
    'condition_cols':[('tsmondiff',[ "COL>0 and COL<=1"
                                    ,"COL>0 and COL<=3"
                                    ,"COL>0 and COL<=6"]),

                      ('cdflg',[ "COL='in'"
                                ,"COL='out'"]),

                      ('tshour',[ "COL in (22,23,0)"
                                 ,"COL in (1,2,3,4)"
                                 ,"COL in (5,6,7)"
                                 ,"COL in (8,9,10)"
                                 ,"COL in (11,12,13)"
                                 ,"COL in (14,15,16,17)"
                                 ,"COL in (18,19,20,21)"])


    ],
    'tag':'set5',
    'comment':'交易金额transamt - ｜近1/3/6个月｜转入/转出｜一天中的某个时段|'
}
# 3.6 交易金额transamt - ｜近1/3/6个月｜一天中的某个时段｜交易类型｜
config6 = {
    'objects':['transamt'],
    'function':['sum','avg','count'],
    'condition_cols':[('tsmondiff',[ "COL>0 and COL<=1"
                                    ,"COL>0 and COL<=3"
                                    ,"COL>0 and COL<=6"]),

                      ('tshour',[ "COL in (22,23,0)"
                                 ,"COL in (1,2,3,4)"
                                 ,"COL in (5,6,7)"
                                 ,"COL in (8,9,10)"
                                 ,"COL in (11,12,13)"
                                 ,"COL in (14,15,16,17)"
                                 ,"COL in (18,19,20,21)"]),

                      ('transtype',[ "transtype='type1'"
                                    ,"transtype='type2'"
                                    ,"transtype='type3'"])
    ],
    'tag':'set6',
    'comment':'相对交易时间tsdaydiff - ｜近1/3/6个月｜一天中的某个时段｜交易类型｜'
}
# 3.7 交易金额transamt - ｜近1/3/6个月｜转入/转出|交易金额区间|
config7 = {
    'objects':['transamt'],
    'function':['sum','count'],
    'condition_cols':[('tsmondiff',[ "COL>0 and COL<=1"
                                    ,"COL>0 and COL<=3"
                                    ,"COL>0 and COL<=6"]),

                      ('cdflg',[ "COL='in'"
                                ,"COL='out'"]),

                      ('transamt',[ "COL>=0     and COL<5     "
                                   ,"COL>=5     and COL<10    "
                                   ,"COL>=10    and COL<50    "
                                   ,"COL>=50    and COL<100   "
                                   ,"COL>=100   and COL<500   "
                                   ,"COL>=500   and COL<1000  "
                                   ,"COL>=1000  and COL<5000  "
                                   ,"COL>=5000  and COL<10000 "
                                   ,"COL>=10000 and COL<50000 "
                                   ,"COL>=50000               "  ]),

    ],
    'tag':'set7',
    'comment':'交易金额transamt - ｜近1/3/6个月｜转入/转出|交易金额区间|'
}
# 3.8 交易日期tsdate - ｜近1/3/6个月｜转入/转出｜借记卡/贷记卡|
config8 = {
    'objects':['tsdate'],
    'function':['cntdist'],
    'condition_cols':[('tsmondiff',[ "COL>0 and COL<=1"
                                    ,"COL>0 and COL<=3"
                                    ,"COL>0 and COL<=6"]),

                      ('cdflg',[ "COL='in'"
                                ,"COL='out'"]),

                      ('cardattri',[ "COL='credict'"
                                    ,"COL='debit'"])
    ],
    'tag':'set8',
    'comment':'交易日期tsdate - ｜近1/3/6个月｜转入/转出｜借记卡/贷记卡|'
}
# 3.9 交易日期tsdate - ｜近1/3/6个月｜借记卡/贷记卡|
config9 = {
    'objects':['tsdate'],
    'function':['cntdist'],
    'condition_cols':[('tsmondiff',[ "COL>0 and COL<=1"
                                    ,"COL>0 and COL<=3"
                                    ,"COL>0 and COL<=6"]),

                      ('cardattri',[ "COL='credict'"
                                    ,"COL='debit'"])
    ],
    'tag':'set9',
    'comment':'交易日期tsdate - ｜近1/3/6个月｜借记卡/贷记卡|'
}
# 3.10 交易对手counterpart - ｜近1/3/6个月｜借记卡/贷记卡|
config10 = {
    'objects':['counterpart'],
    'function':['cntdist'],
    'condition_cols':[('tsmondiff',[ "COL>0 and COL<=1"
                                    ,"COL>0 and COL<=3"
                                    ,"COL>0 and COL<=6"]),

                      ('cdflg',[ "COL='in'"
                                ,"COL='out'"]),

                      ('cardattri',[ "COL='credict'"
                                    ,"COL='debit'"])
    ],
    'tag':'set10',
    'comment':'交易对手counterpart - ｜近1/3/6个月｜借记卡/贷记卡|'
}
# 3.11 相对交易时间tsdaydiff - |近6个月｜借记卡/贷记卡｜转入/转出｜
config11 = {
    'objects':['tsdaydiff'],
    'function':['min'],
    'condition_cols':[('tsmondiff',["COL>0 and COL<=6"]),

                      ('cdflg',[ "COL='in'"
                                ,"COL='out'"]),

                      ('cardattri',[ "COL='credict'"
                                    ,"COL='debit'"])
    ],
    'tag':'set11',
    'comment':'相对交易时间tsdaydiff - |近6个月｜借记卡/贷记卡｜转入/转出｜'
}




#%%
autosql = AutoSql.AutosqlFlowTable(basic_param,fun_dict)
basetable_ddl = autosql.BaseTableCreate()
base_feature_sql,features = autosql.BaseFeatureCreate([config1,config2,config3,config4,config5,config6,config7,config8,config9,config10,config11],'entityid',window='win2')
percent_feature_sql = autosql.PercentFeatureCreate([config1,config2,config3,config4,config5,config7],features)


#%%

with open('./sql_command.sql','w') as f:
    f.write('--基础流水表ddl\n')
    f.write(basetable_ddl)
    f.write('--基本特征select语句\n')
    f.write(base_feature_sql)
    f.write('--百分比特征select语句\n')
    f.write(percent_feature_sql)


#%%

basic_param = {
        'database': 'mydatabase',  #数据库名
        'table': ('SnapshotTable','表注释'),  #表名
        'entitys':[('entityid','string','客户id')],  #实体字段
        
        #维度字段
        'dimensions':[ 
                         ('transtype','string',"交易类型 ")
                        ,('transamt','double','总交易额 ')
                        ,('resamt','double','账户余额')
                        ,('transcnt','double','交易次数')
                    ],
        #快照标记字段
        'snapshot_col':[('tsyyyymm','int',"年月")],
        
        #窗口相对位置标记字段
        'winpos_col':[ ('tsmondiff','int',"相对月份 (1,2,3,4...)")],
        #窗口名称
        'win_nm':['win1','win2'],

        'extra_col':[],  #其他字段
        'partitioned_by':[('tsyyyymm','int','yyyymmdd')],  #分区字段
        'clustered_by':['entityid'],   #分桶字段
        'bucket_num':135,    # 分桶数量
        'field_sep':',',    # 字段分隔符
        'stored_as':'orc',   #表格式
        'location':'hdfs://master:9000/user/hive/warehouse/mydatabase.db/' #表路径

        
}



fun_dict = {
                'sum':      'sum           (  COL  ) ',
                'avg':      'avg           (  COL  ) ',
                'max':      'max           (  COL  ) ',
                'min':      'min           (  COL  ) ',
                'std':      'std           (  COL  ) ',
                'count':    'count         (  COL  ) ',
                'cntdist':  'count(distinct(  COL  ))',
                'sumabs':   'sum(abs       (  COL  ))',
                'maxabs':   'max(abs       (  COL  ))',
                'minabs':   'min(abs       (  COL  ))',
                'avgabs':   'avg(abs       (  COL  ))',
          }

autosql = AutoSql.AutosqlSnapshotTable(basic_param,fun_dict)
print(autosql.SequenceFeatureCreate(corr_pair_ls=[('a','b')],regression_pair_ls=[('a','b')],change_col_ls=[('a','b')]))