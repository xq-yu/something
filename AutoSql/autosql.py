param = {
        'database': 'mydatabase',  #数据库名
        'table': ('tmp','表注释'),  #表名
        'entitys':[('entityid','string','客户id')],  #实体字段
        
        #维度字段
        'dimensions':[
                         ('tsdate','int','yyyymmdd')
                        ,('tsyyyymm','int',"年月")
                        ,('tsyear','int',"年_yyyy",)
                        ,('tsmon','int',"月 [1,12]",)
                        ,('tsmondiff','int',"相对月份 [1,3,6,7]")
                        ,('tsday','int',"日 [1,31]",)
                        ,('tshour','int',"小时 [0,23]")
                        ,('tsmin','int',"分 [0,59]")
                        ,('tssec','int',"秒 [0,59]")
                        ,('tsdayofweek','int',"星期 [1,7]")
                        ,('tsweekofyear','int',"一年中的第几个星期 [1,53]")
                        ,('tsdayofyear','int',"一年中的第几天 [1,365]")
                        ,('tsunixtime','int',"uninx 时间戳，单位为秒" )  
                        ,('transtype','string',"交易类型 ")
                        ,('transamt','double','交易金额')
                    ],
        'extra_col':[],  #其他字段
        'partitioned_by':[('tsdate','int','yyyymmdd')],  #分区字段
        'clustered_by':['entityid','ts_yyyymm'],   #分桶字段
        'bucket_num':135,    # 分桶数量
        'field_sep':',',    # 字段分隔符
        'stored_as':'orc',   #表格式
        'location':'hdfs://master:9000/user/hive/warehouse/mydatabase.db/' #表路径
}

class AutosqlFlowTable():
    def __init__(self,basic_param):
        """
        purpose:
            流水数据特征自动sql脚本生成
        basic_param:dict 流水数据基础表结构参数
        """
        self.basic_param = basic_param
        self.basic_columns = self._get_basic_col_()
        self.fun_dict = {'mean':  'avg           (  COL  ) ',
                        'avg':    'avg           (  COL  ) ',
                        'max':    'max           (  COL  ) ',
                        'min':    'min           (  COL  ) ',
                        'std':    'std           (  COL  ) ',
                        'count':  'count         (  COL  ) ',
                        'cntdist':'count(distinct(  COL  ))'}

    def _get_basic_col_(self):
        col_ls = []
        for i in ['entitys','dimensions','extra_col','partitioned_by']:
            for col in param[i]:
                    col_ls.append(col[0])
        col_ls = list(set(col_ls))
        return col_ls

    def autosql_initialize(self):
        """
        purpose:
            定义初始流水表结构的DDL语句
        output:
            sql_command: str DDL语句
        """
        param = self.basic_param
        partition_cols = [x[0] for x in param['partitioned_by']]
        col_ls = []
        for i in ['entitys','dimensions','extra_col']:
            for col in param[i]:
                if col[0] not in partition_cols:
                    col_ls.append(col)
        sql_command = []
        #create table
        sql_command.append("create table %s.%s (\n"%(param['database'],param['table'][0]))

        #column define
        sql_command.append("%s %s comment '%s'\n"%(col_ls[0][0],col_ls[0][1],col_ls[0][2]))
        for col in col_ls[1:]:
            sql_command.append(", %s %s comment '%s'\n"%(col[0],col[1],col[2]))
        sql_command.append(")\n")
        
        #table comment
        sql_command.append("comment '%s'\n"%(param['table'][1]))
        
        #partitioned by 
        if len(param['partitioned_by'])>0:
            sql_command.append("partitioned by (\n")
            sql_command.append("%s %s comment '%s'\n"%(param['partitioned_by'][0][0],param['partitioned_by'][0][1],param['partitioned_by'][0][2]))
            for col in param['partitioned_by'][1:]:
                sql_command.append(", %s %s comment '%s')\n"%(col[0],col[1],col[2]))     
            sql_command.append(")\n")

        #clustered by
        if len(param['clustered_by'])>0:
            sql_command.append("clustered by %s\n"%(str(param['clustered_by']).replace("'",'').replace("[",'(').replace("]",')')))
            sql_command.append("into %s buckets\n"%(param['bucket_num']))
        #
        sql_command.append("row format delimited fields terminated by '%s'\n"%(param['field_sep']))


        #stored as 
        sql_command.append("stored as %s\n"%(param['stored_as']))
        
        #loacation
        sql_command.append("location\n")
        sql_command.append("'%s%s'\n"%(param['location'],param['table'][0]))

        sql_command = ''.join(sql_command)
        return sql_command

    def _condition_combine_(self, res, condition, condition_cols):
        """
        purpose:
            条件组合
        input:
            res: list 用于传递指针
            condition: tuple like ('condition','name')  
            condition_cols: list of tuples like [('col_nm','condition')]
        """
        if condition_cols:
            col_nm = condition_cols[0][0]
            for i,v in enumerate(condition_cols[0][1]):
                tmp = v.replace('COL',col_nm)
                self._condition_combine_(res,(condition[0]+' and '+tmp,condition[1]+'_'+col_nm+str(i)),condition_cols[1:])
        else:
            res.append(condition)
        
    def feature_create(self,config_ls,entity):
        """
        purpose:
            特征构建sql脚本自动生成
        input:
            config_ls: list of dict 特征构建参数
            entity: 分析实体
        output:
            sql_command: str sql 脚本
        """
        sql_command = []
        sql_command.append("select\n")
        sql_command.append("%s\n"%(entity))

        for config in config_ls:
            if self._feasql_(config):
                sql_command = sql_command+self._feasql_(config)
            else:
                return

        # from table group by col
        sql_command.append("from %s.%s\n"%(self.basic_param['database'],self.basic_param['table'][0]))
        sql_command.append("group by %s\n"%(entity))

        sql_command = ''.join(sql_command)
        return sql_command


    def _feasql_(self,config):
        """
        purpose:
            特征构建sql脚本自动生成
        input:
            config: dict 特征构建参数
        output:
            sql_command: list 特征构建脚本
        """

        # 检查字段是否存在
        for i in [x[0] for x in config['condition_cols']]+config['objects']:
            if i not in self.basic_columns:
                print("columns %s not matched"%(i))
                return

        condition_ls = []
        self._condition_combine_(condition_ls,('',''),config['condition_cols'])
        
        max_len = max([len("case when %s then  else null end"%(x[0][5:])) for x in condition_ls])+max([len(x) for x in config['objects']])

        sql_command = []
        # 生成特征脚本
        for obj in config['objects']:
            for condition in condition_ls:
                for fun_nm in config['function']:
                    fun = self.fun_dict[fun_nm]
                    col_new = fun_nm+'_'+obj+condition[1]

                    tmp = "{:%s}"%(max([120,max_len]))
                    tmp = tmp.format("case when %s then %s else null end"%(condition[0][5:],obj))
                    tmp = fun.replace('COL',tmp)
                    sql_command.append(",%s as %s\n"%(tmp,col_new))
        
        return sql_command


config0 = {
    'objects':['transamt'],
    'function':['avg','max','min','std','count','cntdist'],
    'condition_cols':[('transtype',["COL='type1'","COL='type2'","COL='type3'"]),
                      ('tsdayofweek',["COL in (6,7)"])
    ]
}

config1 = {
    'objects':['transamt'],
    'function':['avg','max','min','std','count','cntdist'],
    'condition_cols':[('transtype',["COL='type1'","COL='type2'","COL='type3'"]),
                      ('tsmondiff',["COL<=1", "COL<=3","COL<=6"]),
                      ('tsdayofweek',["COL in (6,7)"])
    ]
}


config2 = {
    'objects':['transamt'],
    'function':['avg','max','min','std','count','cntdist'],
    'condition_cols':[('transtype',["COL='type1'","COL='type2'","COL='type3'"])
    ]
}

config3 = {
    'objects':['tsdate','transtype'],
    'function':['cntdist'],
    'condition_cols':[('transtype',["COL='type1'","COL='type2'","COL='type3'"])
    ]
}

autosql = AutosqlFlowTable(param)
print(autosql.feature_create([config0,config1,config2,config3],'entityid'))
            








