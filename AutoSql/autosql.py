from collections import Counter


class AutosqlFlowTable():
    """
    流水数据脚本自动化生成
    """

    def __init__(self,basic_param,fun_dict):
        """
        purpose:
            流水数据特征自动sql脚本生成
        basic_param:dict 流水数据基础表结构参数
        """
        self.basic_param = basic_param
        self.basic_columns = self._get_basic_col_()
        self.fun_dict = fun_dict
        self.winpos_col = [x[0] for x in basic_param['winpos_col']]

    def _get_basic_col_(self):
        col_ls = []
        for i in ['entitys','dimensions','extra_col','partitioned_by']:
            for col in self.basic_param[i]:
                    col_ls.append(col[0])
        col_ls = list(set(col_ls))
        return col_ls

    def _condition_combine_(self, res, condition, condition_cols,window):
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
                if col_nm in self.winpos_col:
                    tmp = '('+v.replace('COL',col_nm+'_'+window)+')'
                else:
                    tmp = '('+v.replace('COL',col_nm)+')'
                self._condition_combine_(res,(condition[0]+' and '+tmp,condition[1]+'_'+col_nm+str(i)),condition_cols[1:],window)
        else:
            res.append(condition)

    def _feasql_(self,config,window):
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
            if i not in self.basic_columns and i not in self.winpos_col:
                print("columns %s not matched"%(i))
                return

        condition_ls = []
        self._condition_combine_(condition_ls,('',''),config['condition_cols'],window)
        
        max_len = max([len("case when %s then  else null end"%(x[0][5:])) for x in condition_ls])+max([len(x) for x in config['objects']])

        sql_command = []
        # 生成特征脚本
        fea_num = 0
        col_ls = []
        for obj in config['objects']:
            for condition in condition_ls:
                for fun_nm in config['function']:
                    fun = self.fun_dict[fun_nm]

                    col_new = config['tag']+'_'+fun_nm+'_'+obj+condition[1]
                    tmp = "{:%s}"%(max([60,len(col_new)]))
                    col_new = tmp.format(col_new)

                    casewhen = "{:%s}"%(max([120,max_len]))
                    casewhen = casewhen.format("case when %s then %s else null end"%(condition[0][5:],obj))
                    casewhen = fun.replace('COL',casewhen)
                    sql_command.append(",%s as %s  --%s\n"%(casewhen,col_new,config['comment']))
                    fea_num+=1
                    col_ls.append(col_new)
        
        if len(set(col_ls))!=len(col_ls):
            print('column name duplicated,please check')
            return
        else:
            return sql_command,col_ls

    def BaseTableCreate(self):
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

        #normal column define
        sql_command.append("%s %s comment '%s'\n"%(col_ls[0][0],col_ls[0][1],col_ls[0][2]))
        for col in col_ls[1:]:
            sql_command.append(", %s %s comment '%s'\n"%(col[0],col[1],col[2]))
        
        #slip window columns define
        for col in param['winpos_col']:
            for i in param['win_nm']:      
                sql_command.append(", %s_%s %s comment '%s'\n"%(col[0],i,col[1],col[2]))
        
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
        sql_command.append(";\n\n\n\n")

        sql_command = ''.join(sql_command)
        return sql_command


        
    def BaseFeatureCreate(self,config_ls,entity,window):
        """
        purpose:
            特征构建sql脚本自动生成
        input:
            config_ls: list of dict 特征构建参数
            entity: string 分析实体
            window: string 窗口选择
        output:
            sql_command: str sql 脚本
        """
        if window not in self.basic_param['win_nm']:
            print('window %s not matched'%(window))
            return


        sql_command = []
        sql_command.append("select\n")
        sql_command.append("%s\n"%(entity))


        col_ls = []
        for i,config in enumerate(config_ls):
            tmp = self._feasql_(config,window)
            if tmp:
                sql_command = sql_command+tmp[0]
                col_ls += tmp[1]
            else:
                return

        # from table group by col
        sql_command.append("from %s.%s\n"%(self.basic_param['database'],self.basic_param['table'][0]))
        sql_command.append("group by %s\n"%(entity))
        sql_command.append(";\n\n\n\n")

        sql_command = ''.join(sql_command)
        print('%s features were created'%(len(col_ls)))
        return sql_command,col_ls

    def PercentFeatureCreate(self,config_ls,features):
        """
        purpose:
            用于构建比例特征
        input:
            config_ls: list of dict 特征构建参数
            entity: string 分析实体
            window: string 窗口选择
        output:
            sql_command: str sql 脚本
        """

        tags = [x['tag'] for x in config_ls]
        features = [x for x in features if x.split('_')[0] in tags and x.split('_')[1] in ('sum','count')] 
        group = {}
        
        for fea in features:
            key = '_'.join(fea.split('_')[0:-1])
            value = fea.split('_')[-1]
            if key not in group:
                group[key]=[value]
            else:
                group[key].append(value)

        sql_command = []
        sql_command.append('select * \n')
        for key in group.keys():
            lower = '+'.join([key+'_'+x for x in group[key]])
            for i in group[key]:
                upper = key+'_'+i
                col_new = upper.replace('_sum_','_percentsum_').replace('_count_','_percentcount_')
                sql_command.append(", %s/(%s) as %s\n"%(upper,lower,col_new))

        sql_command.append('from table_tmp\n')
        sql_command.append(";\n\n\n\n")

        sql_command = ''.join(sql_command)
        return sql_command






class AutosqlSnapshotTable():
    """
    快照数据自动化特征脚本生成
    """
    def __init__(self,basic_param,fun_dict):
        """
        purpose:
            流水数据特征自动sql脚本生成
        basic_param:dict 快照基础表结构参数
        """
        self.basic_param = basic_param
        self.basic_columns = self._get_basic_col_()
        self.fun_dict = fun_dict
        self.winpos_col = [x[0] for x in basic_param['winpos_col']]

    def _get_basic_col_(self):
        col_ls = []
        for i in ['entitys','dimensions','extra_col','partitioned_by','snapshot_col']:
            for col in self.basic_param[i]:
                    col_ls.append(col[0])
        col_ls = list(set(col_ls))
        return col_ls

    def BaseTableCreate(self):
        """
        purpose:
            定义初始快照表结构的DDL语句
        output:
            sql_command: str DDL语句
        """
        param = self.basic_param
        partition_cols = [x[0] for x in param['partitioned_by']]
        col_ls = []
        for i in ['entitys','dimensions','extra_col','snapshot_col']:
            for col in param[i]:
                if col[0] not in partition_cols:
                    col_ls.append(col)
        sql_command = []
        #create table
        sql_command.append("create table %s.%s (\n"%(param['database'],param['table'][0]))

        #normal column define
        sql_command.append("%s %s comment '%s'\n"%(col_ls[0][0],col_ls[0][1],col_ls[0][2]))
        for col in col_ls[1:]:
            sql_command.append(", %s %s comment '%s'\n"%(col[0],col[1],col[2]))
        
        #slip window columns define
        for col in param['winpos_col']:
            for i in param['win_nm']:      
                sql_command.append(", %s_%s %s comment '%s'\n"%(col[0],i,col[1],col[2]))
        
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
        sql_command.append(";\n\n\n\n")

        sql_command = ''.join(sql_command)
        return sql_command
    
    def SequenceFeatureCreate(self,entity,snapshot_col,corr_pair_ls,regression_pair_ls,change_pair_ls):
        """
        purpose:
            基于快找表构建特征
        input:
            corr_pair_ls: list of tuple  相关性特征组
            regression_pair_ls: list of tuple  线性回归特征组
            change_col_ls: list of string  单特征变化特征组

        """
        command_sql = []
        # 相关性特征
        for pair in corr_pair_ls:
            new_col = "corr_%s_%s"%(pair[0],pair[1])
            command_sql.append(",corr(%s,%s) as %s \n" %(pair[0],pair[1],new_col))
        
        # 线性拟合特征
        for pair in regression_pair_ls:
            new_col = "rega_%s_%s"%(pair[0],pair[1])
            a = "(count(*)*sum(%s*%s)-sum(%s)*sum(%s))/(count(*)*sum(%s*%s)-sum(%s)*sum(%s))"%(pair[0],pair[1],pair[0],pair[1],pair[0],pair[0],pair[1],pair[1])
            command_sql.append(",%s as %s \n"\
                %(a,new_col))
            new_col = "regb_%s_%s"%(pair[0],pair[1])
            command_sql.append("avg(%s)-%s*avg(%s) as %s"%(pair[1],a,pair[0],new_col))

        # 平均增长
        for col in change_pair_ls:
            ("avg(col[1] - (lag( %s ) over partition by( %s order by %s) )) as new_col"%( col[1], entity,col[0]))
            
            
        return ''.join(command_sql)
