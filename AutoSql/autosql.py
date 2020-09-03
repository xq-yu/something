from collections import Counter
class AutosqlFlowTable():
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
        
    def feature_create(self,config_ls,entity,window):
        """
        purpose:
            特征构建sql脚本自动生成
        input:
            config_ls: list of dict 特征构建参数
            entity: 分析实体
        output:
            sql_command: str sql 脚本
        """
        if window not in self.basic_param['win_nm']:
            print('window %s not matched'%(window))
            return


        sql_command = []
        sql_command.append("select\n")
        sql_command.append("%s\n"%(entity))


        fea_num = 0
        for config in config_ls:
            tmp = self._feasql_(config,window)
            if tmp:
                sql_command = sql_command+tmp[0]
                fea_num += tmp[1]
            else:
                return

        # from table group by col
        sql_command.append("from %s.%s\n"%(self.basic_param['database'],self.basic_param['table'][0]))
        sql_command.append("group by %s\n"%(entity))
        sql_command.append(";\n\n\n\n")

        sql_command = ''.join(sql_command)
        print('%s features were created'%(fea_num))
        return sql_command


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
                    col_new = fun_nm+'_'+obj+condition[1]

                    tmp = "{:%s}"%(max([120,max_len]))
                    tmp = tmp.format("case when %s then %s else null end"%(condition[0][5:],obj))
                    tmp = fun.replace('COL',tmp)
                    sql_command.append(",%s as %s\n"%(tmp,col_new))
                    fea_num+=1
                    col_ls.append(col_new)
        
        if len(set(col_ls))!=len(col_ls):
            print('column name duplicated,please check')
            return
        else:
            return sql_command,fea_num











