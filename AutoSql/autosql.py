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
        self.fun_dict = self.__getfundict__(fun_dict)
        self.winpos_col = [x[0].upper() for x in basic_param['winpos_col']]

    def __getfundict__(self,fun_dict):
        fun_dict_new = {}
        for i in fun_dict.keys():
            fun_dict_new[i.upper()] = fun_dict[i]
        return fun_dict_new

    def _get_basic_col_(self):
        col_ls = []
        for i in ['entitys','dimensions','extra_col','partitioned_by']:
            for col in self.basic_param[i]:
                    col_ls.append(col[0].upper())
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
                if col_nm.upper() in self.winpos_col:
                    tmp = '('+v[0].replace('COL',col_nm+'_'+window)+')'
                else:
                    tmp = '('+v[0].replace('COL',col_nm)+')'
                self._condition_combine_(res,
                                        (condition[0]+' and '+tmp,
                                        condition[1]+'__'+v[2],
                                        condition[2]+'|'+v[1]
                                        ),
                                        condition_cols[1:],
                                        window)
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
        for i in [x[0] for x in config['condition_cols']]+[x for x in config['objects']]:
            if i.upper() not in self.basic_columns and i.upper() not in self.winpos_col:
                raise ValueError("columns %s not matched with base table"%(i))

        condition_ls = []
        self._condition_combine_(condition_ls,('','',''),config['condition_cols'],window)
        
        max_len = max([len("case when %s then  else null end"%(x[0][5:])) for x in condition_ls])+max([len(x) for x in config['objects']])

        sql_command = []
        # 生成特征脚本
        fea_num = 0
        col_ls = []
        for obj in (x for x in config['objects']):
            for condition in condition_ls:
                for fun_nm in [x for x in config['function']]:
                    fun = self.fun_dict[fun_nm]
                    col_new = config['tag']+'__'+fun_nm+'__'+obj+condition[1]
                    tmp = "{:%s}"%(max([80,len(col_new)]))
                    col_new = tmp.format(col_new)

                    casewhen = "{:%s}"%(max([200,max_len]))
                    casewhen = casewhen.format("case when %s then %s else null end"%(condition[0][5:],obj))
                    casewhen = fun.replace('COL',casewhen)

                    comment = obj+'|'+fun_nm+'|$'+condition[2]
                    sql_command.append(",%s as %s   --%s\n"%(casewhen,col_new,comment))#config['comment']))
                    fea_num+=1
                    col_ls.append(col_new)
        
        if len(set(col_ls))!=len(col_ls):
            raise ValueError('column name duplicated,please check')
        else:
            return sql_command,col_ls


    def __ReadConfig__(self,file):
        """
        purpose:
            从excel文件中读取特征配置信息
        input: 
            file: string excel 文件路径
        output:
            config_list: list 配置字典列表
        """

        # 读取excel 文件
        import xlrd
        import numpy as np
        data= xlrd.open_workbook(file)
        data = data.sheet_by_index(0)
        table = [['' for i in range(5)] for i in range(data.nrows)]
        for i in range(5):
            for j in range(data.nrows):
                table[j][i] = data.cell(j,i).value.strip().upper()
        table = [x for x in table if sum([len(i) for i in x])!=0]

        # 空值向下填充方便后续操作
        for i in range(2):
            r = 0
            while r<len(table):
                if table[r][i]=='':
                    table[r][i]=table[r-1][i]
                r+=1
        config_orig = []
        i=0
        j=1
        while i<len(table) and j<len(table):
            while j<len(table) and table[j][0].upper()!='TAG':
                j+=1
            config_orig.append(table[i:j])
            i=j
            j=i+1

        # 配置解析
        config = {}
        config_list = []
        for config_tmp in config_orig:
            config = {'objects':[],'function':[],'condition_cols':[]}
            config_tmp = np.array(config_tmp)
            config['objects'] = list(config_tmp[config_tmp[:,0]=='OBJECTS',1])
            config['function'] = list(config_tmp[config_tmp[:,0]=='FUNCTION',1])
            config['tag'] = config_tmp[config_tmp[:,0]=='TAG',1][0]
            config['comment'] = config_tmp[config_tmp[:,0]=='COMMENT',1][0]
            cols = config_tmp[config_tmp[:,0]=='CONDITION_COLS',1]
            for col in set(cols):
                tmp = (col,[tuple(x) for x in config_tmp[(config_tmp[:,1]==col) &(config_tmp[:,0]=='CONDITION_COLS'),:][:,[2,3,4]]])
                config['condition_cols'].append(tmp)
            config_list.append(config.copy())
        return config_list


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
            config_ls: list of dict 特征构建参数|excel 格式文件
            entity: list 分析实体
            window: string 窗口选择
        output:
            sql_command: str sql 脚本
        """
        if type(config_ls)==str:
            config_ls = self.__ReadConfig__(config_ls)

        if window not in self.basic_param['win_nm']:
            raise ValueError('window %s not matched with base table'%(window))

        sql_command = []
        sql_command.append("select\n")
        sql_command.append("%s\n"%(','.join(entity)))


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
        sql_command.append("group by %s\n"%(','.join(entity)))
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
        self.fun_dict = self.__getfundict__(fun_dict)
        self.winpos_col = [x[0] for x in basic_param['winpos_col']]

    def __getfundict__(self,fun_dict):
        fun_dict_new = {}
        for i in fun_dict.keys:
            fun_dict_new[i.upper()] = fun_dict[i]
        return fun_dict_new

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
