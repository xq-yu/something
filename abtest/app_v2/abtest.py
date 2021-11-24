"""
############################################## 2021-06 ################################################
本程序用于abtest实验创建与管理。
基于mysql对实验参数进行存储和管理，独立于生产环境
通过自动生成sql脚本进行实验分组，因此并不直接对生产环境产生影响
由于没有对接生产系统，暂时只支持离线跑批分组功能
本程序负责基本的ABtest要求，任何实验之间、之外的影响因素需要创建人做好细致考虑


已实现功能：
    实验参数化配置
    实验创建与冲突检验
    实验层创建与正交性控制
    实验分布可视化
    自动生成分组sql
    历史实验信息检索
    实验效果检验(完成部分指标)
待实现功能:
    实验流量调整（待定）
    流量独占实验

不同类型实验的配置方案：
1. 页面展示方案对比类: 前置活跃用户以及其他筛选条件，不同分组使用不同展示方案（创建一个实验即可）
2. 用户筛选营销效果分析类: 前置用户筛选条件(模型or规则) ,设置实验组和对照组，实验组做营销，对照组不做营销（创建一个实验即可）
3. 规则筛选用户和模型筛选用户效果对比: 
    方案一:在同一层创建两个实验，一个规则实验，一个模型实验，，每个实验分实验组和对照组，实验之间用户流量互斥（流量利用率不高,在小流量测试时可以使用）
    方案二:人为设置3个前置条件，分别为单模型筛选用户，单规则筛选用户，规则模型交集，每个实验分实验组和对照组，每个实验在不同层（流量利用率较高）
4. 根据模型分数分组对比: 
    方案一:根据用户组别分别设置不同的实验，在同一个层创建不同的实验，每个实验分实验组和对照组，实验之间流量互斥(流量利用率不高,在小流量测试时可以使用)
    方案二:根据用户组别分别设置不同的实验，每个实验在不同的实验层，每个实验分实验组和对照组(实验管理困难，不推荐)
    方案三:只建立一个实验2，分组通过后期手动分析(推荐)
#######################################################################################################
"""
import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes

class abtest():
    def __init__(self,cur):
        self.cur = cur

    def create_layer(self,layer_info):
        """
        purpose:
            创建实验层
        input:
            layer_info: dict 实验层信息
        """

        self.cur.execute('select field_id from fields')
        exists_fields = [x[0] for x in self.cur.fetchall()]

        # 检查field是否存在
        if layer_info['field_id'] not in exists_fields:
            raise ValueError("field %s not exists!"%(layer_info['field_id']))
        
         # 实验层是否存在
        self.cur.execute('select field_id,layer_id from layers')
        exists_layer = [[x[0],x[1]] for x in self.cur.fetchall()]
        if [layer_info['field_id'],layer_info['layer_id']] in exists_layer:
            raise ValueError("the layer %s in field %s already exists!"%(layer_info['layer_id'],layer_info['field_id']))


        sql = """
        INSERT INTO layers
        (layer_id 
        ,field_id
        ,hash_col
        ,hash_method 
        ,bucket_num 
        ,comment
        ,creator
        ,available_flg 
        ,create_time )
        VALUES
        (%s,%s,'%s','%s',%s,'%s','%s',%s,'%s');
        """%(layer_info['layer_id']
        ,layer_info['field_id']
        ,layer_info['hash_col']
        ,layer_info['hash_method']
        ,layer_info['bucket_num']
        ,layer_info['comment']
        ,layer_info['creator']
        ,layer_info['available_flg']
        ,layer_info['create_time'])

        self.cur.execute(sql)

    def __exp_info_check__(self,exp_id,exp_info,group_info):
        """
        purpose:
            检查待创建实验信息的准确性
        input:
            exp_id: int 实验id
            exp_info: dict 实验信息
            group_info: dict 实验分组信息
        """

        # 实验层是否存在
        self.cur.execute('select field_id,layer_id from layers')
        exists_layer = [[x[0],x[1]] for x in self.cur.fetchall()]
        if [exp_info['field'],exp_info['layer']] not in exists_layer:
            raise ValueError("the layer %s in field %s not exists!"%(exp_info['layer'],exp_info['field']))
        
        # 检查实验id是否已经存在
        self.cur.execute('select exp_id from experiment')
        exists_exp = [x[0] for x in self.cur.fetchall()]
        if exp_id in exists_exp:
            raise ValueError("exp_id %s already exists!"%(exp_id))
        
        # 检查是否有重复的桶被申请
        applied_buckets = []
        for group_id in group_info.keys():
            applied_buckets = applied_buckets+group_info[group_id][0]
        if len(set(applied_buckets))!=len(applied_buckets):
            raise ValueError("there are duplicated buckets to apply!")
        
        # 检查桶编号是否超出当前层桶数的最大值
        sql = "select bucket_num from layers where field_id=%s and layer_id=%s"%(exp_info['field'],exp_info['layer'])
        self.cur.execute(sql)
        max_bucket_id = self.cur.fetchall()[0][0]
        if max(applied_buckets) > max_bucket_id:
            raise ValueError("max bucket_id is over limited!")
        
        # 检查桶是否已经被其他实验占用,仅仅提示警告,在激活实验时才会提示错误
        used_buckets = self.used_buckets()
        used_buckets = list(used_buckets.loc[(used_buckets.field==exp_info['field']) & (used_buckets.layer==exp_info['layer']),'bucket_id'])
        for bucket_id in applied_buckets: 
            if bucket_id in used_buckets:
                raise ValueError("bucket %s has been used!"%(bucket_id))
        return 


    def create_exp(self,exp_id,exp_info,group_info):
        """
        purpose:
            创建实验和分组
        input:
            exp_id: int 实验id
            exp_info: dict 实验基本信息
            group_info: dict 实验分组信息
        """

        #数据检查
        print('start experiment infomation pre check...')
        self.__exp_info_check__(exp_id,exp_info,group_info)
        print('experiment infomation is statisfied!')

        print('start create experiment')
        sql = """
        insert into experiment(
        exp_id
        ,exp_name
        ,department
        ,admin
        ,field
        ,layer
        ,create_time
        ,comment
        ,valid_flg
        ,pre_sql  )     
        VALUES
        (%s,'%s','%s','%s',%s,%s,'%s','%s',%s,'%s');
        """%(exp_id,
            exp_info['exp_name'],
            exp_info['department'],
            exp_info['admin'],
            exp_info['field'],
            exp_info['layer'],
            exp_info['create_time'],
            exp_info['comment'],
            0,
            exp_info['pre_sql'])
        self.cur.execute(sql)

        print('start applying for experiment buckets')
        # 创建分组
        rows = []
        for group_id in group_info.keys():
            buckets = group_info[group_id][0]
            comment = group_info[group_id][1]
            for bucket_id in buckets:
                rows.append("(%s,%s,%s,'%s')"%(exp_id,bucket_id,group_id,comment))

        sql = """
        insert into exp_group(
            exp_id
            ,bucket_id
            ,group_id
            ,comment  )  
        VALUES
        %s;
        """%(','.join(rows))
        try:
            self.cur.execute(sql)
        except Exception as e:
            sql = """
            delete from experiment where exp_id=%s
            """%(exp_id)
            self.cur.execute(sql)
            print('apply for buckets failed!')
            raise e
        return 


    def run_exp(self,exp_id):
        """
        purpose:
            启动实验
        input:
            exp_id: int 实验id
        """
        sql = "select * from experiment where exp_id=%s"%(exp_id)
        self.cur.execute(sql)
        cols = [x[0] for x in self.cur.description]
        exp_info = pd.DataFrame(self.cur.fetchall(),columns=cols)
        
        # 实验id是否存在
        if len(exp_info)==0:
            raise ValueError("experiment %s not exists"%(exp_id))     

        # 实验id是否已经激活
        if exp_info['valid_flg'].iloc[0]:
            print("experiment %s already acitvated !")
            return 
        
        
        # 检查实验流量桶是否被占用
        used_buckets = self.used_buckets()
        used_buckets = list(used_buckets.loc[(used_buckets.field==exp_info['field'].iloc[0]) & (used_buckets.layer==exp_info['layer'].iloc[0]),'bucket_id'])
  
        sql = "select bucket_id from exp_group where exp_id=%s"%(exp_id)
        self.cur.execute(sql)
        applied_buckets = [x[0] for x in self.cur.fetchall()]
        for bucket_id in applied_buckets:    
            if bucket_id in used_buckets:
                raise ValueError("bucket %s has been used!"%(bucket_id))
        
        sql = "update experiment set valid_flg=1 where exp_id=%s"%(exp_id)
        self.cur.execute(sql)
        return
    
    def stop_exp(self,exp_id):
        """
        purpose:
            停止实验
        input:
            exp_id: int 实验id
        """
        sql = "select * from experiment where exp_id=%s"%(exp_id)
        self.cur.execute(sql)
        cols = [x[0] for x in self.cur.description]
        exp_info = pd.DataFrame(self.cur.fetchall(),columns=cols)

        # 实验id是否存在
        if len(exp_info)==0:
            raise ValueError("experiment %s not exists"%(exp_id))     

        # 实验id是否停止
        if not exp_info['valid_flg'].iloc[0]:
            print("experiment %s already stoped !")
            return 

        sql = "update experiment set valid_flg=0 where exp_id=%s"%(exp_id)
        self.cur.execute(sql)


    def delete_exp(self,exp_id):
        """
        purpose:
            停止实验
        input:
            exp_id: int 实验id
        """
        sql = "select * from experiment where exp_id=%s"%(exp_id)
        self.cur.execute(sql)
        cols = [x[0] for x in self.cur.description]
        exp_info = pd.DataFrame(self.cur.fetchall(),columns=cols)

        # 实验id是否存在
        if len(exp_info)==0:
            raise ValueError("experiment %s not exists"%(exp_id))     

        sql = "delete from experiment where exp_id=%s"%(exp_id)
        self.cur.execute(sql)
        sql = "delete from exp_group where exp_id=%s"%(exp_id)
        self.cur.execute(sql)


    def get_exp(self,exp_id):
        """
        purpose:
            获取实验信息
        input:
            exp_id: 实验id
        output:
            exp_info: df 实验信息
            group_info: df 实验分组信息
            sql: dict 分组sql
        """

        # 实验基本信息
        sql = "select * from experiment where exp_id=%s"%(exp_id)
        self.cur.execute(sql)
        cols = [x[0] for x in self.cur.description]
        exp_info = pd.DataFrame(self.cur.fetchall(),columns=cols)
        if len(exp_info)==0:
            raise ValueError("experiment %s not exists"%(exp_id))

        # 实验分组信息
        sql = "select * from exp_group where exp_id=%s"%(exp_id)
        self.cur.execute(sql)
        cols = [x[0] for x in self.cur.description]
        group_info = pd.DataFrame(self.cur.fetchall(),columns=cols)
        group_info = group_info.groupby(['exp_id','group_id','comment']).apply(lambda x:list(x['bucket_id']))

        # 实验样本分组sql
        # 实验层信息
        layer_id = exp_info['layer'].iloc[0]
        sql = "select * from layers where layer_id=%s"%(layer_id)
        self.cur.execute(sql)
        cols = [x[0] for x in self.cur.description]
        layer_info = pd.DataFrame(self.cur.fetchall(),columns=cols)
        
        # 实验域信息
        field_id = exp_info['field'].iloc[0]
        sql = "select * from fields where field_id=%s"%(field_id)
        self.cur.execute(sql)
        cols = [x[0] for x in self.cur.description]
        field_info = pd.DataFrame(self.cur.fetchall(),columns=cols)

        # 实验层hash
        hash_code = "conv(substring(md5(concat(%s,'_',%s)),1,5),16,10)"%(layer_info['hash_col'].iloc[0],layer_info['layer_id'].iloc[0])
        

        # 生成case when sql语句
        sql_when = []
        for i in range(len(group_info)):
            buckets = str(tuple(group_info.iloc[i]))
            group_id = group_info.index[i][1]
            
            tmp = "when (%s%%%s+1 in %s) and (%s) then %s"%(hash_code,layer_info['bucket_num'].iloc[0],buckets,field_info['con'].iloc[0],group_id)
            sql_when.append(tmp)
        sql_when = '\n'.join(sql_when)


        exp_name = exp_info['exp_name'].iloc[0]
        sql= """
insert overwrite table ml_abtest__usrid_group_info_ext partition(exp_id=%s) 
select
*
from
    (
    select 
        usrid,
        '%s' as exp_name,
        %s as layer_id,
        case
        %s
        end as group_id,
        ext_col1,
        ext_col2,
        ext_col3,
        ext_col4,
        ext_col5
    from 
        (%s) t 
    ) t
where group_id is not null
"""%(exp_id,exp_name,layer_id,sql_when,exp_info['pre_sql'].iloc[0])

        return exp_info,group_info,sql

    
    def used_buckets(self):
        """
        purpose:
            当前激活实验占占用的实验桶信息
        return:
            used_buckets: df 已激活实验占用的桶
        """

        used_buckets = {}

        sql = """
        select
        t0.field,
        t0.layer,
        t1.bucket_id,
        t0.exp_id
        from
        (select exp_id,field,layer from experiment where valid_flg=1) t0
        inner JOIN 
        (select exp_id,bucket_id from exp_group) t1
        on t0.exp_id=t1.exp_id; 
        """
        self.cur.execute(sql)
        cols = [x[0] for x in self.cur.description]
        used_buckets = pd.DataFrame(self.cur.fetchall(),columns=cols)
        return used_buckets


    def over_view(self):
        """
        purpose:
            展示整体实验分布情况
        """
        #plt.figure(figsize=(3, 30))
        text = [] 
        fig, ax = plt.subplots(figsize = (30, 3))
        ax.axis([0,100,0,10])
        used_buckets = self.used_buckets()
        
        sql = "select layer_id,bucket_num from layers where field_id=1"
        self.cur.execute(sql)
        layers = self.cur.fetchall()
        for i,layer in enumerate(layers):
            tmp = ''
            layer_id =layer[0]
            bucket_num = layer[1]
            for j in range(bucket_num):
                bucket_id = j+1
                if len(used_buckets.loc[(used_buckets.layer==layer_id) & (used_buckets.bucket_id==bucket_id)])>0:
                    rect = mpathes.Rectangle((1*j,1*i),0.8,0.8,color = 'r')
                    tmp+='+'
                else:
                    rect = mpathes.Rectangle((1*j,1*i),0.8,0.8,color = 'g')
                    tmp+='-'
                ax.add_patch(rect)
            text.append('%s | %s'%(layer_id,tmp))
        #plt.show()
        plt.savefig('./static/img/overview.png')
        plt.close()
        print('\n'.join(text))


    def coupon_eval(self):
        """
        purpose:
            生成票券总体统计sql脚本
        """

        # 领取型票券相关指标
        param = {
        'push_job_name': 'cksy20210511'                      #决策引擎pushjob名称
        ,'cupon_id':'3102021042675206'                       #票券id
        ,'push_time': "hp_settle_dt in ('2021-05-11')"       #决策引擎push时间
        ,'coupon_get_time': "hp_settle_dt>=20210511"         #票券领取时间
        ,'coupon_use_time': "hp_settle_dt>=20210511"         #票券核销时间
        ,'coupon_overdue_time': "hp_settle_dt>=20210511"     #票券失效时间
        }

        sql_push = """
            --抽奖资格发放
            select 
                usr_id,
                substring(update_time,1,19) as ts
            from 
                (select *,upload_day as hp_settle_dt from tbl_des_cust_chance_update ) t
            where 
                1=1
                and job_type='%s' 
                and %s
        """%(param['push_job_name'],param['push_time'])

        sql_coupon_get  ="""
            --票券领取
            select 
                entity_id as usr_id,
                substring(rec_crt_ts,1,19) as ts
            from 
                viw_chhis_coupon_opera_flow_for_tsg 
            where 
                1=1
                and discount_id='%s' 
                and opera_tp='DL'
                and %s
        """%(param['cupon_id'],param['coupon_get_time'])

        sql_coupon_use = """
            --票券核销
            select 
                usr_id,
                from_unixtime(unix_timestamp(concat(trans_dt,trans_ts),'yyyyMMddHHmmss'), 'yyyy-MM-dd HH:mm:ss') as ts
            from 
                tbl_biysf_qp_app_vds_reduce_d_dtl 
            where 
                1=1
                and activity_id='%s' 
                and %s      
        """%(param['cupon_id'],param['coupon_use_time'])

        sql_coupon_overdue = """
            --票券过期
            select 
                entity_id as usr_id,
                substring(rec_crt_ts,1,19) as ts
            from 
                viw_chhis_coupon_opera_flow_for_tsg 
            where 
                1=1
                and discount_id='%s' 
                and opera_tp='RT'
                and %s
        """%(param['cupon_id'],param['coupon_overdue_time'])


        eval_sql = """
        select
            count(*),                 --样本量
            sum(coupon_get_flg),      --领取量
            sum(coupon_use_flg),      --承兑量
            sum(coupon_overdue_flg)   --失效量
        from
        (select
            t5.usr_id,
            if(t2.ts>t1.ts,1,0) as coupon_get_flg,                      --领取标志
            if(t2.ts>t1.ts and t3.ts>t2.ts,1,0) as coupon_use_flg,      --领取后承兑标志
            if(t2.ts>t1.ts and t4.ts>t2.ts,1,0) as coupon_overdue_flg,  --领取后失效标志
            t5.prob_1
        from
            (
                %s
            ) t1  --抽奖资格发放
        left join
            (
                %s
            ) t2 --票券领取
        on t1.usr_id=t2.usr_id
        left join
            (
                %s
            ) t3 --票券核销
        on t1.usr_id=t3.usr_id
        left join
            (
                %s
            ) t4 --票券过期
        on t1.usr_id=t4.usr_id
        right join 
            (
                extra_table
            ) t5 --待拼接样本表
        on t1.usr_id=t5.usr_id
        ) t
        ;
        """%(sql_push,sql_coupon_get,sql_coupon_use,sql_coupon_overdue)

        return eval_sql

    def active_eval(self):
        
        # 交易活跃活跃
        param = {
        'trans_type': 'credit_repay'
        ,'trans_time':"hp_settle_dt>=20210511"
        }
        eval_sql = """
        select
            count(*),        --样本数
            sum(active_flg)  --活跃样本数
        from
            (
                select 
                    t2.usr_id,
                    if(t1.usr_id is not null,1,0) as active_flg
                from
                    (select 
                        cdhd_usr_id as usr_id
                    from 
                        tbl_upf_app_trans_d_incr_dtl 
                    where 
                        1=1
                        and %s
                        and trans_label='%s'
                    group by cdhd_usr_id
                    ) t1  --活跃用户表
                right_join
                    (
                        extra_table
                    )  t2  --实验样本表
                on t1.usr_id=t2.usr_id
            ) t
        
        """%(param['trans_time'],param['trans_type'])

        return eval_sql


    def coupon_eval_by_group(self,eval_sql,exp_id):
        """
        purpose:
            abtest分组统计对比
        input:
            exp_id: int 实验id  
        """
        _,_,sample = self.get_exp(exp_id)
        sql = eval_sql.replace('extra_table',sample)
        return sql

    def exp_only(self,exp_id):
        """
        purpose:
            实验流量独占
        input:
            exp_id
        """
        return

    

    


