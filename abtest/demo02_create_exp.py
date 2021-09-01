import abtest
import importlib
importlib.reload(abtest)
import pandas as pd
import pymysql
import sys
mysql_host='localhost'
mysql_port=3306
passwd='123456'
user='root'
db='ABtest'
conn = pymysql.connect(host=mysql_host, port=mysql_port, user=user, passwd=passwd, db=db,autocommit = True)
conn.ping()
cur = conn.cursor()
ab = abtest.abtest(cur)





def group_clean(group):
    group = group.split(';')
    group_id = int(group[0].strip())
    
    if '~' in group[1]:
        s = int(group[1].split('~')[0].strip())
        e = int(group[1].split('~')[1].strip())
        group_buckets = list(range(s,e+1))
    else:
        group_buckets = [int(x.strip()) for x in group[1].spit(',')]
    
    group_comment = group[2]

    return group_id,group_buckets,group_comment

if len(sys.argv)==11:
    exp_id = int(sys.argv[1])

    exp_info = {}
    exp_info['exp_name'] = sys.argv[2]
    exp_info['department'] = sys.argv[3]
    exp_info['admin'] = sys.argv[4]
    exp_info['field'] = int(sys.argv[5])
    exp_info['layer'] = int(sys.argv[6])
    exp_info['create_time'] = sys.argv[7]
    exp_info['comment'] = sys.argv[8]
    exp_info['pre_sql'] = sys.argv[9].replace('\'','\\\'').replace('\"','\\\"')

    group_info = {}
    tmp = sys.argv[10]
    for group in tmp.split('\n'):
        if len(group.strip())>0:
            group_id,group_buckets,group_comment = group_clean(group.strip())
            group_info[group_id] = (group_buckets,group_comment)

    ab.create_exp(exp_id,exp_info,group_info)  #创建实验
    
    quit()



###########################################
# 2. 创建实验和分组
# 用户可以通过参数配置新增实验
# 新增实验的样本桶不能和当前占用的样本桶发生冲突
# 新建实验后不会立即占用所配置的样本桶，只有当实验激活时该实验的样本桶被锁定，无法被其他实验占用知道实验停止
# 当发生样本桶冲突时需要用户自己对通进行调整

# 功能测试:
# 实验参数化配置, 实验冲突检验(实验层冲突，实验样本桶冲突),实验分桶信息保存，实验启停
###########################################


'''
##########################################################################
##########################################################################


#实验id
exp_id = 1
#实验基本信息
exp_info = {
'exp_name': '湖南分公司用户流失挽回实验-模型实验'      #实验名称
,'department':'技术支持'                   #实验部门
,'admin':'俞晓强'                          #实验负责人
,'field':1                                #实验域
,'layer':1                                #实验层
,'create_time':'2021-07-15 00:00:00'      #实验创建时间
,'comment': '湖南分公司用户流失挽回实验，分析模型挽回效果'   #说明
,'pre_sql':"""
select
    usrid
from
    ml_credit_repay_custlost__predict
where
    part_dt = 20210701
    and rank <= 60000
"""#前置样本选择
}

# 实验分组 组别:(分桶id,分组说明)
group_info = {
1:(list(range(1,21)),'对照组,不做任何营销实验')
,2:(list(range(21,91)),'实验组，发放6.2元还款券')
}

ab.create_exp(exp_id,exp_info,group_info)  #创建实验

##########################################################################
##########################################################################

#实验id
exp_id = 2
#实验基本信息
exp_info = {
'exp_name': '湖南分公司用户流失挽回实验-规则营销实验'      #实验名称
,'department':'技术支持'                    #实验部门
,'admin':'俞晓强'                           #实验负责人
,'field':1                                 #实验域
,'layer':1                                 #实验层
,'create_time':'2021-07-15 00:00:00'       #实验创建时间
,'comment': '湖南分公司用户流失挽回实验，规则为选择过去一个月未使用信用卡还款的用户'   #说明
,'pre_sql':"""
select 
t1.usrid
from
    (select 
        cdhd_usr_id as usrid 
    from 
        tbl_upf_app_trans_d_incr_dtl 
    where 
        substring(hp_settle_dt,1,6) in (
            substr(regexp_replace(add_months(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'),-2),'-',''),1,6),
            substr(regexp_replace(add_months(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'),-3),'-',''),1,6)
            )  
        and trans_label='credit_repay'
    group by cdhd_usr_id
    ) t1
inner join
    (
    select
        loc_enc.usr_id as usr_id
    from
        (
        select
            recent.usr_id,
            recent.fre_domin_id,
            recent.fre_branch
        from
            (
            select
                cdhd_usr_id as usr_id,
                fre_domin_id,
                fre_branch,
                row_number() over (partition by cdhd_usr_id order by hp_settle_mon desc) as r
            from
                tbl_chhis_usr_frequently_gps_inf_m_dtl
            where
                hp_settle_mon in (
                    from_unixtime(unix_timestamp(), 'yyyyMM'),
                    substr(regexp_replace(date_sub(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'),30),'-',''),1,6),
                    substr(regexp_replace(date_sub(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'),60),'-',''),1,6)
                    )
            ) recent
        where
            recent.r = 1
        ) loc_enc

        left outer join 
        (select region_cd, city, branch from tbl_qht_gps_region_inf_dim_all_scl) decr 
        on loc_enc.fre_domin_id = decr.region_cd
        
        left outer join (
        select
            '0800013600' as branch_id,
            '安徽' as branch
        union all
        select
            '0800011000' as branch_id,
            '北京' as branch
        union all
        select
            '0800012220' as branch_id,
            '大连' as branch
        union all
        select
            '0800013900' as branch_id,
            '福建' as branch
        union all
        select
            '0800018200' as branch_id,
            '甘肃' as branch
        union all
        select
            '0800015800' as branch_id,
            '广东' as branch
        union all
        select
            '0800016100' as branch_id,
            '广西' as branch
        union all
        select
            '0800017000' as branch_id,
            '贵州' as branch
        union all
        select
            '0800016400' as branch_id,
            '海南' as branch
        union all
        select
            '0800011200' as branch_id,
            '河北' as branch
        union all
        select
            '0800014900' as branch_id,
            '河南' as branch
        union all
        select
            '0800012600' as branch_id,
            '黑龙江' as branch
        union all
        select
            '0800015210' as branch_id,
            '湖北' as branch
        union all
        select
            '0800015500' as branch_id,
            '湖南' as branch
        union all
        select
            '0800012400' as branch_id,
            '吉林' as branch
        union all
        select
            '0800013000' as branch_id,
            '江苏' as branch
        union all
        select
            '0800014200' as branch_id,
            '江西' as branch
        union all
        select
            '0800012210' as branch_id,
            '辽宁' as branch
        union all
        select
            '0800011900' as branch_id,
            '内蒙古' as branch
        union all
        select
            '0800013320' as branch_id,
            '宁波' as branch
        union all
        select
            '0800018700' as branch_id,
            '宁夏' as branch
        union all
        select
            '0800014520' as branch_id,
            '青岛' as branch
        union all
        select
            '0800018500' as branch_id,
            '青海' as branch
        union all
        select
            '0800013930' as branch_id,
            '厦门' as branch
        union all
        select
            '0800014500' as branch_id,
            '山东' as branch
        union all
        select
            '0800011600' as branch_id,
            '山西' as branch
        union all
        select
            '0800017900' as branch_id,
            '陕西' as branch
        union all
        select
            '0800012900' as branch_id,
            '上海' as branch
        union all
        select
            '0800015840' as branch_id,
            '深圳' as branch
        union all
        select
            '0800016500' as branch_id,
            '四川' as branch
        union all
        select
            '0800011100' as branch_id,
            '天津' as branch
        union all
        select
            '0800017700' as branch_id,
            '西藏' as branch
        union all
        select
            '0800018800' as branch_id,
            '新疆' as branch
        union all
        select
            '0800017310' as branch_id,
            '云南' as branch
        union all
        select
            '0800013310' as branch_id,
            '浙江' as branch
        union all
        select
            '0800016530' as branch_id,
            '重庆' as branch
        ) as branch_cor on branch_cor.branch_id = loc_enc.fre_branch
    where
        (decr.city like "%长沙%")
        or (decr.city is null and branch_cor.branch like "%湖南%")
    group by
        loc_enc.usr_id
    ) t2 
    on t1.usrid = t2.usr_id

    left join
    (select 
        cdhd_usr_id as usrid 
    from 
        tbl_upf_app_trans_d_incr_dtl 
    where 
        substring(hp_settle_dt,1,6) in (
            substr(regexp_replace(add_months(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'),-1),'-',''),1,6)
            )  
        and trans_label='credit_repay'
    group by cdhd_usr_id
    ) t3
    on t1.usrid = t3.usrid

    where
        t3.usrid is null
""".replace('\'','\\\'').replace('\"','\\\"')   #前置样本选择
}

# 实验分组 组别:(分桶id,分组说明)
group_info = {
1:(list(range(91,96)),'对照组,不做任何营销实验')
,2:(list(range(96,101)),'实验组，发放6.2元还款券')
}

ab.create_exp(exp_id,exp_info,group_info)  #创建实验



##########################################################################
##########################################################################
#实验id
exp_id = 3
#实验基本信息
exp_info = {
'exp_name': '湖南分公司用户流失挽回实验-随机营销实验'      #实验名称
,'department':'技术支持'                    #实验部门
,'admin':'俞晓强'                           #实验负责人
,'field':1                                 #实验域
,'layer':1                                 #实验层
,'create_time':'2021-07-15 00:00:00'       #实验创建时间
,'comment': '湖南分公司用户流失挽回实验，随机选择用户'   #说明
,'pre_sql':"""
select 
t1.usrid
from
    (select 
        cdhd_usr_id as usrid 
    from 
        tbl_upf_app_trans_d_incr_dtl 
    where 
        substring(hp_settle_dt,1,6) in (
            substr(regexp_replace(add_months(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'),-1),'-',''),1,6),
            substr(regexp_replace(add_months(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'),-2),'-',''),1,6),
            substr(regexp_replace(add_months(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'),-3),'-',''),1,6)
            )  
        and trans_label='credit_repay'
    group by cdhd_usr_id
    ) t1
inner join
    (
    select
        loc_enc.usr_id as usr_id
    from
        (
        select
            recent.usr_id,
            recent.fre_domin_id,
            recent.fre_branch
        from
            (
            select
                cdhd_usr_id as usr_id,
                fre_domin_id,
                fre_branch,
                row_number() over (partition by cdhd_usr_id order by hp_settle_mon desc) as r
            from
                tbl_chhis_usr_frequently_gps_inf_m_dtl
            where
                hp_settle_mon in (
                    from_unixtime(unix_timestamp(), 'yyyyMM'),
                    substr(regexp_replace(date_sub(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'),30),'-',''),1,6),
                    substr(regexp_replace(date_sub(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'),60),'-',''),1,6)
                    )
            ) recent
        where
            recent.r = 1
        ) loc_enc

        left outer join 
        (select region_cd, city, branch from tbl_qht_gps_region_inf_dim_all_scl) decr 
        on loc_enc.fre_domin_id = decr.region_cd
        
        left outer join (
        select
            '0800013600' as branch_id,
            '安徽' as branch
        union all
        select
            '0800011000' as branch_id,
            '北京' as branch
        union all
        select
            '0800012220' as branch_id,
            '大连' as branch
        union all
        select
            '0800013900' as branch_id,
            '福建' as branch
        union all
        select
            '0800018200' as branch_id,
            '甘肃' as branch
        union all
        select
            '0800015800' as branch_id,
            '广东' as branch
        union all
        select
            '0800016100' as branch_id,
            '广西' as branch
        union all
        select
            '0800017000' as branch_id,
            '贵州' as branch
        union all
        select
            '0800016400' as branch_id,
            '海南' as branch
        union all
        select
            '0800011200' as branch_id,
            '河北' as branch
        union all
        select
            '0800014900' as branch_id,
            '河南' as branch
        union all
        select
            '0800012600' as branch_id,
            '黑龙江' as branch
        union all
        select
            '0800015210' as branch_id,
            '湖北' as branch
        union all
        select
            '0800015500' as branch_id,
            '湖南' as branch
        union all
        select
            '0800012400' as branch_id,
            '吉林' as branch
        union all
        select
            '0800013000' as branch_id,
            '江苏' as branch
        union all
        select
            '0800014200' as branch_id,
            '江西' as branch
        union all
        select
            '0800012210' as branch_id,
            '辽宁' as branch
        union all
        select
            '0800011900' as branch_id,
            '内蒙古' as branch
        union all
        select
            '0800013320' as branch_id,
            '宁波' as branch
        union all
        select
            '0800018700' as branch_id,
            '宁夏' as branch
        union all
        select
            '0800014520' as branch_id,
            '青岛' as branch
        union all
        select
            '0800018500' as branch_id,
            '青海' as branch
        union all
        select
            '0800013930' as branch_id,
            '厦门' as branch
        union all
        select
            '0800014500' as branch_id,
            '山东' as branch
        union all
        select
            '0800011600' as branch_id,
            '山西' as branch
        union all
        select
            '0800017900' as branch_id,
            '陕西' as branch
        union all
        select
            '0800012900' as branch_id,
            '上海' as branch
        union all
        select
            '0800015840' as branch_id,
            '深圳' as branch
        union all
        select
            '0800016500' as branch_id,
            '四川' as branch
        union all
        select
            '0800011100' as branch_id,
            '天津' as branch
        union all
        select
            '0800017700' as branch_id,
            '西藏' as branch
        union all
        select
            '0800018800' as branch_id,
            '新疆' as branch
        union all
        select
            '0800017310' as branch_id,
            '云南' as branch
        union all
        select
            '0800013310' as branch_id,
            '浙江' as branch
        union all
        select
            '0800016530' as branch_id,
            '重庆' as branch
        ) as branch_cor on branch_cor.branch_id = loc_enc.fre_branch
    where
        (decr.city like "%长沙%")
        or (decr.city is null and branch_cor.branch like "%湖南%")
    group by
        loc_enc.usr_id
    ) t2 
    on t1.usrid = t2.usr_id
""".replace('\'','\\\'').replace('\"','\\\"')   #前置样本选择

}

# 实验分组 组别:(分桶id,分组说明)
group_info = {
1:(list(range(91,96)),'对照组,不做任何营销实验')
,2:(list(range(96,101)),'实验组，发放6.2元还款券')
}

ab.create_exp(exp_id,exp_info,group_info)  #创建实验



##########################################################################
# 账户服务团队潜客挖掘-模型
##########################################################################

#实验id
exp_id = 4
#实验基本信息
exp_info = {
'exp_name': '账户服务团队信用卡还款潜客挖掘实验-模型营销实验'      #实验名称
,'department':'技术支持'                   #实验部门
,'admin':'俞晓强'                          #实验负责人
,'field':1                                #实验域
,'layer':2                                #实验层
,'create_time':'2021-07-25 00:00:00'      #实验创建时间
,'comment': '账户服务团队信用卡还款潜客挖掘实验，模型选择用户营销'   #说明
,'pre_sql':"""
select 
t1.usrid,
if(t3.usrid is null,0,1) as ext_col1  --是否绑定信用卡标记
from
    (select usrid from ml_credit_repay_potential_cust__predict_day where part_dt=20210701 where score>0.5 group by usrid) t1  --根据模型评分是筛选
left join
    (select cdhd_usr_id as usrid from tbl_upf_app_trans_d_incr_dtl where hp_settle_dt<20210701 group by cdhd_usr_id) t2  --剔除已经还过信用卡的用户
on t1.usrid=t2.usrid
left join 
    (
    select
        usr_id as usrid
    from
        viw_uchis_ucbiz_bind_card_inf_for_tsg
    where 
        bind_tp in ('02','03')
    group by usr_id
    ) t3 --是否绑信用卡标记
where t2.usrid is null   --剔除非首还用户
""".replace('\'','\\\'').replace('\"','\\\"')   #前置样本选择
}

# 实验分组 组别:(分桶id,分组说明)
group_info = {
1:(list(range(1,7)),'实验组1,push+短信+5元券+个性化文案+还款日发送')
,2:(list(range(7,13)),'实验组2,仅push+5元券+个性化文案+还款日发送')
,3:(list(range(13,19)),'实验组3,仅push+5元券+通用旧文案+批量发送')
,4:(list(range(19,25)),'实验组4,仅push+5元券+个性化文案+批量发送')
,5:(list(range(25,31)),'实验组5,仅push+10元券+个性化文案+批量发送')
,6:(list(range(31,37)),'实验组6,仅push+15元券+个性化文案+批量发送')
,7:(list(range(37,51)),'对照组，不做营销')
}

ab.create_exp(exp_id,exp_info,group_info)  #创建实验


##########################################################################
# 账户服务团队潜客挖掘-规则
##########################################################################
#实验id
exp_id = 5
#实验基本信息
exp_info = {
'exp_name': '账户服务团队信用卡还款潜客挖掘实验-规则营销实验'      #实验名称
,'department':'技术支持'                   #实验部门
,'admin':'俞晓强'                          #实验负责人
,'field':1                                #实验域
,'layer':2                                #实验层
,'create_time':'2021-07-25 00:00:00'      #实验创建时间
,'comment': '账户服务团队信用卡还款潜客挖掘实验，规则选择用户营销'   #说明
,'pre_sql':"""
select 
t1.usrid,
1 as ext_col1  --是否绑定信用卡标记
from
    (select
        usr_id as usrid
    from
        viw_uchis_ucbiz_bind_card_inf_for_tsg
    where 
        bind_tp in ('02','03')
    group by usr_id) t1  --根据模型评分是筛选
left join
    (select cdhd_usr_id as usrid from tbl_upf_app_trans_d_incr_dtl where hp_settle_dt<20210701 group by cdhd_usr_id) t2  --剔除已经还过信用卡的用户
on t1.usrid=t2.usrid
where t2.usrid is null   --剔除非首还用户
""".replace('\'','\\\'').replace('\"','\\\"')   #前置样本选择
}

# 实验分组 组别:(分桶id,分组说明)
group_info = {
1:(list(range(51,95)),'实验组1,push+5元券+个性化文案+还款日发送')
,2:(list(range(95,101)),'对照组，不做营销')
}

ab.create_exp(exp_id,exp_info,group_info)  #创建实验





##########################################################################
# 账户服务团队信用卡还款活跃用户流失模型--模型筛选
##########################################################################
#实验id
exp_id=6
#实验基本信息
exp_info = {
'exp_name': '账户服务团队信用卡还款活跃用户流失-模型实验'      #实验名称
,'department':'技术支持'                   #实验部门
,'admin':'俞晓强'                          #实验负责人
,'field':1                                #实验域
,'layer':3                                #实验层
,'create_time':'2021-08-23 00:00:00'      #实验创建时间
,'comment': '账户服务团队信用卡还款活跃用户流失-模型筛选用户'   #说明
,'pre_sql':"""
select 
t1.usrid,
null as ext_col1,  
t4.ext_col2 as ext_col2,    --最近还款日
t1.score as ext_col3        --模型评分
from
    (select usrid,score from ml_all_usr_crm__credit_repay_predict_result where part_dt=${model_date} and model='huoyueliushi' and score>0.46 ) t1  --根据模型评分是筛选
inner join
    (select 
        cdhd_usr_id as usrid 
    from 
        tbl_upf_app_trans_d_incr_dtl 
    where 
        hp_settle_dt between regexp_replace(date_sub(${today_date2},30),'-','') and ${today_date}
        and trans_label='credit_repay' 
    group by 
        cdhd_usr_id) t2  --过去30天发生过信用卡还款的用户
on t1.usrid=t2.usrid

left join
    (
    select
        usrid
        ,min(case when add_months(repay_deadline,month_diff)> ${today_date2} then add_months(repay_deadline,month_diff)
                    when add_months(repay_deadline,month_diff)<= ${today_date2} then add_months(repay_deadline,month_diff+1)
                    else null end) as ext_col2	  --最近还款日
    from 
        (
        select 
            usrid
            ,pri_acct_no_sm3
            ,from_unixtime(unix_timestamp(max(repay_deadline),'yyyyMMdd'), "yyyy-MM-dd") as repay_deadline
            ,cast(months_between( trunc(${today_date2},'MM'),
                                    trunc(from_unixtime(unix_timestamp(max(repay_deadline),'yyyyMMdd'), "yyyy-MM-dd"),'MM')) as int) as month_diff   --账单还款日距离本月月数
        from 
            ml_usr_crm__repaydate_month 
        group by 
            usrid,pri_acct_no_sm3
        ) t
    group by 
        usrid
    ) t4
on t1.usrid=t4.usrid

left join 
    (
    select
        distinct usr_id as usrid
    from
        viw_uchis_ucbiz_cdhd_bas_inf_for_tsg
    where
        substr(bin(func_bmp), -20, 1) = '1'
        and usr_id is not null
        and trim(usr_id) <> ''
    ) reg 
on t1.usrid = reg.usrid

where reg.usrid is null  --剔除海外用户

""".replace('\'','\\\'').replace('\"','\\\"')   #前置样本选择
}

# 实验分组 组别:(分桶id,分组说明)
group_info = {
1:(list(range(1,81)),'实验组1,push+5元券+还款日发送')
,2:(list(range(81,86)),'实验组1,push+10元券+还款日发送')
,3:(list(range(86,91)),'实验组1,push+15元券+还款日发送')
,4:(list(range(91,99)),'实验组2,不做营销')
}

ab.create_exp(exp_id,exp_info,group_info)  #创建实验





##########################################################################
# 账户服务团队信用卡还款活跃用户流失模型--随机筛选
##########################################################################
#实验id
exp_id=7
#实验基本信息
exp_info = {
'exp_name': '账户服务团队信用卡还款活跃用户流失-随机实验'      #实验名称
,'department':'技术支持'                   #实验部门
,'admin':'俞晓强'                          #实验负责人
,'field':1                                #实验域
,'layer':3                                #实验层
,'create_time':'2021-08-23 00:00:00'      #实验创建时间
,'comment': '账户服务团队信用卡还款活跃用户流失-随机实验'   #说明
,'pre_sql':"""
select 
t1.usrid,
null as ext_col1,  
t4.ext_col2 as ext_col2,    --最近还款日
null as ext_col3        --模型评分
from
    (select 
        cdhd_usr_id as usrid 
    from 
        tbl_upf_app_trans_d_incr_dtl 
    where 
        hp_settle_dt between regexp_replace(date_sub(${today_date2},30),'-','') and ${today_date}
        and trans_label='credit_repay' 
    group by 
        cdhd_usr_id) t1  --过去30天发生过信用卡还款的用户

left join
    (
    select
        usrid
        ,min(case when add_months(repay_deadline,month_diff)> ${today_date2} then add_months(repay_deadline,month_diff)
                    when add_months(repay_deadline,month_diff)<= ${today_date2} then add_months(repay_deadline,month_diff+1)
                    else null end) as ext_col2	  --最近还款日
    from 
        (
        select 
            usrid
            ,pri_acct_no_sm3
            ,from_unixtime(unix_timestamp(max(repay_deadline),'yyyyMMdd'), "yyyy-MM-dd") as repay_deadline
            ,cast(months_between( trunc(${today_date2},'MM'),
                                    trunc(from_unixtime(unix_timestamp(max(repay_deadline),'yyyyMMdd'), "yyyy-MM-dd"),'MM')) as int) as month_diff   --账单还款日距离本月月数
        from 
            ml_usr_crm__repaydate_month 
        group by 
            usrid,pri_acct_no_sm3
        ) t
    group by 
        usrid
    ) t4
on t1.usrid=t4.usrid

left join 
    (
    select
        distinct usr_id as usrid
    from
        viw_uchis_ucbiz_cdhd_bas_inf_for_tsg
    where
        substr(bin(func_bmp), -20, 1) = '1'
        and usr_id is not null
        and trim(usr_id) <> ''
    ) reg 
on t1.usrid = reg.usrid

where reg.usrid is null  --剔除海外用户

""".replace('\'','\\\'').replace('\"','\\\"')   #前置样本选择
}

# 实验分组 组别:(分桶id,分组说明)
group_info = {
1:([99],'实验组1,push+5元券+还款日发送')
,2:([100],'实验组2,不做营销')
}

ab.create_exp(exp_id,exp_info,group_info)  #创建实验





##########################################################################
# 账户服务团队信用卡还款浅睡用户流失模型--全量
##########################################################################
#实验id
exp_id=8
#实验基本信息
exp_info = {
'exp_name': '账户服务团队信用卡还款浅睡用户流失模型--全量营销'      #实验名称
,'department':'技术支持'                   #实验部门
,'admin':'俞晓强'                          #实验负责人
,'field':1                                #实验域
,'layer':4                                #实验层
,'create_time':'2021-08-27 00:00:00'      #实验创建时间
,'comment': '账户服务团队信用卡还款浅睡用户流失模型--全量营销'   #说明
,'pre_sql':"""
select 
    usrid
	,ext_col1 as ext_col1   
	,ext_col2 as ext_col2   --最近还款日
	,ext_col3 as ext_col3   --模型评分
	,ext_col4 as ext_col4   --是否模型定义浅睡用户
	,null as ext_col5
from 
    (
    select 
    t1.usrid,
    null as ext_col1,  
    t4.ext_col2 as ext_col2,    --最近还款日
    t3.score as ext_col3,        --模型评分
    if(model_flg.usrid is not null,1,0)  as ext_col4  --是否模型定义浅睡用户
    from
        (select 
            cdhd_usr_id as usrid 
        from 
            tbl_upf_app_trans_d_incr_dtl 
        where 
            hp_settle_dt between regexp_replace(date_sub(${today_date2},90),'-','')  and  ${today_date}
            and trans_label='credit_repay' 
        group by 
            cdhd_usr_id) t1  --过去90天发生过信用卡还款的用户
    
    left join
        (select 
            cdhd_usr_id as usrid 
        from 
            tbl_upf_app_trans_d_incr_dtl 
        where 
            hp_settle_dt between regexp_replace(date_sub(${today_date2},60),'-','')  and  ${today_date}
            and trans_label='credit_repay' 
        group by 
            cdhd_usr_id) model_flg  --过去60天发生过信用卡还款的用户
    on t1.usrid = model_flg.usrid
    
    left join
        (select 
            cdhd_usr_id as usrid 
        from 
            tbl_upf_app_trans_d_incr_dtl 
        where 
            hp_settle_dt between regexp_replace(date_sub(${today_date2},30),'-','') and ${today_date}
            and trans_label='credit_repay' 
        group by 
            cdhd_usr_id) t2  --过去30天发生过信用卡还款的用户
    on t1.usrid=t2.usrid
    
    left join
        (select usrid,score from ml_all_usr_crm__credit_repay_predict_result where part_dt=${model_date} and model='qianshuiliushi') t3  --根据模型评分是筛选
    on t1.usrid=t3.usrid
    
    left join
        (
        select
            usrid
            ,min(case when add_months(repay_deadline,month_diff)> ${today_date2} then add_months(repay_deadline,month_diff)
                        when add_months(repay_deadline,month_diff)<= ${today_date2} then add_months(repay_deadline,month_diff+1)
                        else null end) as ext_col2	  --最近还款日
        from 
            (
            select 
                usrid
                ,pri_acct_no_sm3
                ,from_unixtime(unix_timestamp(max(repay_deadline),'yyyyMMdd'), "yyyy-MM-dd") as repay_deadline
                ,cast(months_between( trunc(${today_date2},'MM'),
                                        trunc(from_unixtime(unix_timestamp(max(repay_deadline),'yyyyMMdd'), "yyyy-MM-dd"),'MM')) as int) as month_diff   --账单还款日距离本月月数
            from 
                ml_usr_crm__repaydate_month 
            group by 
                usrid,pri_acct_no_sm3
            ) t
        group by 
            usrid
        ) t4
    on t1.usrid=t4.usrid
    
    left join 
        (
        select
            distinct usr_id as usrid
        from
            viw_uchis_ucbiz_cdhd_bas_inf_for_tsg
        where
            substr(bin(func_bmp), -20, 1) = '1'
            and usr_id is not null
            and trim(usr_id) <> ''
        ) reg 
    on t1.usrid = reg.usrid
    where reg.usrid is null  --剔除海外用户 
    and t2.usrid is null
    ) t 

""".replace('\'','\\\'').replace('\"','\\\"')   #前置样本选择
}

# 实验分组 组别:(分桶id,分组说明)
group_info = {
1:(list(range(1,96)),'实验组1,push+5元券+还款日发送')
,2:([96,97,98,99,100],'实验组2,不做营销')
}

ab.create_exp(exp_id,exp_info,group_info)  #创建实验

'''