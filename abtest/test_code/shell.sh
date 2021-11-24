cd /Users/yu/Desktop/work/unionpay/项目/abtest/

#####################创建实验层
# layer_id=1
# field_id=1
# hash_col='usrid'
# hash_method='conv(substring(md5(usrid),1,5),16,10)'
# bucket_num=100
# comment='湖南分公司信用卡还款用户流失换回实验层，直接发送还款券，分析模型挽回效果'
# creator='俞晓强'
# available_flg=1
# create_time='2021-07-15 00:00:00'

# python3 demo01_create_layer.py "$layer_id" "$field_id" "$hash_col" "$hash_method" "$bucket_num"  "$comment" "$creator" "$available_flg" "$create_time" 


# ####################创建实验

# exp_id=9 
# exp_name='账户服务深睡用户唤醒-全量营销'
# department='技术支持'
# admin='俞晓强'
# field=1
# layer=5
# create_time='2021-07-30 00:00:00' 
# comment='账户服务深睡用户唤醒-全量营销'
# pre_sql="""
# select 
#     usrid
# 	,ext_col1 as ext_col1   
# 	,ext_col2 as ext_col2   --最近还款日
# 	,ext_col3 as ext_col3   --模型评分
# 	,ext_col4 as ext_col4   --是否模型定义深睡用户
# 	,null as ext_col5
# from 
#     (
#     select 
#     t1.usrid,
#     null as ext_col1,  
#     t4.ext_col2 as ext_col2,    --最近还款日
#     t3.score as ext_col3,        --模型评分
#     if(model_flg.usrid is not null,1,0)  as ext_col4  --是否模型定义浅睡用户
#     from
#         (select 
#             cdhd_usr_id as usrid 
#         from 
#             tbl_upf_app_trans_d_incr_dtl 
#         where 
#             hp_settle_dt between regexp_replace(date_sub(${today_date2},365),'-','')  and  ${today_date}
#             and trans_label='credit_repay' 
#         group by 
#             cdhd_usr_id) t1  --过去90天发生过信用卡还款的用户
    
#     left join
#         (select 
#             cdhd_usr_id as usrid 
#         from 
#             tbl_upf_app_trans_d_incr_dtl 
#         where 
#             hp_settle_dt between regexp_replace(date_sub(${today_date2},180),'-','')  and  ${today_date}
#             and trans_label='credit_repay' 
#         group by 
#             cdhd_usr_id) model_flg  --过去180天发生过信用卡还款的用户
#     on t1.usrid = model_flg.usrid
    
#     left join
#         (select 
#             cdhd_usr_id as usrid 
#         from 
#             tbl_upf_app_trans_d_incr_dtl 
#         where 
#             hp_settle_dt between regexp_replace(date_sub(${today_date2},90),'-','') and ${today_date}
#             and trans_label='credit_repay' 
#         group by 
#             cdhd_usr_id) t2  --过去90天发生过信用卡还款的用户
#     on t1.usrid=t2.usrid
    
#     left join
#         (select usrid,score from ml_all_usr_crm__credit_repay_predict_result where part_dt=${model_date} and model='qianshuiliushi') t3  --根据模型评分是筛选
#     on t1.usrid=t3.usrid
    
#     left join
#         (
#         select
#             usrid
#             ,min(case when add_months(repay_deadline,month_diff)> ${today_date2} then add_months(repay_deadline,month_diff)
#                         when add_months(repay_deadline,month_diff)<= ${today_date2} then add_months(repay_deadline,month_diff+1)
#                         else null end) as ext_col2	  --最近还款日
#         from 
#             (
#             select 
#                 usrid
#                 ,pri_acct_no_sm3
#                 ,from_unixtime(unix_timestamp(max(repay_deadline),'yyyyMMdd'), "yyyy-MM-dd") as repay_deadline
#                 ,cast(months_between( trunc(${today_date2},'MM'),
#                                         trunc(from_unixtime(unix_timestamp(max(repay_deadline),'yyyyMMdd'), "yyyy-MM-dd"),'MM')) as int) as month_diff   --账单还款日距离本月月数
#             from 
#                 ml_usr_crm__repaydate_month 
#             group by 
#                 usrid,pri_acct_no_sm3
#             ) t
#         group by 
#             usrid
#         ) t4
#     on t1.usrid=t4.usrid
    
#     left join 
#         (
#         select
#             distinct usr_id as usrid
#         from
#             viw_uchis_ucbiz_cdhd_bas_inf_for_tsg
#         where
#             substr(bin(func_bmp), -20, 1) = '1'
#             and usr_id is not null
#             and trim(usr_id) <> ''
#         ) reg 
#     on t1.usrid = reg.usrid
#     where reg.usrid is null  --剔除海外用户 
#     and t2.usrid is null
#     ) t 
# """

# group_info="""
# 1;1~95;实验组1,push+5元券+还款日发送
# 2;96~100;对照组,不做营销
# """
# python3 demo02_create_exp.py "$exp_id" "$exp_name" "$department" "$admin" "$field" "$layer" "$create_time" "$comment" "$pre_sql" "$group_info" 



###################管理实验
exp_id=9
#'start' 'stop' 'info' 'delete'
python3 demo06_exp_manager.py "$exp_id" 'delete'



###################查看实验分布
# python3 demo04_over_view.py


