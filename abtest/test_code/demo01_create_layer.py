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


layer_info = {}
if len(sys.argv)==10:
    layer_info['layer_id'] = int(sys.argv[1])
    layer_info['field_id'] = int(sys.argv[2])
    layer_info['hash_col'] = sys.argv[3]
    layer_info['hash_method'] = sys.argv[4]
    layer_info['bucket_num'] =  int(sys.argv[5])
    layer_info['comment'] = sys.argv[6]
    layer_info['creator'] = sys.argv[7]
    layer_info['available_flg'] = sys.argv[8]
    layer_info['create_time'] = sys.argv[9]

    ab.create_layer(layer_info)
    
    quit()

###########################################
# 1. 创建实验层
# 尽量在现有的实验层进行ab实验，除非现有实验层的样本桶已经被占用导致样本不足
# 实验层只能由管理员进行新增或删除

# 功能测试:
# 实验层参数化配置, 实验层冲突检验
###########################################
# layer_info={
# 'layer_id':1
# ,'field_id':1
# ,'hash_col':'usrid'
# ,'hash_method':'conv(substring(md5(usrid),1,5),16,10)'
# ,'bucket_num':100
# ,'comment':'湖南分公司信用卡还款用户流失换回实验层，直接发送还款券，分析模型挽回效果'
# ,'creator':'俞晓强'
# ,'available_flg':1
# ,'create_time':'2021-07-15 00:00:00'
# }

# ab.create_layer(layer_info)


# layer_info={
# 'layer_id':2
# ,'field_id':1
# ,'hash_col':'usrid'
# ,'hash_method':'conv(substring(md5(usrid),1,5),16,10)'
# ,'bucket_num':100
# ,'comment':'湖南分公司信用卡还款用户拉新，分析模型拉新效果'
# ,'creator':'俞晓强'
# ,'available_flg':1
# ,'create_time':'2021-07-15 00:00:00'
# }
# ab.create_layer(layer_info)


# layer_info={
# 'layer_id':3
# ,'field_id':1
# ,'hash_col':'usrid'
# ,'hash_method':'conv(substring(md5(usrid),1,5),16,10)'
# ,'bucket_num':100
# ,'comment':'账户服务团队活跃用户流失模型测试'
# ,'creator':'俞晓强'
# ,'available_flg':1
# ,'create_time':'2021-08-23 00:00:00'
# }
# ab.create_layer(layer_info)



# layer_info={
# 'layer_id':4
# ,'field_id':1
# ,'hash_col':'usrid'
# ,'hash_method':'conv(substring(md5(usrid),1,5),16,10)'
# ,'bucket_num':100
# ,'comment':'账户服务团队浅睡用户流失模型测试'
# ,'creator':'俞晓强'
# ,'available_flg':1
# ,'create_time':'2021-08-27 00:00:00'
# }
# ab.create_layer(layer_info)



# layer_info={
# 'layer_id':5
# ,'field_id':1
# ,'hash_col':'usrid'
# ,'hash_method':'conv(substring(md5(usrid),1,5),16,10)'
# ,'bucket_num':100
# ,'comment':'账户服务团队深睡用户唤醒模型测试'
# ,'creator':'俞晓强'
# ,'available_flg':1
# ,'create_time':'2021-08-27 00:00:00'
# }
# ab.create_layer(layer_info)







