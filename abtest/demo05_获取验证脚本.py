import abtest
import importlib
importlib.reload(abtest)
import pandas as pd
import pymysql
mysql_host='localhost'
mysql_port=3306
passwd='123456'
user='root'
db='ABtest'
conn = pymysql.connect(host=mysql_host, port=mysql_port, user=user, passwd=passwd, db=db,autocommit = True)
conn.ping()
cur = conn.cursor()
ab = abtest.abtest(cur)

############################
#5. 验证分组效果
# 本程序已经内置了部分验证指标，如票券领取率，承兑率，失效率，用户交易活跃率等
# 对于特殊的验证指标用户需要根据本程序规定的sql格式自行编写

# 功能测试:
# 实验指标分析脚本输出
#############################
print(ab.coupon_eval_by_group(ab.coupon_eval(),4))
print(ab.coupon_eval_by_group(ab.active_eval(),4))