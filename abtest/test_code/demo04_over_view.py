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


##############################
#4. 查看实验分布与占用
# 管理员可以查看实验层和当前生效实验的分布

# 功能测试:
# 实验层和样本桶可视化展示
#############################
ab.over_view()
