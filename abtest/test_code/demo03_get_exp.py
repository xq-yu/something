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


if len(sys.argv)==2:
    exp_id = int(sys.argv[1])
    exp,group,sql = ab.get_exp(exp_id)
    print("#########################")
    print(exp)
    print("#########################")
    print(group)
    print("#########################")
    print(sql)

    quit()


'''

#############################
#3. 获取实验策略
# 用户可以根据自己的实验id查看该实验的基本信息，样本分组信息以及样本分组脚本
# 用户将得到的样本分组脚本迁移到生产执行
# 如需对生成的脚本进行部分修改，必须特别慎重，特别是对于分桶部分的sql语句。


# 功能测试:
# 获取实验信息和分组sql脚本
#############################
exp,group,sql = ab.get_exp(6)
print(exp)
print(group)
print(sql)

'''
