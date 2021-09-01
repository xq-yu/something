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

if len(sys.argv)==3:
    exp_id = int(sys.argv[1])
    if sys.argv[2]=='start':
        ab.run_exp(exp_id)   #启动实验
    elif sys.argv[2]=='stop':
        ab.stop_exp(exp_id)  #停止实验
    elif sys.argv[2]=='info': 
        exp,group,sql = ab.get_exp(exp_id)#获取实验信息
        print("#########################")
        print(exp)
        print("#########################")
        print(group)
        print("#########################")
        print(sql)

    elif sys.argv[2]=='delete':
        ab.delete_exp(exp_id) 

    else:
        raise ValueError("Parameter Error!")
