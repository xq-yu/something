from shutil import copyfile
copyfile('/Users/yu/Desktop/coding/github/something/AutoSql/AutoSql.py','./AutoSql.py') 
import AutoSql
import importlib
importlib.reload(AutoSql)




#%%
autosql = AutoSql.AutosqlFlowTable('./config2.xlsx',fun_dict)
basetable_ddl = autosql.base_table_create()
base_feature_sql,features = autosql.base_feature_create('./config2.xlsx',['prikey'],window='d20201201')

with open('./sql_command2.sql','w') as f:
    f.write('--基础流水表ddl\n')
    f.write(basetable_ddl)
    f.write('--基本特征select语句\n')
    f.write(base_feature_sql)

