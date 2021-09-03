from flask import Flask,send_file,render_template,flash
from flask import request
import abtest
import importlib
importlib.reload(abtest)
import pandas as pd
import pymysql
import sys
import random
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
    """
    页面分组信息格式化整理
    """
    group = group.split(';')
    group_id = int(group[0].strip())
    
    if '~' in group[1]:
        s = int(group[1].split('~')[0].strip())
        e = int(group[1].split('~')[1].strip())
        group_buckets = list(range(s,e+1))
    else:
        group_buckets = [int(x.strip()) for x in group[1].split(',')]
    
    group_comment = group[2]

    return group_id,group_buckets,group_comment


app = Flask(__name__)
@app.route('/')
def main_page():
    """
    主页
    """
    sub_page = request.args.get('sub_page')
    return render_template('./main_page.html',sub_page=sub_page)

@app.route('/create_layer')
def create_layer():
    """
    创建实验层
    """
    return render_template('./create_layer.html')

@app.route('/create_exp')
def create_exp():
    """
    创建实验
    """
    return render_template('./create_exp.html')

@app.route('/over_view')
def over_view():
    """
    实验分布概览
    """
    ab.over_view()
    return render_template('./over_view.html',random_para=random.randint(0,10000))

@app.route('/exp_manage')
def exp_manage():
    """
    实验管理
    """
    cur.execute('select * from experiment')
    exp = cur.fetchall()
    
    return render_template('./exp_manage.html',exp=exp)




@app.route('/create_layer/create')
def create_layer_create():
    """
    创建实验层->创建
    """
    layer_info = {}
    layer_info['layer_id'] = int(request.args.get('layer_id'))
    layer_info['field_id'] = int(request.args.get('field_id'))
    layer_info['hash_col'] = request.args.get('hash_col')
    layer_info['hash_method'] = request.args.get('hash_method')
    layer_info['bucket_num'] =  int(request.args.get('bucket_num'))
    layer_info['comment'] = request.args.get('comment')
    layer_info['creator'] = request.args.get('creator')
    layer_info['available_flg'] = request.args.get('available_flg')
    layer_info['create_time'] = request.args.get('create_time')

    ab.create_layer(layer_info)

    return render_template('./create_layer.html')



@app.route('/create_exp/create')
def create_exp_create():
    """
    创建实验->创建
    """
    exp_id = int(request.args.get('exp_id'))

    exp_info = {}
    exp_info['exp_name'] = request.args.get('exp_name')
    exp_info['department'] = request.args.get('department')
    exp_info['admin'] = request.args.get('admin')
    exp_info['field'] = int(request.args.get('field'))
    exp_info['layer'] = int(request.args.get('layer'))
    exp_info['create_time'] = request.args.get('create_time')
    exp_info['comment'] = request.args.get('comment')
    exp_info['pre_sql'] = request.args.get('pre_sql').replace('\'','\\\'').replace('\"','\\\"')
    group_info = {}
    tmp = request.args.get('group_info')
    for group in tmp.split('\n'):
        if len(group.strip())>0:
            group_id,group_buckets,group_comment = group_clean(group.strip())
            group_info[group_id] = (group_buckets,group_comment)

    ab.create_exp(exp_id,exp_info,group_info)  #创建实验

    return render_template('./create_exp.html',exp_id=exp_id
                                                ,exp_name = request.args.get('exp_name')
                                                ,department = request.args.get('department')
                                                ,admin = request.args.get('admin')
                                                ,field = int(request.args.get('field'))
                                                ,layer = int(request.args.get('layer'))
                                                ,create_time = request.args.get('create_time')
                                                ,comment = request.args.get('comment')
                                                ,pre_sql = request.args.get('pre_sql')
                                                ,group_info = request.args.get('group_info')
                                                )


@app.route('/exp_manage/opt/')
def exp_manage_opt():
    """
    实验管理->操作
    """
    exp_id=int(request.args.get('exp_id'))
    opt = request.args.get('opt')

    if opt=='start':
        ab.run_exp(exp_id)   #启动实验
        cur.execute('select * from experiment')
        exp = cur.fetchall()
        return render_template('./exp_manage.html',exp=exp)
    elif opt=='stop':
        ab.stop_exp(exp_id)  #停止实验
        cur.execute('select * from experiment')
        exp = cur.fetchall()
        return render_template('./exp_manage.html',exp=exp)
    elif opt=='more': 
        exp,group,sql = ab.get_exp(exp_id)#获取实验信息
        return render_template('./experiment_more.html',exp=exp,group=group,sql=sql)
    elif opt=='edit':
        cur.execute('select * from experiment where exp_id=%s'%(exp_id))
        exp = cur.fetchall()[0]
        _,group_info,_ = ab.get_exp(exp_id)
        tmp = []
        for i in range(len(group_info)):
            group_id = group_info.index[i][1]
            comment = group_info.index[i][2]
            buckets = ','.join([str(x) for x in group_info.iloc[i]])
            s = '%s;%s;%s'%(group_id,buckets,comment)
            tmp.append(s)      
        group_info = '\n\n'.join(tmp)
        return render_template('./edit_exp.html', exp_id = exp[0]
                                                ,exp_name = exp[1]
                                                ,department = exp[2]
                                                ,admin = exp[3]
                                                ,field = exp[4]
                                                ,layer = exp[5]
                                                ,create_time = exp[6]
                                                ,comment = exp[7]
                                                ,pre_sql = exp[9]
                                                ,group_info = group_info)

    elif opt=='delete':
        ab.delete_exp(exp_id)
        cur.execute('select * from experiment')
        exp = cur.fetchall()
        return render_template('./exp_manage.html',exp=exp)

    else:
        raise ValueError("Parameter Error!")


@app.route('/exp_manage/edit')
def edit_experiment():
    """
    实验修改更新
    """
    exp_id = int(request.args.get('exp_id'))

    cur.execute('select valid_flg from experiment where exp_id=%s'%(exp_id))   #更新实验
    if cur.fetchall()[0][0]==1:
        message = '实验运行中,请先停止实验!实验更新失败'
    else:
        update_sql = """
        UPDATE experiment 
        SET 
            exp_name='%s',
            department='%s',
            admin='%s',
            field=%s,
            layer=%s,
            create_time='%s',
            comment='%s',
            pre_sql='%s' 
        where 
            exp_id=%s
        """%(request.args.get('exp_name'),
        request.args.get('department'),
        request.args.get('admin'),
        int(request.args.get('field')),
        int(request.args.get('layer')),
        request.args.get('create_time'),
        request.args.get('comment'),
        request.args.get('pre_sql').replace('\'','\\\'').replace('\"','\\\"'),
        exp_id)
        cur.execute(update_sql)   #更新实验
        message = '实验更新成功!'

    return render_template('./edit_exp.html', exp_id = exp_id
        ,exp_name = request.args.get('exp_name')
        ,department = request.args.get('department')
        ,admin = request.args.get('admin')
        ,field = int(request.args.get('field'))
        ,layer = int(request.args.get('layer'))
        ,create_time = request.args.get('create_time')
        ,comment = request.args.get('comment')
        ,pre_sql = request.args.get('pre_sql')
        ,group_info = request.args.get('group_info')
        ,message=message)


@app.route('/over_view/refresh')
def over_view_refresh():
    ab.over_view()
    return render_template('./over_view.html',random_para=random.randint(0,10000))

if __name__ == '__main__':
    app.run(debug=True,threaded=True)