import sqlparse
import os
os.chdir('/Users/yu/Desktop/coding/github/something/AutoSql')
import re
from py2neo import Graph

def extract_table_depend(sql_str):
    """
    purpose:
        提取sql语句中的目标表以及数据源表
    input:
        sql_str:string 单条可执行sql语句
    output:
        tuple like (target_table,(origin_table1,origin_table2,...))
    """

    # remove the /* */ comments
    q = re.sub(r"/\*[^*]*\*+(?:[^*/][^*]*\*+)*/", "", sql_str)

    # remove whole line -- and # comments
    lines = [line for line in q.splitlines() if not re.match("^\s*(--|#)", line)]

    # remove trailing -- and # comments
    q = " ".join([re.split("--|#", line)[0] for line in lines])

    # split on blanks, parens and semicolons
    tokens = re.split(r"[\s)(;]+", q)

    # scan the tokens. if we see a FROM or JOIN, we set the get_next
    # flag, and grab the next one (unless it's SELECT).

    table_depend = []
    get_next = False
    for token in tokens:
        if get_next:
            if token.lower() not in ["", "select"]:
                table_depend.append(token)
            get_next = False
        get_next = token.lower() in ["from", "join"]

    i=0
    table_target=None
    while i<len(tokens) and tokens[i].lower() not in ('create','insert') :
        i+=1

    # create [temporary,external] table 
    if i<len(tokens) and tokens[i].lower()=='create':
        if  tokens[i+1].lower() in ('temporary','external'):
            table_target =  tokens[i+3]
        else:
            table_target = tokens[i+2]
    # insert into/overwrite table
    elif i<len(tokens) and tokens[i].lower()=='insert':
        table_target = tokens[i+3]

    return (table_target, set(table_depend))

def cypher_command(relation_pair,project_tag=None):
    """
    purpose:
        根据依赖关系生成
    input: 
        relation_pair:list/set [(依赖表,目标表)]
        project_tag: string 项目识别标签，作为节点的一个属性
    output:
        cypher:string cpyerher 语句
    """

    cypher = []
    cypher.append('create')

    # 定义node
    nodes_def = []
    nodes = set([x[0] for x in relation_pair]+[x[1] for x in relation_pair])
    node_alias = {}
    for i,node in enumerate(nodes):
        tmp = 'node_'+str(i)
        node_alias[node]=tmp
        if project_tag:
            nodes_def.append("(%s:table{name:'%s',project_tag:'%s'})"%(tmp,node,project_tag))
        else:
            nodes_def.append("(%s:table{name:'%s'})"%(tmp,node))
    nodes_def = ','.join(nodes_def)
    cypher.append(nodes_def)

    # 定义relationship
    cypher.append(',')
    relation_def = []
    for relation in relation_pair:
        relation_def.append("(%s)-[:to]->(%s)"%(node_alias[relation[0]],node_alias[relation[1]]))
    relation_def = ','.join(relation_def)
    cypher.append(relation_def)

    cypher.append("return 'finished'")
    cypher = '\n'.join(cypher)
    return cypher


if __name__ == '__main__':
    # 1.读取sql脚本
    with open('./test.sql') as f:
        relation = []
        # 多条sql语句分解
        sql_ls = sqlparse.split(''.join(f.readlines()))
        for sql in sql_ls:
            relation.append(extract_table_depend(sql)) 

    # 2.定义关系对
    relation_pair = []
    for i in relation:
        if i[0] is not None:
            for j in i[1]:
                relation_pair.append((j.lower(), i[0].lower()))
    relation_pair=set(relation_pair)

    # 3.生成网络
    cypher = cypher_command(relation_pair,project_tag='table_relation')
    graph = Graph('http://localhost:7474',username='neo4j',password='myneo4j')
    graph.run(cypher)


    # 4.删除节点
    graph.run("match (n) where n.project_tag='table_relation' detach delete n return 'finished'")



