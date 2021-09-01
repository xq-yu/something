
sql_head_head="""
use mydatabase;
add file hdfs://master:9000/tmp/filt.py;
add file hdfs://master:9000/tmp/merge.py;


select 
TRANSFORM (prikey,hiseventid,histimestamp_ms,timestamp_ms) using "merge.py" as prikey,hiseventid,histimestamp_ms,timestamp_ms,hisseriesid
from 
	(
	select 
	TRANSFORM (prikey,hiseventid,histimestamp_ms,timestamp_ms) using "filt.py" as prikey,hiseventid,histimestamp_ms,timestamp_ms,hisseriesid
	from
	(select * FROM  maidian1 distribute by prikey sort by histimestamp_ms desc) t
	distribute by prikey sort by histimestamp_ms desc
) t
;
"""

sql_ahead = """
-- 属性扩展
insert table trans_base_expand
select 
    prikey                 --主键
    ,timestamp_ms          --当前时间
    ,hiseventid            --历史埋点操作id
    ,histimestamp_ms       --历史埋点时间
    ,hisseriesid           --历史轨迹编号
    ,prikey_histime_start  --主键埋点起始时间
    ,prikey_histime_end    --主键埋点结束时间
    ,series_histime_start  --序列埋点起始时间
    ,series_histime_end    --序列埋点结束时间
    ,duration              --埋点持续时间
    ,eventid_back1         --前1个埋点id
    ,eventid_back2         --前2个埋点id
    ,eventid_back3         --前3个埋点id
    ,eventid_back4         --前4个埋点id
    ,rownum_desc           --倒数行为编号
    ,rownum_asc            --正数行为编号
    ,series_rownum_desc    --序列倒数行为编号
    ,series_rownum_asc     --序列正数行为编号

    ,series_histime_end - series_histime_start as series_duration  --序列时间
    ,timestamp_ms - histimestamp_ms as time_to_now
from
(
    select 
        prikey             --主键
        ,timestamp_ms      --当前时间
        ,hiseventid        --历史埋点操作id
        ,histimestamp_ms   --历史埋点时间
        ,hisseriesid       --历史轨迹编号
        ,min(histimestamp_ms) over (partition by prikey) as prikey_histime_start  --主键埋点起始时间
        ,max(histimestamp_ms) over (partition by prikey) as prikey_histime_end  --主键埋点结束时间
        ,min(histimestamp_ms) over (partition by prikey,hisseriesid) as series_histime_start  --序列埋点起始时间
        ,max(histimestamp_ms) over (partition by prikey,hisseriesid) as series_histime_end  --序列埋点结束时间
        ,histimestamp_ms-lag(histimestamp_ms,-1) over (partition by prikey order by histimestamp_ms) as duration  --埋点持续时间
        ,lag(hiseventid,1) over (partition by prikey order by histimestamp_ms) as eventid_back1   --前1个埋点id
        ,lag(hiseventid,2) over (partition by prikey order by histimestamp_ms) as eventid_back2   --前2个埋点id
        ,lag(hiseventid,3) over (partition by prikey order by histimestamp_ms) as eventid_back3   --前3个埋点id
        ,lag(hiseventid,4) over (partition by prikey order by histimestamp_ms) as eventid_back4   --前四个埋点id
        ,row_number() over (partition by prikey order by histimestamp_ms desc) as rownum_desc   --行为序号
        ,row_number() over (partition by prikey order by histimestamp_ms) as rownum_asc         --行为序号
        ,row_number() over (partition by prikey,hisseriesid order by histimestamp_ms desc) as series_rownum_desc
        ,row_number() over (partition by prikey,hisseriesid order by histimestamp_ms) as series_rownum_asc
    from 
        trans_base
) t;
"""

#固定窗口内埋点行为统计
def events_feature1(time_range):
    columns = []
    for ran in time_range.keys():
        # 总次数统计
        col_nm =  'cnt__hiseventid__%s'%(time_range[ran])
        print(",count(case when %s then hiseventid else null end) as %s"%(ran,col_nm))
        columns.append(col_nm)

        col_nm =  'cntdist__hiseventid__%s'%(time_range[ran])
        print(",count(distinct(case when %s then hiseventid else null end)) as %s"%(ran,col_nm))
        columns.append(col_nm)

        col_nm =  'sum__duration__%s'%(time_range[ran])
        print(",sum(case when %s and hiseventid<>'$append_' then duration else null end) as %s"%(ran,col_nm))
        columns.append(col_nm)

        # 单行为统计
        for eventid in eventid_ls:
            col_nm = 'cnt__hiseventid__%s__%s'%(time_range[ran],eventid.replace('$',''))
            print(",count(case when %s and hiseventid='%s' then hiseventid else null end) as %s"%(ran,eventid,col_nm))
            columns.append(col_nm)

            col_nm = 'sum__duration__%s__%s'%(time_range[ran],eventid.replace('$',''))
            print(",sum(case when %s and hiseventid='%s' then duration else null end) as %s"%(ran,eventid,col_nm))
            columns.append(col_nm)
    return columns

def events_feature2(n):
    columns = []
    for i in range(n):
        col_nm = 'min__time_to_now__series_rownum_desc_%s__hisseriesid_0'%(i)
        print(",min(case when series_rownum_desc=%s and hisseriesid=0 then time_to_now else null end) as %s"%(i,col_nm))
        columns.append(col_nm)
    return columns

if __name__ == "__main__":
    columns = []
    print(sql_head_head)
    print(sql_ahead)

    print("select prikey")
    prikey = 'prikey'
    eventid_ls = [  'apppageview_transfermoneypg'
                    ,'appevent_transferunicardcl'
                    ,'appevent_transferunipercl'
                    ,'appevent_transferinfpercl'
                    ,'transferdetcl_'
                    #通讯录转账埋点
                    ,'appevent_transferseapercl'
                    ,'transferdetpercl_'
                    ,'transferonepercl_'
                    ,'transferomitpercl_'
                    ,'appevent_transferomitsurecl'
                    ,'appevent_transferomitseecl'
                    ,'appevent_transferperbookcl'
                    ,'appevent_transferperpaycardcl'
                    ,'appevent_transferperpaybalacl'
                    ,'appevent_transferperopenpostcl'
                    ,'appevent_transferperhidepostcl'
                    ,'appevent_transferperpostcl'
                    ,'transferpermoneyconfcl_'

                    #转到银行卡
                    ,'appevent_transfercardbookcl'
                    ,'appevent_transfercardnamecl'
                    ,'appevent_transfercardcardnumcl'
                    ,'appevent_transfercardselectcl'
                    ,'appevent_transfercardbalacl'
                    ,'appevent_transfercardopenpostcl'
                    ,'appevent_transfercardhidepostcl'
                    ,'appevent_transfercardpostcl'
                    ,'transfercardmoneyconfcl_'

                    #转到云闪付用户
                    ,'appevent_transferuserphonecl'
                    ,'appevent_transferuseraddbookcl'
                    ,'appevent_transferusernextcl'
                    ,'appevent_transferusernextnamecl'
                    ,'appevent_transferusernextcardcl'
                    ,'appevent_transferusernextbalacl'
                    ,'appevent_transferusernextopenpostcl'
                    ,'appevent_transferusernexthidepostcl'
                    ,'appevent_transferusernextpostcl'
                    ,'transferusermoneyconfcl_'	

                    #起始结束标志
                    ,'$appstart_'
                    ,'$append_'
                    #首页相关
                    ,'apppageview_mainpg'
                    ,'bottomappcl_']


    time_range_condition = {
        "time_to_now<3600*1000":"time_to_now_0_60min"   #最近60min
        ,"time_to_now<1800*1000":"time_to_now_0_30min"  #最近30min
        ,"time_to_now<300*1000":"time_to_now_0_5min"    #最近5min
        ,"time_to_now<60*1000":"time_to_now_0_1min"     #最近1min
        ,"hisseriesid=0":"hisseriesid_0"                #最近一段行为序列
    }

    #固定条件内历史埋点统计
    columns = columns+events_feature1(time_range_condition)
    #最近n次埋点
    columns = columns+events_feature2(4)

    #其他特征
    print(',max(case when hisseriesid=0 then series_duration else null end) as max__series_duration__hisseriesid_0')

    columns.append('max__series_duration__hisseriesid_0')
    print("from events_table_base_expand group by prikey")

    [print(', '+x+' double') for x in columns]


