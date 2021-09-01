
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

select prikey
,count(case when time_to_now<3600*1000 then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min
,count(distinct(case when time_to_now<3600*1000 then hiseventid else null end)) as cntdist__hiseventid__time_to_now_0_60min
,sum(case when time_to_now<3600*1000 and hiseventid<>'$append_' then duration else null end) as sum__duration__time_to_now_0_60min
,count(case when time_to_now<3600*1000 and hiseventid='apppageview_transfermoneypg' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__apppageview_transfermoneypg
,sum(case when time_to_now<3600*1000 and hiseventid='apppageview_transfermoneypg' then duration else null end) as sum__duration__time_to_now_0_60min__apppageview_transfermoneypg
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferunicardcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferunicardcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferunicardcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferunicardcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferunipercl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferunipercl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferunipercl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferunipercl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferinfpercl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferinfpercl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferinfpercl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferinfpercl
,count(case when time_to_now<3600*1000 and hiseventid='transferdetcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__transferdetcl_
,sum(case when time_to_now<3600*1000 and hiseventid='transferdetcl_' then duration else null end) as sum__duration__time_to_now_0_60min__transferdetcl_
,count(case when time_to_now<3600*1000 and hiseventid='transferdetpercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__transferdetpercl_
,sum(case when time_to_now<3600*1000 and hiseventid='transferdetpercl_' then duration else null end) as sum__duration__time_to_now_0_60min__transferdetpercl_
,count(case when time_to_now<3600*1000 and hiseventid='transferonepercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__transferonepercl_
,sum(case when time_to_now<3600*1000 and hiseventid='transferonepercl_' then duration else null end) as sum__duration__time_to_now_0_60min__transferonepercl_
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferseapercl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferseapercl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferseapercl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferseapercl
,count(case when time_to_now<3600*1000 and hiseventid='transferdetpercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__transferdetpercl_
,sum(case when time_to_now<3600*1000 and hiseventid='transferdetpercl_' then duration else null end) as sum__duration__time_to_now_0_60min__transferdetpercl_
,count(case when time_to_now<3600*1000 and hiseventid='transferonepercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__transferonepercl_
,sum(case when time_to_now<3600*1000 and hiseventid='transferonepercl_' then duration else null end) as sum__duration__time_to_now_0_60min__transferonepercl_
,count(case when time_to_now<3600*1000 and hiseventid='transferomitpercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__transferomitpercl_
,sum(case when time_to_now<3600*1000 and hiseventid='transferomitpercl_' then duration else null end) as sum__duration__time_to_now_0_60min__transferomitpercl_
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferomitsurecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferomitsurecl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferomitsurecl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferomitsurecl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferomitseecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferomitseecl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferomitseecl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferomitseecl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferperbookcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferperbookcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferperbookcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferperbookcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferperpaycardcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferperpaycardcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferperpaycardcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferperpaycardcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferperpaybalacl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferperpaybalacl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferperpaybalacl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferperpaybalacl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferperopenpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferperopenpostcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferperopenpostcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferperopenpostcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferperhidepostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferperhidepostcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferperhidepostcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferperhidepostcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferperpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferperpostcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferperpostcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferperpostcl
,count(case when time_to_now<3600*1000 and hiseventid='transferpermoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__transferpermoneyconfcl_
,sum(case when time_to_now<3600*1000 and hiseventid='transferpermoneyconfcl_' then duration else null end) as sum__duration__time_to_now_0_60min__transferpermoneyconfcl_
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardbookcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transfercardbookcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardbookcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transfercardbookcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardnamecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transfercardnamecl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardnamecl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transfercardnamecl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardcardnumcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transfercardcardnumcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardcardnumcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transfercardcardnumcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardselectcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transfercardselectcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardselectcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transfercardselectcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardbalacl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transfercardbalacl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardbalacl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transfercardbalacl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardopenpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transfercardopenpostcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardopenpostcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transfercardopenpostcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardhidepostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transfercardhidepostcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardhidepostcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transfercardhidepostcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transfercardpostcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transfercardpostcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transfercardpostcl
,count(case when time_to_now<3600*1000 and hiseventid='transfercardmoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__transfercardmoneyconfcl_
,sum(case when time_to_now<3600*1000 and hiseventid='transfercardmoneyconfcl_' then duration else null end) as sum__duration__time_to_now_0_60min__transfercardmoneyconfcl_
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferuserphonecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferuserphonecl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferuserphonecl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferuserphonecl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferuseraddbookcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferuseraddbookcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferuseraddbookcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferuseraddbookcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferusernextcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferusernextcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferusernextcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferusernextcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferusernextnamecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferusernextnamecl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferusernextnamecl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferusernextnamecl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferusernextcardcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferusernextcardcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferusernextcardcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferusernextcardcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferusernextbalacl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferusernextbalacl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferusernextbalacl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferusernextbalacl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferusernextopenpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferusernextopenpostcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferusernextopenpostcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferusernextopenpostcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferusernexthidepostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferusernexthidepostcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferusernexthidepostcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferusernexthidepostcl
,count(case when time_to_now<3600*1000 and hiseventid='appevent_transferusernextpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appevent_transferusernextpostcl
,sum(case when time_to_now<3600*1000 and hiseventid='appevent_transferusernextpostcl' then duration else null end) as sum__duration__time_to_now_0_60min__appevent_transferusernextpostcl
,count(case when time_to_now<3600*1000 and hiseventid='transferusermoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__transferusermoneyconfcl_
,sum(case when time_to_now<3600*1000 and hiseventid='transferusermoneyconfcl_' then duration else null end) as sum__duration__time_to_now_0_60min__transferusermoneyconfcl_
,count(case when time_to_now<3600*1000 and hiseventid='$appstart_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__appstart_
,sum(case when time_to_now<3600*1000 and hiseventid='$appstart_' then duration else null end) as sum__duration__time_to_now_0_60min__appstart_
,count(case when time_to_now<3600*1000 and hiseventid='$append_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__append_
,sum(case when time_to_now<3600*1000 and hiseventid='$append_' then duration else null end) as sum__duration__time_to_now_0_60min__append_
,count(case when time_to_now<3600*1000 and hiseventid='apppageview_mainpg' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__apppageview_mainpg
,sum(case when time_to_now<3600*1000 and hiseventid='apppageview_mainpg' then duration else null end) as sum__duration__time_to_now_0_60min__apppageview_mainpg
,count(case when time_to_now<3600*1000 and hiseventid='bottomappcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_60min__bottomappcl_
,sum(case when time_to_now<3600*1000 and hiseventid='bottomappcl_' then duration else null end) as sum__duration__time_to_now_0_60min__bottomappcl_
,count(case when time_to_now<1800*1000 then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min
,count(distinct(case when time_to_now<1800*1000 then hiseventid else null end)) as cntdist__hiseventid__time_to_now_0_30min
,sum(case when time_to_now<1800*1000 and hiseventid<>'$append_' then duration else null end) as sum__duration__time_to_now_0_30min
,count(case when time_to_now<1800*1000 and hiseventid='apppageview_transfermoneypg' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__apppageview_transfermoneypg
,sum(case when time_to_now<1800*1000 and hiseventid='apppageview_transfermoneypg' then duration else null end) as sum__duration__time_to_now_0_30min__apppageview_transfermoneypg
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferunicardcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferunicardcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferunicardcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferunicardcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferunipercl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferunipercl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferunipercl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferunipercl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferinfpercl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferinfpercl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferinfpercl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferinfpercl
,count(case when time_to_now<1800*1000 and hiseventid='transferdetcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__transferdetcl_
,sum(case when time_to_now<1800*1000 and hiseventid='transferdetcl_' then duration else null end) as sum__duration__time_to_now_0_30min__transferdetcl_
,count(case when time_to_now<1800*1000 and hiseventid='transferdetpercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__transferdetpercl_
,sum(case when time_to_now<1800*1000 and hiseventid='transferdetpercl_' then duration else null end) as sum__duration__time_to_now_0_30min__transferdetpercl_
,count(case when time_to_now<1800*1000 and hiseventid='transferonepercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__transferonepercl_
,sum(case when time_to_now<1800*1000 and hiseventid='transferonepercl_' then duration else null end) as sum__duration__time_to_now_0_30min__transferonepercl_
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferseapercl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferseapercl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferseapercl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferseapercl
,count(case when time_to_now<1800*1000 and hiseventid='transferdetpercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__transferdetpercl_
,sum(case when time_to_now<1800*1000 and hiseventid='transferdetpercl_' then duration else null end) as sum__duration__time_to_now_0_30min__transferdetpercl_
,count(case when time_to_now<1800*1000 and hiseventid='transferonepercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__transferonepercl_
,sum(case when time_to_now<1800*1000 and hiseventid='transferonepercl_' then duration else null end) as sum__duration__time_to_now_0_30min__transferonepercl_
,count(case when time_to_now<1800*1000 and hiseventid='transferomitpercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__transferomitpercl_
,sum(case when time_to_now<1800*1000 and hiseventid='transferomitpercl_' then duration else null end) as sum__duration__time_to_now_0_30min__transferomitpercl_
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferomitsurecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferomitsurecl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferomitsurecl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferomitsurecl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferomitseecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferomitseecl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferomitseecl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferomitseecl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferperbookcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferperbookcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferperbookcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferperbookcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferperpaycardcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferperpaycardcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferperpaycardcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferperpaycardcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferperpaybalacl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferperpaybalacl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferperpaybalacl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferperpaybalacl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferperopenpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferperopenpostcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferperopenpostcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferperopenpostcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferperhidepostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferperhidepostcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferperhidepostcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferperhidepostcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferperpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferperpostcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferperpostcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferperpostcl
,count(case when time_to_now<1800*1000 and hiseventid='transferpermoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__transferpermoneyconfcl_
,sum(case when time_to_now<1800*1000 and hiseventid='transferpermoneyconfcl_' then duration else null end) as sum__duration__time_to_now_0_30min__transferpermoneyconfcl_
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardbookcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transfercardbookcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardbookcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transfercardbookcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardnamecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transfercardnamecl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardnamecl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transfercardnamecl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardcardnumcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transfercardcardnumcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardcardnumcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transfercardcardnumcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardselectcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transfercardselectcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardselectcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transfercardselectcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardbalacl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transfercardbalacl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardbalacl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transfercardbalacl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardopenpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transfercardopenpostcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardopenpostcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transfercardopenpostcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardhidepostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transfercardhidepostcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardhidepostcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transfercardhidepostcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transfercardpostcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transfercardpostcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transfercardpostcl
,count(case when time_to_now<1800*1000 and hiseventid='transfercardmoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__transfercardmoneyconfcl_
,sum(case when time_to_now<1800*1000 and hiseventid='transfercardmoneyconfcl_' then duration else null end) as sum__duration__time_to_now_0_30min__transfercardmoneyconfcl_
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferuserphonecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferuserphonecl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferuserphonecl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferuserphonecl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferuseraddbookcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferuseraddbookcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferuseraddbookcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferuseraddbookcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferusernextcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferusernextcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferusernextcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferusernextcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferusernextnamecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferusernextnamecl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferusernextnamecl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferusernextnamecl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferusernextcardcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferusernextcardcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferusernextcardcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferusernextcardcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferusernextbalacl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferusernextbalacl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferusernextbalacl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferusernextbalacl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferusernextopenpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferusernextopenpostcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferusernextopenpostcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferusernextopenpostcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferusernexthidepostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferusernexthidepostcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferusernexthidepostcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferusernexthidepostcl
,count(case when time_to_now<1800*1000 and hiseventid='appevent_transferusernextpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appevent_transferusernextpostcl
,sum(case when time_to_now<1800*1000 and hiseventid='appevent_transferusernextpostcl' then duration else null end) as sum__duration__time_to_now_0_30min__appevent_transferusernextpostcl
,count(case when time_to_now<1800*1000 and hiseventid='transferusermoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__transferusermoneyconfcl_
,sum(case when time_to_now<1800*1000 and hiseventid='transferusermoneyconfcl_' then duration else null end) as sum__duration__time_to_now_0_30min__transferusermoneyconfcl_
,count(case when time_to_now<1800*1000 and hiseventid='$appstart_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__appstart_
,sum(case when time_to_now<1800*1000 and hiseventid='$appstart_' then duration else null end) as sum__duration__time_to_now_0_30min__appstart_
,count(case when time_to_now<1800*1000 and hiseventid='$append_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__append_
,sum(case when time_to_now<1800*1000 and hiseventid='$append_' then duration else null end) as sum__duration__time_to_now_0_30min__append_
,count(case when time_to_now<1800*1000 and hiseventid='apppageview_mainpg' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__apppageview_mainpg
,sum(case when time_to_now<1800*1000 and hiseventid='apppageview_mainpg' then duration else null end) as sum__duration__time_to_now_0_30min__apppageview_mainpg
,count(case when time_to_now<1800*1000 and hiseventid='bottomappcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_30min__bottomappcl_
,sum(case when time_to_now<1800*1000 and hiseventid='bottomappcl_' then duration else null end) as sum__duration__time_to_now_0_30min__bottomappcl_
,count(case when time_to_now<300*1000 then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min
,count(distinct(case when time_to_now<300*1000 then hiseventid else null end)) as cntdist__hiseventid__time_to_now_0_5min
,sum(case when time_to_now<300*1000 and hiseventid<>'$append_' then duration else null end) as sum__duration__time_to_now_0_5min
,count(case when time_to_now<300*1000 and hiseventid='apppageview_transfermoneypg' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__apppageview_transfermoneypg
,sum(case when time_to_now<300*1000 and hiseventid='apppageview_transfermoneypg' then duration else null end) as sum__duration__time_to_now_0_5min__apppageview_transfermoneypg
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferunicardcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferunicardcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferunicardcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferunicardcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferunipercl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferunipercl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferunipercl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferunipercl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferinfpercl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferinfpercl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferinfpercl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferinfpercl
,count(case when time_to_now<300*1000 and hiseventid='transferdetcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__transferdetcl_
,sum(case when time_to_now<300*1000 and hiseventid='transferdetcl_' then duration else null end) as sum__duration__time_to_now_0_5min__transferdetcl_
,count(case when time_to_now<300*1000 and hiseventid='transferdetpercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__transferdetpercl_
,sum(case when time_to_now<300*1000 and hiseventid='transferdetpercl_' then duration else null end) as sum__duration__time_to_now_0_5min__transferdetpercl_
,count(case when time_to_now<300*1000 and hiseventid='transferonepercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__transferonepercl_
,sum(case when time_to_now<300*1000 and hiseventid='transferonepercl_' then duration else null end) as sum__duration__time_to_now_0_5min__transferonepercl_
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferseapercl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferseapercl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferseapercl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferseapercl
,count(case when time_to_now<300*1000 and hiseventid='transferdetpercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__transferdetpercl_
,sum(case when time_to_now<300*1000 and hiseventid='transferdetpercl_' then duration else null end) as sum__duration__time_to_now_0_5min__transferdetpercl_
,count(case when time_to_now<300*1000 and hiseventid='transferonepercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__transferonepercl_
,sum(case when time_to_now<300*1000 and hiseventid='transferonepercl_' then duration else null end) as sum__duration__time_to_now_0_5min__transferonepercl_
,count(case when time_to_now<300*1000 and hiseventid='transferomitpercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__transferomitpercl_
,sum(case when time_to_now<300*1000 and hiseventid='transferomitpercl_' then duration else null end) as sum__duration__time_to_now_0_5min__transferomitpercl_
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferomitsurecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferomitsurecl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferomitsurecl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferomitsurecl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferomitseecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferomitseecl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferomitseecl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferomitseecl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferperbookcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferperbookcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferperbookcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferperbookcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferperpaycardcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferperpaycardcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferperpaycardcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferperpaycardcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferperpaybalacl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferperpaybalacl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferperpaybalacl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferperpaybalacl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferperopenpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferperopenpostcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferperopenpostcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferperopenpostcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferperhidepostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferperhidepostcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferperhidepostcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferperhidepostcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferperpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferperpostcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferperpostcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferperpostcl
,count(case when time_to_now<300*1000 and hiseventid='transferpermoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__transferpermoneyconfcl_
,sum(case when time_to_now<300*1000 and hiseventid='transferpermoneyconfcl_' then duration else null end) as sum__duration__time_to_now_0_5min__transferpermoneyconfcl_
,count(case when time_to_now<300*1000 and hiseventid='appevent_transfercardbookcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transfercardbookcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transfercardbookcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transfercardbookcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transfercardnamecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transfercardnamecl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transfercardnamecl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transfercardnamecl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transfercardcardnumcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transfercardcardnumcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transfercardcardnumcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transfercardcardnumcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transfercardselectcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transfercardselectcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transfercardselectcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transfercardselectcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transfercardbalacl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transfercardbalacl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transfercardbalacl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transfercardbalacl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transfercardopenpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transfercardopenpostcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transfercardopenpostcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transfercardopenpostcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transfercardhidepostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transfercardhidepostcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transfercardhidepostcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transfercardhidepostcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transfercardpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transfercardpostcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transfercardpostcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transfercardpostcl
,count(case when time_to_now<300*1000 and hiseventid='transfercardmoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__transfercardmoneyconfcl_
,sum(case when time_to_now<300*1000 and hiseventid='transfercardmoneyconfcl_' then duration else null end) as sum__duration__time_to_now_0_5min__transfercardmoneyconfcl_
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferuserphonecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferuserphonecl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferuserphonecl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferuserphonecl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferuseraddbookcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferuseraddbookcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferuseraddbookcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferuseraddbookcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferusernextcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferusernextcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferusernextcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferusernextcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferusernextnamecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferusernextnamecl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferusernextnamecl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferusernextnamecl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferusernextcardcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferusernextcardcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferusernextcardcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferusernextcardcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferusernextbalacl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferusernextbalacl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferusernextbalacl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferusernextbalacl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferusernextopenpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferusernextopenpostcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferusernextopenpostcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferusernextopenpostcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferusernexthidepostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferusernexthidepostcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferusernexthidepostcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferusernexthidepostcl
,count(case when time_to_now<300*1000 and hiseventid='appevent_transferusernextpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appevent_transferusernextpostcl
,sum(case when time_to_now<300*1000 and hiseventid='appevent_transferusernextpostcl' then duration else null end) as sum__duration__time_to_now_0_5min__appevent_transferusernextpostcl
,count(case when time_to_now<300*1000 and hiseventid='transferusermoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__transferusermoneyconfcl_
,sum(case when time_to_now<300*1000 and hiseventid='transferusermoneyconfcl_' then duration else null end) as sum__duration__time_to_now_0_5min__transferusermoneyconfcl_
,count(case when time_to_now<300*1000 and hiseventid='$appstart_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__appstart_
,sum(case when time_to_now<300*1000 and hiseventid='$appstart_' then duration else null end) as sum__duration__time_to_now_0_5min__appstart_
,count(case when time_to_now<300*1000 and hiseventid='$append_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__append_
,sum(case when time_to_now<300*1000 and hiseventid='$append_' then duration else null end) as sum__duration__time_to_now_0_5min__append_
,count(case when time_to_now<300*1000 and hiseventid='apppageview_mainpg' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__apppageview_mainpg
,sum(case when time_to_now<300*1000 and hiseventid='apppageview_mainpg' then duration else null end) as sum__duration__time_to_now_0_5min__apppageview_mainpg
,count(case when time_to_now<300*1000 and hiseventid='bottomappcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_5min__bottomappcl_
,sum(case when time_to_now<300*1000 and hiseventid='bottomappcl_' then duration else null end) as sum__duration__time_to_now_0_5min__bottomappcl_
,count(case when time_to_now<60*1000 then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min
,count(distinct(case when time_to_now<60*1000 then hiseventid else null end)) as cntdist__hiseventid__time_to_now_0_1min
,sum(case when time_to_now<60*1000 and hiseventid<>'$append_' then duration else null end) as sum__duration__time_to_now_0_1min
,count(case when time_to_now<60*1000 and hiseventid='apppageview_transfermoneypg' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__apppageview_transfermoneypg
,sum(case when time_to_now<60*1000 and hiseventid='apppageview_transfermoneypg' then duration else null end) as sum__duration__time_to_now_0_1min__apppageview_transfermoneypg
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferunicardcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferunicardcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferunicardcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferunicardcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferunipercl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferunipercl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferunipercl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferunipercl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferinfpercl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferinfpercl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferinfpercl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferinfpercl
,count(case when time_to_now<60*1000 and hiseventid='transferdetcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__transferdetcl_
,sum(case when time_to_now<60*1000 and hiseventid='transferdetcl_' then duration else null end) as sum__duration__time_to_now_0_1min__transferdetcl_
,count(case when time_to_now<60*1000 and hiseventid='transferdetpercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__transferdetpercl_
,sum(case when time_to_now<60*1000 and hiseventid='transferdetpercl_' then duration else null end) as sum__duration__time_to_now_0_1min__transferdetpercl_
,count(case when time_to_now<60*1000 and hiseventid='transferonepercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__transferonepercl_
,sum(case when time_to_now<60*1000 and hiseventid='transferonepercl_' then duration else null end) as sum__duration__time_to_now_0_1min__transferonepercl_
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferseapercl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferseapercl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferseapercl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferseapercl
,count(case when time_to_now<60*1000 and hiseventid='transferdetpercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__transferdetpercl_
,sum(case when time_to_now<60*1000 and hiseventid='transferdetpercl_' then duration else null end) as sum__duration__time_to_now_0_1min__transferdetpercl_
,count(case when time_to_now<60*1000 and hiseventid='transferonepercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__transferonepercl_
,sum(case when time_to_now<60*1000 and hiseventid='transferonepercl_' then duration else null end) as sum__duration__time_to_now_0_1min__transferonepercl_
,count(case when time_to_now<60*1000 and hiseventid='transferomitpercl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__transferomitpercl_
,sum(case when time_to_now<60*1000 and hiseventid='transferomitpercl_' then duration else null end) as sum__duration__time_to_now_0_1min__transferomitpercl_
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferomitsurecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferomitsurecl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferomitsurecl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferomitsurecl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferomitseecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferomitseecl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferomitseecl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferomitseecl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferperbookcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferperbookcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferperbookcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferperbookcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferperpaycardcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferperpaycardcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferperpaycardcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferperpaycardcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferperpaybalacl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferperpaybalacl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferperpaybalacl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferperpaybalacl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferperopenpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferperopenpostcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferperopenpostcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferperopenpostcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferperhidepostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferperhidepostcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferperhidepostcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferperhidepostcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferperpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferperpostcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferperpostcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferperpostcl
,count(case when time_to_now<60*1000 and hiseventid='transferpermoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__transferpermoneyconfcl_
,sum(case when time_to_now<60*1000 and hiseventid='transferpermoneyconfcl_' then duration else null end) as sum__duration__time_to_now_0_1min__transferpermoneyconfcl_
,count(case when time_to_now<60*1000 and hiseventid='appevent_transfercardbookcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transfercardbookcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transfercardbookcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transfercardbookcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transfercardnamecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transfercardnamecl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transfercardnamecl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transfercardnamecl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transfercardcardnumcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transfercardcardnumcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transfercardcardnumcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transfercardcardnumcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transfercardselectcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transfercardselectcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transfercardselectcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transfercardselectcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transfercardbalacl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transfercardbalacl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transfercardbalacl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transfercardbalacl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transfercardopenpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transfercardopenpostcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transfercardopenpostcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transfercardopenpostcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transfercardhidepostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transfercardhidepostcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transfercardhidepostcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transfercardhidepostcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transfercardpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transfercardpostcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transfercardpostcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transfercardpostcl
,count(case when time_to_now<60*1000 and hiseventid='transfercardmoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__transfercardmoneyconfcl_
,sum(case when time_to_now<60*1000 and hiseventid='transfercardmoneyconfcl_' then duration else null end) as sum__duration__time_to_now_0_1min__transfercardmoneyconfcl_
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferuserphonecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferuserphonecl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferuserphonecl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferuserphonecl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferuseraddbookcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferuseraddbookcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferuseraddbookcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferuseraddbookcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferusernextcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferusernextcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferusernextcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferusernextcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferusernextnamecl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferusernextnamecl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferusernextnamecl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferusernextnamecl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferusernextcardcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferusernextcardcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferusernextcardcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferusernextcardcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferusernextbalacl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferusernextbalacl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferusernextbalacl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferusernextbalacl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferusernextopenpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferusernextopenpostcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferusernextopenpostcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferusernextopenpostcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferusernexthidepostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferusernexthidepostcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferusernexthidepostcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferusernexthidepostcl
,count(case when time_to_now<60*1000 and hiseventid='appevent_transferusernextpostcl' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appevent_transferusernextpostcl
,sum(case when time_to_now<60*1000 and hiseventid='appevent_transferusernextpostcl' then duration else null end) as sum__duration__time_to_now_0_1min__appevent_transferusernextpostcl
,count(case when time_to_now<60*1000 and hiseventid='transferusermoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__transferusermoneyconfcl_
,sum(case when time_to_now<60*1000 and hiseventid='transferusermoneyconfcl_' then duration else null end) as sum__duration__time_to_now_0_1min__transferusermoneyconfcl_
,count(case when time_to_now<60*1000 and hiseventid='$appstart_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__appstart_
,sum(case when time_to_now<60*1000 and hiseventid='$appstart_' then duration else null end) as sum__duration__time_to_now_0_1min__appstart_
,count(case when time_to_now<60*1000 and hiseventid='$append_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__append_
,sum(case when time_to_now<60*1000 and hiseventid='$append_' then duration else null end) as sum__duration__time_to_now_0_1min__append_
,count(case when time_to_now<60*1000 and hiseventid='apppageview_mainpg' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__apppageview_mainpg
,sum(case when time_to_now<60*1000 and hiseventid='apppageview_mainpg' then duration else null end) as sum__duration__time_to_now_0_1min__apppageview_mainpg
,count(case when time_to_now<60*1000 and hiseventid='bottomappcl_' then hiseventid else null end) as cnt__hiseventid__time_to_now_0_1min__bottomappcl_
,sum(case when time_to_now<60*1000 and hiseventid='bottomappcl_' then duration else null end) as sum__duration__time_to_now_0_1min__bottomappcl_
,count(case when hisseriesid=0 then hiseventid else null end) as cnt__hiseventid__hisseriesid_0
,count(distinct(case when hisseriesid=0 then hiseventid else null end)) as cntdist__hiseventid__hisseriesid_0
,sum(case when hisseriesid=0 and hiseventid<>'$append_' then duration else null end) as sum__duration__hisseriesid_0
,count(case when hisseriesid=0 and hiseventid='apppageview_transfermoneypg' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__apppageview_transfermoneypg
,sum(case when hisseriesid=0 and hiseventid='apppageview_transfermoneypg' then duration else null end) as sum__duration__hisseriesid_0__apppageview_transfermoneypg
,count(case when hisseriesid=0 and hiseventid='appevent_transferunicardcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferunicardcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferunicardcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferunicardcl
,count(case when hisseriesid=0 and hiseventid='appevent_transferunipercl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferunipercl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferunipercl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferunipercl
,count(case when hisseriesid=0 and hiseventid='appevent_transferinfpercl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferinfpercl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferinfpercl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferinfpercl
,count(case when hisseriesid=0 and hiseventid='transferdetcl_' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__transferdetcl_
,sum(case when hisseriesid=0 and hiseventid='transferdetcl_' then duration else null end) as sum__duration__hisseriesid_0__transferdetcl_
,count(case when hisseriesid=0 and hiseventid='transferdetpercl_' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__transferdetpercl_
,sum(case when hisseriesid=0 and hiseventid='transferdetpercl_' then duration else null end) as sum__duration__hisseriesid_0__transferdetpercl_
,count(case when hisseriesid=0 and hiseventid='transferonepercl_' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__transferonepercl_
,sum(case when hisseriesid=0 and hiseventid='transferonepercl_' then duration else null end) as sum__duration__hisseriesid_0__transferonepercl_
,count(case when hisseriesid=0 and hiseventid='appevent_transferseapercl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferseapercl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferseapercl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferseapercl
,count(case when hisseriesid=0 and hiseventid='transferdetpercl_' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__transferdetpercl_
,sum(case when hisseriesid=0 and hiseventid='transferdetpercl_' then duration else null end) as sum__duration__hisseriesid_0__transferdetpercl_
,count(case when hisseriesid=0 and hiseventid='transferonepercl_' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__transferonepercl_
,sum(case when hisseriesid=0 and hiseventid='transferonepercl_' then duration else null end) as sum__duration__hisseriesid_0__transferonepercl_
,count(case when hisseriesid=0 and hiseventid='transferomitpercl_' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__transferomitpercl_
,sum(case when hisseriesid=0 and hiseventid='transferomitpercl_' then duration else null end) as sum__duration__hisseriesid_0__transferomitpercl_
,count(case when hisseriesid=0 and hiseventid='appevent_transferomitsurecl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferomitsurecl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferomitsurecl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferomitsurecl
,count(case when hisseriesid=0 and hiseventid='appevent_transferomitseecl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferomitseecl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferomitseecl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferomitseecl
,count(case when hisseriesid=0 and hiseventid='appevent_transferperbookcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferperbookcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferperbookcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferperbookcl
,count(case when hisseriesid=0 and hiseventid='appevent_transferperpaycardcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferperpaycardcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferperpaycardcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferperpaycardcl
,count(case when hisseriesid=0 and hiseventid='appevent_transferperpaybalacl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferperpaybalacl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferperpaybalacl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferperpaybalacl
,count(case when hisseriesid=0 and hiseventid='appevent_transferperopenpostcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferperopenpostcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferperopenpostcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferperopenpostcl
,count(case when hisseriesid=0 and hiseventid='appevent_transferperhidepostcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferperhidepostcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferperhidepostcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferperhidepostcl
,count(case when hisseriesid=0 and hiseventid='appevent_transferperpostcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferperpostcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferperpostcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferperpostcl
,count(case when hisseriesid=0 and hiseventid='transferpermoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__transferpermoneyconfcl_
,sum(case when hisseriesid=0 and hiseventid='transferpermoneyconfcl_' then duration else null end) as sum__duration__hisseriesid_0__transferpermoneyconfcl_
,count(case when hisseriesid=0 and hiseventid='appevent_transfercardbookcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transfercardbookcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transfercardbookcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transfercardbookcl
,count(case when hisseriesid=0 and hiseventid='appevent_transfercardnamecl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transfercardnamecl
,sum(case when hisseriesid=0 and hiseventid='appevent_transfercardnamecl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transfercardnamecl
,count(case when hisseriesid=0 and hiseventid='appevent_transfercardcardnumcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transfercardcardnumcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transfercardcardnumcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transfercardcardnumcl
,count(case when hisseriesid=0 and hiseventid='appevent_transfercardselectcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transfercardselectcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transfercardselectcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transfercardselectcl
,count(case when hisseriesid=0 and hiseventid='appevent_transfercardbalacl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transfercardbalacl
,sum(case when hisseriesid=0 and hiseventid='appevent_transfercardbalacl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transfercardbalacl
,count(case when hisseriesid=0 and hiseventid='appevent_transfercardopenpostcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transfercardopenpostcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transfercardopenpostcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transfercardopenpostcl
,count(case when hisseriesid=0 and hiseventid='appevent_transfercardhidepostcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transfercardhidepostcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transfercardhidepostcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transfercardhidepostcl
,count(case when hisseriesid=0 and hiseventid='appevent_transfercardpostcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transfercardpostcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transfercardpostcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transfercardpostcl
,count(case when hisseriesid=0 and hiseventid='transfercardmoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__transfercardmoneyconfcl_
,sum(case when hisseriesid=0 and hiseventid='transfercardmoneyconfcl_' then duration else null end) as sum__duration__hisseriesid_0__transfercardmoneyconfcl_
,count(case when hisseriesid=0 and hiseventid='appevent_transferuserphonecl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferuserphonecl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferuserphonecl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferuserphonecl
,count(case when hisseriesid=0 and hiseventid='appevent_transferuseraddbookcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferuseraddbookcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferuseraddbookcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferuseraddbookcl
,count(case when hisseriesid=0 and hiseventid='appevent_transferusernextcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferusernextcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferusernextcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferusernextcl
,count(case when hisseriesid=0 and hiseventid='appevent_transferusernextnamecl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferusernextnamecl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferusernextnamecl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferusernextnamecl
,count(case when hisseriesid=0 and hiseventid='appevent_transferusernextcardcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferusernextcardcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferusernextcardcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferusernextcardcl
,count(case when hisseriesid=0 and hiseventid='appevent_transferusernextbalacl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferusernextbalacl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferusernextbalacl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferusernextbalacl
,count(case when hisseriesid=0 and hiseventid='appevent_transferusernextopenpostcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferusernextopenpostcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferusernextopenpostcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferusernextopenpostcl
,count(case when hisseriesid=0 and hiseventid='appevent_transferusernexthidepostcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferusernexthidepostcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferusernexthidepostcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferusernexthidepostcl
,count(case when hisseriesid=0 and hiseventid='appevent_transferusernextpostcl' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appevent_transferusernextpostcl
,sum(case when hisseriesid=0 and hiseventid='appevent_transferusernextpostcl' then duration else null end) as sum__duration__hisseriesid_0__appevent_transferusernextpostcl
,count(case when hisseriesid=0 and hiseventid='transferusermoneyconfcl_' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__transferusermoneyconfcl_
,sum(case when hisseriesid=0 and hiseventid='transferusermoneyconfcl_' then duration else null end) as sum__duration__hisseriesid_0__transferusermoneyconfcl_
,count(case when hisseriesid=0 and hiseventid='$appstart_' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__appstart_
,sum(case when hisseriesid=0 and hiseventid='$appstart_' then duration else null end) as sum__duration__hisseriesid_0__appstart_
,count(case when hisseriesid=0 and hiseventid='$append_' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__append_
,sum(case when hisseriesid=0 and hiseventid='$append_' then duration else null end) as sum__duration__hisseriesid_0__append_
,count(case when hisseriesid=0 and hiseventid='apppageview_mainpg' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__apppageview_mainpg
,sum(case when hisseriesid=0 and hiseventid='apppageview_mainpg' then duration else null end) as sum__duration__hisseriesid_0__apppageview_mainpg
,count(case when hisseriesid=0 and hiseventid='bottomappcl_' then hiseventid else null end) as cnt__hiseventid__hisseriesid_0__bottomappcl_
,sum(case when hisseriesid=0 and hiseventid='bottomappcl_' then duration else null end) as sum__duration__hisseriesid_0__bottomappcl_
,min(case when series_rownum_desc=0 and hisseriesid=0 then time_to_now else null end) as min__time_to_now__series_rownum_desc_0__hisseriesid_0
,min(case when series_rownum_desc=1 and hisseriesid=0 then time_to_now else null end) as min__time_to_now__series_rownum_desc_1__hisseriesid_0
,min(case when series_rownum_desc=2 and hisseriesid=0 then time_to_now else null end) as min__time_to_now__series_rownum_desc_2__hisseriesid_0
,min(case when series_rownum_desc=3 and hisseriesid=0 then time_to_now else null end) as min__time_to_now__series_rownum_desc_3__hisseriesid_0
,max(case when hisseriesid=0 then series_duration else null end) as max__series_duration__hisseriesid_0
from events_table_base_expand group by prikey
, cnt__hiseventid__time_to_now_0_60min double
, cntdist__hiseventid__time_to_now_0_60min double
, sum__duration__time_to_now_0_60min double
, cnt__hiseventid__time_to_now_0_60min__apppageview_transfermoneypg double
, sum__duration__time_to_now_0_60min__apppageview_transfermoneypg double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferunicardcl double
, sum__duration__time_to_now_0_60min__appevent_transferunicardcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferunipercl double
, sum__duration__time_to_now_0_60min__appevent_transferunipercl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferinfpercl double
, sum__duration__time_to_now_0_60min__appevent_transferinfpercl double
, cnt__hiseventid__time_to_now_0_60min__transferdetcl_ double
, sum__duration__time_to_now_0_60min__transferdetcl_ double
, cnt__hiseventid__time_to_now_0_60min__transferdetpercl_ double
, sum__duration__time_to_now_0_60min__transferdetpercl_ double
, cnt__hiseventid__time_to_now_0_60min__transferonepercl_ double
, sum__duration__time_to_now_0_60min__transferonepercl_ double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferseapercl double
, sum__duration__time_to_now_0_60min__appevent_transferseapercl double
, cnt__hiseventid__time_to_now_0_60min__transferdetpercl_ double
, sum__duration__time_to_now_0_60min__transferdetpercl_ double
, cnt__hiseventid__time_to_now_0_60min__transferonepercl_ double
, sum__duration__time_to_now_0_60min__transferonepercl_ double
, cnt__hiseventid__time_to_now_0_60min__transferomitpercl_ double
, sum__duration__time_to_now_0_60min__transferomitpercl_ double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferomitsurecl double
, sum__duration__time_to_now_0_60min__appevent_transferomitsurecl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferomitseecl double
, sum__duration__time_to_now_0_60min__appevent_transferomitseecl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferperbookcl double
, sum__duration__time_to_now_0_60min__appevent_transferperbookcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferperpaycardcl double
, sum__duration__time_to_now_0_60min__appevent_transferperpaycardcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferperpaybalacl double
, sum__duration__time_to_now_0_60min__appevent_transferperpaybalacl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferperopenpostcl double
, sum__duration__time_to_now_0_60min__appevent_transferperopenpostcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferperhidepostcl double
, sum__duration__time_to_now_0_60min__appevent_transferperhidepostcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferperpostcl double
, sum__duration__time_to_now_0_60min__appevent_transferperpostcl double
, cnt__hiseventid__time_to_now_0_60min__transferpermoneyconfcl_ double
, sum__duration__time_to_now_0_60min__transferpermoneyconfcl_ double
, cnt__hiseventid__time_to_now_0_60min__appevent_transfercardbookcl double
, sum__duration__time_to_now_0_60min__appevent_transfercardbookcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transfercardnamecl double
, sum__duration__time_to_now_0_60min__appevent_transfercardnamecl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transfercardcardnumcl double
, sum__duration__time_to_now_0_60min__appevent_transfercardcardnumcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transfercardselectcl double
, sum__duration__time_to_now_0_60min__appevent_transfercardselectcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transfercardbalacl double
, sum__duration__time_to_now_0_60min__appevent_transfercardbalacl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transfercardopenpostcl double
, sum__duration__time_to_now_0_60min__appevent_transfercardopenpostcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transfercardhidepostcl double
, sum__duration__time_to_now_0_60min__appevent_transfercardhidepostcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transfercardpostcl double
, sum__duration__time_to_now_0_60min__appevent_transfercardpostcl double
, cnt__hiseventid__time_to_now_0_60min__transfercardmoneyconfcl_ double
, sum__duration__time_to_now_0_60min__transfercardmoneyconfcl_ double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferuserphonecl double
, sum__duration__time_to_now_0_60min__appevent_transferuserphonecl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferuseraddbookcl double
, sum__duration__time_to_now_0_60min__appevent_transferuseraddbookcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferusernextcl double
, sum__duration__time_to_now_0_60min__appevent_transferusernextcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferusernextnamecl double
, sum__duration__time_to_now_0_60min__appevent_transferusernextnamecl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferusernextcardcl double
, sum__duration__time_to_now_0_60min__appevent_transferusernextcardcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferusernextbalacl double
, sum__duration__time_to_now_0_60min__appevent_transferusernextbalacl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferusernextopenpostcl double
, sum__duration__time_to_now_0_60min__appevent_transferusernextopenpostcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferusernexthidepostcl double
, sum__duration__time_to_now_0_60min__appevent_transferusernexthidepostcl double
, cnt__hiseventid__time_to_now_0_60min__appevent_transferusernextpostcl double
, sum__duration__time_to_now_0_60min__appevent_transferusernextpostcl double
, cnt__hiseventid__time_to_now_0_60min__transferusermoneyconfcl_ double
, sum__duration__time_to_now_0_60min__transferusermoneyconfcl_ double
, cnt__hiseventid__time_to_now_0_60min__appstart_ double
, sum__duration__time_to_now_0_60min__appstart_ double
, cnt__hiseventid__time_to_now_0_60min__append_ double
, sum__duration__time_to_now_0_60min__append_ double
, cnt__hiseventid__time_to_now_0_60min__apppageview_mainpg double
, sum__duration__time_to_now_0_60min__apppageview_mainpg double
, cnt__hiseventid__time_to_now_0_60min__bottomappcl_ double
, sum__duration__time_to_now_0_60min__bottomappcl_ double
, cnt__hiseventid__time_to_now_0_30min double
, cntdist__hiseventid__time_to_now_0_30min double
, sum__duration__time_to_now_0_30min double
, cnt__hiseventid__time_to_now_0_30min__apppageview_transfermoneypg double
, sum__duration__time_to_now_0_30min__apppageview_transfermoneypg double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferunicardcl double
, sum__duration__time_to_now_0_30min__appevent_transferunicardcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferunipercl double
, sum__duration__time_to_now_0_30min__appevent_transferunipercl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferinfpercl double
, sum__duration__time_to_now_0_30min__appevent_transferinfpercl double
, cnt__hiseventid__time_to_now_0_30min__transferdetcl_ double
, sum__duration__time_to_now_0_30min__transferdetcl_ double
, cnt__hiseventid__time_to_now_0_30min__transferdetpercl_ double
, sum__duration__time_to_now_0_30min__transferdetpercl_ double
, cnt__hiseventid__time_to_now_0_30min__transferonepercl_ double
, sum__duration__time_to_now_0_30min__transferonepercl_ double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferseapercl double
, sum__duration__time_to_now_0_30min__appevent_transferseapercl double
, cnt__hiseventid__time_to_now_0_30min__transferdetpercl_ double
, sum__duration__time_to_now_0_30min__transferdetpercl_ double
, cnt__hiseventid__time_to_now_0_30min__transferonepercl_ double
, sum__duration__time_to_now_0_30min__transferonepercl_ double
, cnt__hiseventid__time_to_now_0_30min__transferomitpercl_ double
, sum__duration__time_to_now_0_30min__transferomitpercl_ double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferomitsurecl double
, sum__duration__time_to_now_0_30min__appevent_transferomitsurecl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferomitseecl double
, sum__duration__time_to_now_0_30min__appevent_transferomitseecl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferperbookcl double
, sum__duration__time_to_now_0_30min__appevent_transferperbookcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferperpaycardcl double
, sum__duration__time_to_now_0_30min__appevent_transferperpaycardcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferperpaybalacl double
, sum__duration__time_to_now_0_30min__appevent_transferperpaybalacl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferperopenpostcl double
, sum__duration__time_to_now_0_30min__appevent_transferperopenpostcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferperhidepostcl double
, sum__duration__time_to_now_0_30min__appevent_transferperhidepostcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferperpostcl double
, sum__duration__time_to_now_0_30min__appevent_transferperpostcl double
, cnt__hiseventid__time_to_now_0_30min__transferpermoneyconfcl_ double
, sum__duration__time_to_now_0_30min__transferpermoneyconfcl_ double
, cnt__hiseventid__time_to_now_0_30min__appevent_transfercardbookcl double
, sum__duration__time_to_now_0_30min__appevent_transfercardbookcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transfercardnamecl double
, sum__duration__time_to_now_0_30min__appevent_transfercardnamecl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transfercardcardnumcl double
, sum__duration__time_to_now_0_30min__appevent_transfercardcardnumcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transfercardselectcl double
, sum__duration__time_to_now_0_30min__appevent_transfercardselectcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transfercardbalacl double
, sum__duration__time_to_now_0_30min__appevent_transfercardbalacl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transfercardopenpostcl double
, sum__duration__time_to_now_0_30min__appevent_transfercardopenpostcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transfercardhidepostcl double
, sum__duration__time_to_now_0_30min__appevent_transfercardhidepostcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transfercardpostcl double
, sum__duration__time_to_now_0_30min__appevent_transfercardpostcl double
, cnt__hiseventid__time_to_now_0_30min__transfercardmoneyconfcl_ double
, sum__duration__time_to_now_0_30min__transfercardmoneyconfcl_ double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferuserphonecl double
, sum__duration__time_to_now_0_30min__appevent_transferuserphonecl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferuseraddbookcl double
, sum__duration__time_to_now_0_30min__appevent_transferuseraddbookcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferusernextcl double
, sum__duration__time_to_now_0_30min__appevent_transferusernextcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferusernextnamecl double
, sum__duration__time_to_now_0_30min__appevent_transferusernextnamecl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferusernextcardcl double
, sum__duration__time_to_now_0_30min__appevent_transferusernextcardcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferusernextbalacl double
, sum__duration__time_to_now_0_30min__appevent_transferusernextbalacl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferusernextopenpostcl double
, sum__duration__time_to_now_0_30min__appevent_transferusernextopenpostcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferusernexthidepostcl double
, sum__duration__time_to_now_0_30min__appevent_transferusernexthidepostcl double
, cnt__hiseventid__time_to_now_0_30min__appevent_transferusernextpostcl double
, sum__duration__time_to_now_0_30min__appevent_transferusernextpostcl double
, cnt__hiseventid__time_to_now_0_30min__transferusermoneyconfcl_ double
, sum__duration__time_to_now_0_30min__transferusermoneyconfcl_ double
, cnt__hiseventid__time_to_now_0_30min__appstart_ double
, sum__duration__time_to_now_0_30min__appstart_ double
, cnt__hiseventid__time_to_now_0_30min__append_ double
, sum__duration__time_to_now_0_30min__append_ double
, cnt__hiseventid__time_to_now_0_30min__apppageview_mainpg double
, sum__duration__time_to_now_0_30min__apppageview_mainpg double
, cnt__hiseventid__time_to_now_0_30min__bottomappcl_ double
, sum__duration__time_to_now_0_30min__bottomappcl_ double
, cnt__hiseventid__time_to_now_0_5min double
, cntdist__hiseventid__time_to_now_0_5min double
, sum__duration__time_to_now_0_5min double
, cnt__hiseventid__time_to_now_0_5min__apppageview_transfermoneypg double
, sum__duration__time_to_now_0_5min__apppageview_transfermoneypg double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferunicardcl double
, sum__duration__time_to_now_0_5min__appevent_transferunicardcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferunipercl double
, sum__duration__time_to_now_0_5min__appevent_transferunipercl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferinfpercl double
, sum__duration__time_to_now_0_5min__appevent_transferinfpercl double
, cnt__hiseventid__time_to_now_0_5min__transferdetcl_ double
, sum__duration__time_to_now_0_5min__transferdetcl_ double
, cnt__hiseventid__time_to_now_0_5min__transferdetpercl_ double
, sum__duration__time_to_now_0_5min__transferdetpercl_ double
, cnt__hiseventid__time_to_now_0_5min__transferonepercl_ double
, sum__duration__time_to_now_0_5min__transferonepercl_ double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferseapercl double
, sum__duration__time_to_now_0_5min__appevent_transferseapercl double
, cnt__hiseventid__time_to_now_0_5min__transferdetpercl_ double
, sum__duration__time_to_now_0_5min__transferdetpercl_ double
, cnt__hiseventid__time_to_now_0_5min__transferonepercl_ double
, sum__duration__time_to_now_0_5min__transferonepercl_ double
, cnt__hiseventid__time_to_now_0_5min__transferomitpercl_ double
, sum__duration__time_to_now_0_5min__transferomitpercl_ double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferomitsurecl double
, sum__duration__time_to_now_0_5min__appevent_transferomitsurecl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferomitseecl double
, sum__duration__time_to_now_0_5min__appevent_transferomitseecl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferperbookcl double
, sum__duration__time_to_now_0_5min__appevent_transferperbookcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferperpaycardcl double
, sum__duration__time_to_now_0_5min__appevent_transferperpaycardcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferperpaybalacl double
, sum__duration__time_to_now_0_5min__appevent_transferperpaybalacl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferperopenpostcl double
, sum__duration__time_to_now_0_5min__appevent_transferperopenpostcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferperhidepostcl double
, sum__duration__time_to_now_0_5min__appevent_transferperhidepostcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferperpostcl double
, sum__duration__time_to_now_0_5min__appevent_transferperpostcl double
, cnt__hiseventid__time_to_now_0_5min__transferpermoneyconfcl_ double
, sum__duration__time_to_now_0_5min__transferpermoneyconfcl_ double
, cnt__hiseventid__time_to_now_0_5min__appevent_transfercardbookcl double
, sum__duration__time_to_now_0_5min__appevent_transfercardbookcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transfercardnamecl double
, sum__duration__time_to_now_0_5min__appevent_transfercardnamecl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transfercardcardnumcl double
, sum__duration__time_to_now_0_5min__appevent_transfercardcardnumcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transfercardselectcl double
, sum__duration__time_to_now_0_5min__appevent_transfercardselectcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transfercardbalacl double
, sum__duration__time_to_now_0_5min__appevent_transfercardbalacl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transfercardopenpostcl double
, sum__duration__time_to_now_0_5min__appevent_transfercardopenpostcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transfercardhidepostcl double
, sum__duration__time_to_now_0_5min__appevent_transfercardhidepostcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transfercardpostcl double
, sum__duration__time_to_now_0_5min__appevent_transfercardpostcl double
, cnt__hiseventid__time_to_now_0_5min__transfercardmoneyconfcl_ double
, sum__duration__time_to_now_0_5min__transfercardmoneyconfcl_ double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferuserphonecl double
, sum__duration__time_to_now_0_5min__appevent_transferuserphonecl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferuseraddbookcl double
, sum__duration__time_to_now_0_5min__appevent_transferuseraddbookcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferusernextcl double
, sum__duration__time_to_now_0_5min__appevent_transferusernextcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferusernextnamecl double
, sum__duration__time_to_now_0_5min__appevent_transferusernextnamecl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferusernextcardcl double
, sum__duration__time_to_now_0_5min__appevent_transferusernextcardcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferusernextbalacl double
, sum__duration__time_to_now_0_5min__appevent_transferusernextbalacl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferusernextopenpostcl double
, sum__duration__time_to_now_0_5min__appevent_transferusernextopenpostcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferusernexthidepostcl double
, sum__duration__time_to_now_0_5min__appevent_transferusernexthidepostcl double
, cnt__hiseventid__time_to_now_0_5min__appevent_transferusernextpostcl double
, sum__duration__time_to_now_0_5min__appevent_transferusernextpostcl double
, cnt__hiseventid__time_to_now_0_5min__transferusermoneyconfcl_ double
, sum__duration__time_to_now_0_5min__transferusermoneyconfcl_ double
, cnt__hiseventid__time_to_now_0_5min__appstart_ double
, sum__duration__time_to_now_0_5min__appstart_ double
, cnt__hiseventid__time_to_now_0_5min__append_ double
, sum__duration__time_to_now_0_5min__append_ double
, cnt__hiseventid__time_to_now_0_5min__apppageview_mainpg double
, sum__duration__time_to_now_0_5min__apppageview_mainpg double
, cnt__hiseventid__time_to_now_0_5min__bottomappcl_ double
, sum__duration__time_to_now_0_5min__bottomappcl_ double
, cnt__hiseventid__time_to_now_0_1min double
, cntdist__hiseventid__time_to_now_0_1min double
, sum__duration__time_to_now_0_1min double
, cnt__hiseventid__time_to_now_0_1min__apppageview_transfermoneypg double
, sum__duration__time_to_now_0_1min__apppageview_transfermoneypg double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferunicardcl double
, sum__duration__time_to_now_0_1min__appevent_transferunicardcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferunipercl double
, sum__duration__time_to_now_0_1min__appevent_transferunipercl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferinfpercl double
, sum__duration__time_to_now_0_1min__appevent_transferinfpercl double
, cnt__hiseventid__time_to_now_0_1min__transferdetcl_ double
, sum__duration__time_to_now_0_1min__transferdetcl_ double
, cnt__hiseventid__time_to_now_0_1min__transferdetpercl_ double
, sum__duration__time_to_now_0_1min__transferdetpercl_ double
, cnt__hiseventid__time_to_now_0_1min__transferonepercl_ double
, sum__duration__time_to_now_0_1min__transferonepercl_ double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferseapercl double
, sum__duration__time_to_now_0_1min__appevent_transferseapercl double
, cnt__hiseventid__time_to_now_0_1min__transferdetpercl_ double
, sum__duration__time_to_now_0_1min__transferdetpercl_ double
, cnt__hiseventid__time_to_now_0_1min__transferonepercl_ double
, sum__duration__time_to_now_0_1min__transferonepercl_ double
, cnt__hiseventid__time_to_now_0_1min__transferomitpercl_ double
, sum__duration__time_to_now_0_1min__transferomitpercl_ double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferomitsurecl double
, sum__duration__time_to_now_0_1min__appevent_transferomitsurecl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferomitseecl double
, sum__duration__time_to_now_0_1min__appevent_transferomitseecl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferperbookcl double
, sum__duration__time_to_now_0_1min__appevent_transferperbookcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferperpaycardcl double
, sum__duration__time_to_now_0_1min__appevent_transferperpaycardcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferperpaybalacl double
, sum__duration__time_to_now_0_1min__appevent_transferperpaybalacl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferperopenpostcl double
, sum__duration__time_to_now_0_1min__appevent_transferperopenpostcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferperhidepostcl double
, sum__duration__time_to_now_0_1min__appevent_transferperhidepostcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferperpostcl double
, sum__duration__time_to_now_0_1min__appevent_transferperpostcl double
, cnt__hiseventid__time_to_now_0_1min__transferpermoneyconfcl_ double
, sum__duration__time_to_now_0_1min__transferpermoneyconfcl_ double
, cnt__hiseventid__time_to_now_0_1min__appevent_transfercardbookcl double
, sum__duration__time_to_now_0_1min__appevent_transfercardbookcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transfercardnamecl double
, sum__duration__time_to_now_0_1min__appevent_transfercardnamecl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transfercardcardnumcl double
, sum__duration__time_to_now_0_1min__appevent_transfercardcardnumcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transfercardselectcl double
, sum__duration__time_to_now_0_1min__appevent_transfercardselectcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transfercardbalacl double
, sum__duration__time_to_now_0_1min__appevent_transfercardbalacl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transfercardopenpostcl double
, sum__duration__time_to_now_0_1min__appevent_transfercardopenpostcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transfercardhidepostcl double
, sum__duration__time_to_now_0_1min__appevent_transfercardhidepostcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transfercardpostcl double
, sum__duration__time_to_now_0_1min__appevent_transfercardpostcl double
, cnt__hiseventid__time_to_now_0_1min__transfercardmoneyconfcl_ double
, sum__duration__time_to_now_0_1min__transfercardmoneyconfcl_ double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferuserphonecl double
, sum__duration__time_to_now_0_1min__appevent_transferuserphonecl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferuseraddbookcl double
, sum__duration__time_to_now_0_1min__appevent_transferuseraddbookcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferusernextcl double
, sum__duration__time_to_now_0_1min__appevent_transferusernextcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferusernextnamecl double
, sum__duration__time_to_now_0_1min__appevent_transferusernextnamecl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferusernextcardcl double
, sum__duration__time_to_now_0_1min__appevent_transferusernextcardcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferusernextbalacl double
, sum__duration__time_to_now_0_1min__appevent_transferusernextbalacl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferusernextopenpostcl double
, sum__duration__time_to_now_0_1min__appevent_transferusernextopenpostcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferusernexthidepostcl double
, sum__duration__time_to_now_0_1min__appevent_transferusernexthidepostcl double
, cnt__hiseventid__time_to_now_0_1min__appevent_transferusernextpostcl double
, sum__duration__time_to_now_0_1min__appevent_transferusernextpostcl double
, cnt__hiseventid__time_to_now_0_1min__transferusermoneyconfcl_ double
, sum__duration__time_to_now_0_1min__transferusermoneyconfcl_ double
, cnt__hiseventid__time_to_now_0_1min__appstart_ double
, sum__duration__time_to_now_0_1min__appstart_ double
, cnt__hiseventid__time_to_now_0_1min__append_ double
, sum__duration__time_to_now_0_1min__append_ double
, cnt__hiseventid__time_to_now_0_1min__apppageview_mainpg double
, sum__duration__time_to_now_0_1min__apppageview_mainpg double
, cnt__hiseventid__time_to_now_0_1min__bottomappcl_ double
, sum__duration__time_to_now_0_1min__bottomappcl_ double
, cnt__hiseventid__hisseriesid_0 double
, cntdist__hiseventid__hisseriesid_0 double
, sum__duration__hisseriesid_0 double
, cnt__hiseventid__hisseriesid_0__apppageview_transfermoneypg double
, sum__duration__hisseriesid_0__apppageview_transfermoneypg double
, cnt__hiseventid__hisseriesid_0__appevent_transferunicardcl double
, sum__duration__hisseriesid_0__appevent_transferunicardcl double
, cnt__hiseventid__hisseriesid_0__appevent_transferunipercl double
, sum__duration__hisseriesid_0__appevent_transferunipercl double
, cnt__hiseventid__hisseriesid_0__appevent_transferinfpercl double
, sum__duration__hisseriesid_0__appevent_transferinfpercl double
, cnt__hiseventid__hisseriesid_0__transferdetcl_ double
, sum__duration__hisseriesid_0__transferdetcl_ double
, cnt__hiseventid__hisseriesid_0__transferdetpercl_ double
, sum__duration__hisseriesid_0__transferdetpercl_ double
, cnt__hiseventid__hisseriesid_0__transferonepercl_ double
, sum__duration__hisseriesid_0__transferonepercl_ double
, cnt__hiseventid__hisseriesid_0__appevent_transferseapercl double
, sum__duration__hisseriesid_0__appevent_transferseapercl double
, cnt__hiseventid__hisseriesid_0__transferdetpercl_ double
, sum__duration__hisseriesid_0__transferdetpercl_ double
, cnt__hiseventid__hisseriesid_0__transferonepercl_ double
, sum__duration__hisseriesid_0__transferonepercl_ double
, cnt__hiseventid__hisseriesid_0__transferomitpercl_ double
, sum__duration__hisseriesid_0__transferomitpercl_ double
, cnt__hiseventid__hisseriesid_0__appevent_transferomitsurecl double
, sum__duration__hisseriesid_0__appevent_transferomitsurecl double
, cnt__hiseventid__hisseriesid_0__appevent_transferomitseecl double
, sum__duration__hisseriesid_0__appevent_transferomitseecl double
, cnt__hiseventid__hisseriesid_0__appevent_transferperbookcl double
, sum__duration__hisseriesid_0__appevent_transferperbookcl double
, cnt__hiseventid__hisseriesid_0__appevent_transferperpaycardcl double
, sum__duration__hisseriesid_0__appevent_transferperpaycardcl double
, cnt__hiseventid__hisseriesid_0__appevent_transferperpaybalacl double
, sum__duration__hisseriesid_0__appevent_transferperpaybalacl double
, cnt__hiseventid__hisseriesid_0__appevent_transferperopenpostcl double
, sum__duration__hisseriesid_0__appevent_transferperopenpostcl double
, cnt__hiseventid__hisseriesid_0__appevent_transferperhidepostcl double
, sum__duration__hisseriesid_0__appevent_transferperhidepostcl double
, cnt__hiseventid__hisseriesid_0__appevent_transferperpostcl double
, sum__duration__hisseriesid_0__appevent_transferperpostcl double
, cnt__hiseventid__hisseriesid_0__transferpermoneyconfcl_ double
, sum__duration__hisseriesid_0__transferpermoneyconfcl_ double
, cnt__hiseventid__hisseriesid_0__appevent_transfercardbookcl double
, sum__duration__hisseriesid_0__appevent_transfercardbookcl double
, cnt__hiseventid__hisseriesid_0__appevent_transfercardnamecl double
, sum__duration__hisseriesid_0__appevent_transfercardnamecl double
, cnt__hiseventid__hisseriesid_0__appevent_transfercardcardnumcl double
, sum__duration__hisseriesid_0__appevent_transfercardcardnumcl double
, cnt__hiseventid__hisseriesid_0__appevent_transfercardselectcl double
, sum__duration__hisseriesid_0__appevent_transfercardselectcl double
, cnt__hiseventid__hisseriesid_0__appevent_transfercardbalacl double
, sum__duration__hisseriesid_0__appevent_transfercardbalacl double
, cnt__hiseventid__hisseriesid_0__appevent_transfercardopenpostcl double
, sum__duration__hisseriesid_0__appevent_transfercardopenpostcl double
, cnt__hiseventid__hisseriesid_0__appevent_transfercardhidepostcl double
, sum__duration__hisseriesid_0__appevent_transfercardhidepostcl double
, cnt__hiseventid__hisseriesid_0__appevent_transfercardpostcl double
, sum__duration__hisseriesid_0__appevent_transfercardpostcl double
, cnt__hiseventid__hisseriesid_0__transfercardmoneyconfcl_ double
, sum__duration__hisseriesid_0__transfercardmoneyconfcl_ double
, cnt__hiseventid__hisseriesid_0__appevent_transferuserphonecl double
, sum__duration__hisseriesid_0__appevent_transferuserphonecl double
, cnt__hiseventid__hisseriesid_0__appevent_transferuseraddbookcl double
, sum__duration__hisseriesid_0__appevent_transferuseraddbookcl double
, cnt__hiseventid__hisseriesid_0__appevent_transferusernextcl double
, sum__duration__hisseriesid_0__appevent_transferusernextcl double
, cnt__hiseventid__hisseriesid_0__appevent_transferusernextnamecl double
, sum__duration__hisseriesid_0__appevent_transferusernextnamecl double
, cnt__hiseventid__hisseriesid_0__appevent_transferusernextcardcl double
, sum__duration__hisseriesid_0__appevent_transferusernextcardcl double
, cnt__hiseventid__hisseriesid_0__appevent_transferusernextbalacl double
, sum__duration__hisseriesid_0__appevent_transferusernextbalacl double
, cnt__hiseventid__hisseriesid_0__appevent_transferusernextopenpostcl double
, sum__duration__hisseriesid_0__appevent_transferusernextopenpostcl double
, cnt__hiseventid__hisseriesid_0__appevent_transferusernexthidepostcl double
, sum__duration__hisseriesid_0__appevent_transferusernexthidepostcl double
, cnt__hiseventid__hisseriesid_0__appevent_transferusernextpostcl double
, sum__duration__hisseriesid_0__appevent_transferusernextpostcl double
, cnt__hiseventid__hisseriesid_0__transferusermoneyconfcl_ double
, sum__duration__hisseriesid_0__transferusermoneyconfcl_ double
, cnt__hiseventid__hisseriesid_0__appstart_ double
, sum__duration__hisseriesid_0__appstart_ double
, cnt__hiseventid__hisseriesid_0__append_ double
, sum__duration__hisseriesid_0__append_ double
, cnt__hiseventid__hisseriesid_0__apppageview_mainpg double
, sum__duration__hisseriesid_0__apppageview_mainpg double
, cnt__hiseventid__hisseriesid_0__bottomappcl_ double
, sum__duration__hisseriesid_0__bottomappcl_ double
, min__time_to_now__series_rownum_desc_0__hisseriesid_0 double
, min__time_to_now__series_rownum_desc_1__hisseriesid_0 double
, min__time_to_now__series_rownum_desc_2__hisseriesid_0 double
, min__time_to_now__series_rownum_desc_3__hisseriesid_0 double
, max__series_duration__hisseriesid_0 double
