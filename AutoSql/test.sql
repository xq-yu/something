--逾期黑名单

create table udft.cs_usr_overdue  stored as parquet as
select user_id,max(due_days)  as due_days
from
-- 新网
(select T1.user_id,datediff(payment_due_date,add_months(firstpaymentduedate,mth-1)) as due_days
    from
    (select * from
        (select user_id,partner_order_no,payment_due_date,
            case when  tenorno like '%,%'  then cast(split(tenorno,',')[0] as int) else cast(tenorno as int) end as mth,
            row_number() over(partition by partner_order_no order by cast(penaltyintamount  as double) DESC ) as rn
            from  ods.mk_credit_repayment
            where cast(penaltyintamount as double)>0  and status='2' and to_date(create_date)<='2020-08-31'
        )w1 where rn=1
    )T1
    inner join upw_hive.mk_credit_loan_apply  T2
    on T1.partner_order_no=T2.partner_order_no
union all
-- 广发的无法计算逾期时间+广发逾期人数极少，暂时忽略
-- 农商
select T1.user_id,datediff(payment_due_date,add_months(firstpaymentduedate,mth-1)) as due_days
    from
    (select * from
        (select user_id,partner_order_no,payment_due_date,
            case when  tenorno like '%,%'  then cast(split(tenorno,',')[0] as int)
                else cast(tenorno as int) end as mth,
            row_number() over(partition by partner_order_no order by cast(penaltyintamount  as double) DESC ) as rn
            from  ods.mk_shns_credit_repayment
                where cast(penaltyintamount as double)>0  and status='2' and to_date(create_date)<='2020-08-31'
            )w1 where rn=1
    )T1
    inner join ods.mk_shns_credit_loan_apply  T2
    on T1.partner_order_no=T2.partner_order_no
union all
-- 平安银行
select T1.user_id, datediff(payment_due_date,add_months(to_date(payment_time),mth)) as due_days
    from
    (select * from
        (select user_id,payment_no,repayment_period as mth,to_date(create_date) as payment_due_date,
            row_number() over(partition by payment_no order by create_date  DESC ) as rn
            from  ods.mk_pa_credit_repayment
            where repayment_type='OVERDUE' and to_date(create_date)<='2020-08-31'
            )w1 where rn=1
    ) T1
    inner join ods.mk_pa_credit_loan_apply  T2
    on T1.payment_no=T2.payment_no

) T   where due_days>3
group by user_id
;


--计算逾期期数的个数和逾期金额
create table udft.cs_usr_overdue_times_amt stored as parquet as
--新网
select user_id, max(last_payment_due_date) as last_payment_due_date, sum( ovd_times) as ovd_times, sum(ovd_total_amt) as ovd_total_amt
from (
    select user_id, partner_order_no
          ,count( distinct single_ternorno) as ovd_times
          ,sum( case when rn=1 then totalamount else 0 end ) as ovd_amt
		  ,count( distinct single_ternorno)*sum( case when rn=1 then totalamount else 0 end ) as ovd_total_amt
          ,max( payment_due_date) as last_payment_due_date
    from (
      select user_id,partner_order_no,totalamount,payment_due_date,single_ternorno
            ,row_number() over(partition by partner_order_no order by cast(totalamount  as double) DESC ) as rn
      from  ods.mk_credit_repayment lateral view explode(split(ternorno,',')) t as single_ternorno
      where cast(penaltyintamount as double)>0  and status='2' and to_date(create_date)<='2020-08-31'
    ) t01
	group by user_id, partner_order_no
) t02
group by user_id

union all
--农商
select user_id, max( last_payment_due_date ) as last_payment_due_date, sum( ovd_times) as ovd_times, sum(ovd_total_amt) as ovd_total_amt
from (
    select user_id, partner_order_no
          ,count( distinct single_ternorno) as ovd_times
          ,sum( case when rn=1 then totalamount else 0 end ) as ovd_amt
		  ,count( distinct single_ternorno)*sum( case when rn=1 then totalamount else 0 end ) as ovd_total_amt
          ,max( payment_due_date) as last_payment_due_date
    from (
      select user_id,partner_order_no,totalamount,payment_due_date,single_ternorno
            ,row_number() over(partition by partner_order_no order by cast(totalamount  as double) DESC ) as rn
      from  ods.mk_shns_credit_repayment lateral view explode(split(ternorno,',')) t as single_ternorno
      where cast(penaltyintamount as double)>0  and status='2' and to_date(create_date)<='2020-08-31'
    ) t01
	group by user_id, partner_order_no
) t02
group by user_id


union all
--平安
select user_id, max(last_payment_due_date) as last_payment_due_date, sum( ovd_times) as ovd_times, sum(ovd_total_amt) as ovd_total_amt
from (
    select user_id, partner_order_no
          ,count( distinct single_ternorno) as ovd_times
          ,sum( case when rn=1 then totalamount else 0 end ) as ovd_amt
		      ,count( distinct single_ternorno)*sum( case when rn=1 then totalamount else 0 end ) as ovd_total_amt
          ,max( payment_due_date) as last_payment_due_date
    from (
      select user_id,payment_no as partner_order_no,repayment_amount as totalamount,to_date(create_date) as payment_due_date,single_ternorno
            ,row_number() over(partition by payment_no order by cast(repayment_amount  as double) DESC ) as rn
      from  ods.mk_pa_credit_repayment lateral view explode(split(repayment_period,',')) t as single_ternorno
      where repayment_type='OVERDUE' and to_date(create_date)<='2020-08-31'
    ) t01
	group by user_id, partner_order_no
) t02
group by user_id
;

-- 拼接
create table udft.cs_usr_overdue_times_amt_all stored as parquet as
select T4.certif_id
      ,max(a.due_days) as due_days
      ,max(b.last_payment_due_date) as last_payment_due_date
      ,sum(b.ovd_times) as ovd_times
      ,sum(b.ovd_total_amt) as ovd_total_amt
from udft.cs_usr_overdue a
left outer join udft.cs_usr_overdue_times_amt b on a.user_id=b.user_id
left join ods.mk_user_info T2
on a.user_id=cast(T2.id as string)
left join upw_hive.view_chacc_cdhd_pri_acct_inf T3
on T2.phone=T3.mobile
left join upw_hive.view_ucbiz_cdhd_ext_inf  T4
on T3.cdhd_usr_id=T4.usr_id
group by T4.certif_id
;