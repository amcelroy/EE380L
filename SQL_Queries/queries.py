# age, expire flag
Patient_pysql = """
Select c.subject_id,c.hadm_id, c.hospital_expire_flag, strftime('%Y',c.admittime) - strftime('%Y',c.dob) as age
from
(Select a.subject_id, a.hadm_id, a.hospital_expire_flag, b.dob, a.admittime
from ADMISSIONS a
left join PATIENTS b
on a.subject_id = b.subject_id) c
"""

# total number of drugs
prescription_pysql = """
Select a.subject_id, a.hadm_id, count(a.drug) NumDrugs
from
(Select subject_id, hadm_id, drug
from PRESCRIPTIONS) a
group by 2
"""

# number of procedures
procedures_sql = """
Select subject_id, hadm_id, count(icd9_code) as num_procedures
from PROCEDURES_ICD
group by 2
"""

services_sql = """
Select a.subject_id, a.hadm_id, a.curr_service, a.num_serv
from
(Select subject_id, hadm_id, curr_service, ROW_NUMBER() over (partition by hadm_id order by transfertime desc) as serve_order, count(curr_service) over (partition by hadm_id) num_serv
from SERVICES) a
where a.serve_order = 1
"""

transfers1_sql = """
select d.subject_id, d.hadm_id, d.num_transfers,d.curr_careunit
from
(select *, ROW_NUMBER() over (partition by c.hadm_id order by c.outtime desc) as rownum
from
(select *
from
(select a.subject_id, a.hadm_id, sum(transfer) over (partition by a.hadm_id) num_transfers, a.curr_careunit, los, outtime
from
(Select subject_id, hadm_id, case when eventtype = 'transfer' then 1 else 0 end as transfer, curr_careunit, los, outtime
from TRANSFERS) a) b
where b.curr_careunit is not null) c) d
where d.rownum = 1
"""

transfers2_sql = """
select a.subject_id, a.hadm_id, avg(los) as avg_los, sum(los) as tot_los
from TRANSFERS a
where a.los is not null
group by a.hadm_id
"""
