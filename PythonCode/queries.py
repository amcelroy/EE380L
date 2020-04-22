# age, expire flag
Patient_pysql = """
SELECT c.subject_id,
       c.hadm_id,
       c.hospital_expire_flag,
       strftime('%Y', c.admittime) - strftime('%Y', c.dob) AS age
FROM
  (SELECT a.subject_id,
          a.hadm_id,
          a.hospital_expire_flag,
          b.dob,
          a.admittime
   FROM ADMISSIONS a
   LEFT JOIN PATIENTS b ON a.subject_id = b.subject_id) c
"""

# total number of drugs
prescription_pysql = """
SELECT a.subject_id,
       a.hadm_id,
       count(a.drug) NumDrugs
FROM
  (SELECT subject_id,
          hadm_id,
          drug
   FROM PRESCRIPTIONS) a
GROUP BY 2 -- number of procedures
"""

# number of procedures
procedures_sql = """
SELECT subject_id,
       hadm_id,
       count(icd9_code) AS num_procedures
FROM PROCEDURES_ICD
GROUP BY 2
"""

services_sql = """
SELECT a.subject_id,
       a.hadm_id,
       a.curr_service,
       a.num_serv
FROM
  (SELECT subject_id,
          hadm_id,
          curr_service,
          ROW_NUMBER() OVER (PARTITION BY hadm_id
                             ORDER BY transfertime DESC) AS serve_order,
                            count(curr_service) OVER (PARTITION BY hadm_id) num_serv
   FROM SERVICES) a
WHERE a.serve_order = 1
"""

transfers1_sql = """
  SELECT d.subject_id,
         d.hadm_id,
         d.num_transfers,
         d.curr_careunit
  FROM
    (SELECT *,
            ROW_NUMBER() OVER (PARTITION BY c.hadm_id
                               ORDER BY c.outtime DESC) AS rownum
     FROM
       (SELECT *
        FROM
          (SELECT a.subject_id,
                  a.hadm_id,
                  sum(transfer) OVER (PARTITION BY a.hadm_id) num_transfers,
                                     a.curr_careunit,
                                     los,
                                     outtime
           FROM
             (SELECT subject_id,
                     hadm_id,
                     CASE
                         WHEN eventtype = 'transfer' THEN 1
                         ELSE 0
                     END AS transfer,
                     curr_careunit,
                     los,
                     outtime
              FROM TRANSFERS) a) b
        WHERE b.curr_careunit IS NOT NULL) c) d WHERE d.rownum = 1
"""

transfers2_sql = """
  SELECT a.subject_id,
         a.hadm_id,
         avg(los) AS avg_los,
         sum(los) AS tot_los
  FROM TRANSFERS a WHERE a.los IS NOT NULL
GROUP BY a.hadm_id
"""

chartevents_sql = """
SELECT subject_id,
       hadm_id,
       count(DISTINCT itemid) AS num_unique_reads,
       count(itemid) AS total_reads,
       count(DISTINCT cgid) as uinique_caregivers
FROM CHARTEVENTS
GROUP BY 2
"""

icd9_sql = """
SELECT subject_id,
       hadm_id,
       max(seq_num) AS total_icd9
FROM DIAGNOSES_ICD
GROUP BY 2
"""

icu_time_sql = """
SELECT a.subject_id,
       a.hadm_id,
       sum(icutime) AS total_icu_hours,
       avg(icutime) AS avg_icu_hours,
       count(icustay_id) AS total_icu_stays
FROM
  (SELECT subject_id,
          hadm_id,
          strftime('%d', outtime) - strftime('%d', intime) AS icutime,
          icustay_id
   FROM ICUSTAYS) a
GROUP BY 2
"""
inputevents_cv_sql = """
SELECT c.*,
       b.total_input_drugs,
       b.tot_routes
FROM
  (SELECT a.subject_id,
          a.hadm_id,
          avg(a.total_together) AS avg_num_drug_administered,
          max(a.total_together) AS max_drug_administered
   FROM
     (SELECT subject_id,
             hadm_id,
             count(linkorderid) AS total_together
      FROM INPUTEVENTS_CV
      GROUP BY linkorderid) a
   GROUP BY 2) c
CROSS JOIN
  (SELECT subject_id,
          hadm_id,
          count(orderid) AS total_input_drugs,
          count(DISTINCT originalroute) AS tot_routes
   FROM INPUTEVENTS_CV
   GROUP BY 2) b ON c.hadm_id == b.hadm_id
"""
inputevents_mv_sql = """
SELECT subject_id,
       hadm_id,
       patientweight
FROM INPUTEVENTS_MV
GROUP BY 2
"""
microbiology_sql = """
SELECT subject_id,
       hadm_id,
       org_itemid,
       org_name,
       count(DISTINCT org_itemid) as tot_org
FROM MICROBIOLOGYEVENTS
GROUP BY 2
"""
all_sql = """
SELECT a.subject_id,
       a.hadm_id,
       a.hospital_expire_flag,
       a.age,
       b.NumDrugs,
       c.num_procedures,
       d.curr_service,
       d.num_serv,
       e.num_transfers,
       e.curr_careunit,
       f.avg_los,
       f.tot_los,
       g.num_unique_reads,
       g.total_reads,
       g.uinique_caregivers,
       h.total_icd9,
       i.total_icu_hours,
       i.avg_icu_hours,
       i.total_icu_stays,
       j.avg_num_drug_administered,
       j.max_drug_administered,
       j.total_input_drugs,
       j.tot_routes,
       k.patientweight,
       l.tot_org,
       m.org_name,
       m.org_itemid
FROM a
LEFT JOIN b ON a.hadm_id == b.hadm_id
LEFT JOIN c ON a.hadm_id == c.hadm_id
LEFT JOIN d ON a.hadm_id == d.hadm_id
LEFT JOIN e ON a.hadm_id == e.hadm_id
LEFT JOIN f ON a.hadm_id == f.hadm_id
LEFT JOIN g ON a.hadm_id == g.hadm_id
LEFT JOIN h ON a.hadm_id == h.hadm_id
LEFT JOIN i ON a.hadm_id == i.hadm_id
LEFT JOIN j ON a.hadm_id == j.hadm_id
LEFT JOIN k ON a.hadm_id == k.hadm_id
LEFT JOIN l ON a.hadm_id == l.hadm_id
LEFT JOIN m on a.hadm_id == m.hadm_id
"""
