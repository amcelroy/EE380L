SELECT c.subject_id,
       c.hadm_id,
       c.hospital_expire_flag,
       Date_diff(Date(c.admittime), DATE(c.dob), YEAR) AS age
FROM
  (SELECT a.subject_id,
          a.hadm_id,
          a.hospital_expire_flag,
          b.dob,
          a.admittime
   FROM `physionet-data.mimiciii_clinical.admissions` AS a
   LEFT JOIN `physionet-data.mimiciii_clinical.patients` AS b ON a.subject_id = b.subject_id) AS c-- possibly use icustay instread of hadm
-- get total count of drug types and average length on each drug

SELECT a.subject_id,
       a.hadm_id,
       count(a.drug) AS NumDrugs
FROM
  (SELECT subject_id,
          hadm_id,
          drug
   FROM `physionet-data.mimiciii_clinical.prescriptions`) AS a
GROUP BY 2,
         1 -- number of procedures

SELECT subject_id,
       hadm_id,
       count(icd9_code) AS num_procedures
FROM `physionet-data.mimiciii_clinical.procedures_icd`
GROUP BY 2,
         1 -- number of services
 -- current service

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
   FROM `physionet-data.mimiciii_clinical.services`) AS a
WHERE a.serve_order = 1 -- num transfers, curr care unit

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
              FROM `physionet-data.mimiciii_clinical.transfers`) a) b
        WHERE b.curr_careunit IS NOT NULL) c) d WHERE d.rownum = 1 -- average length of stay, current length of stay

  SELECT a.subject_id,
         a.hadm_id,
         avg(los) AS avg_los,
         sum(los) AS tot_los
  FROM `physionet-data.mimiciii_clinical.transfers` a WHERE a.los IS NOT NULL
GROUP BY 2,
         1 -- type of caregiver
 -- next add chart events

SELECT subject_id,
       hadm_id,
       count(DISTINCT itemid) AS num_unique_reads,
       count(itemid) AS total_reads,
       count(DISTINCT cgid) AS uinique_caregivers
FROM `physionet-data.mimiciii_clinical.chartevents`
GROUP BY 2,
         1 -- tot num procedures

SELECT subject_id,
       hadm_id,
       max(seq_num) AS total_icd9
FROM `physionet-data.mimiciii_clinical.diagnoses_icd`
GROUP BY 2,
         1 -- average time in ICU (hours)
 -- total time in ICU (hours)

SELECT a.subject_id,
       a.hadm_id,
       sum(icutime) AS total_icu_hours,
       avg(icutime) AS avg_icu_hours,
       count(icustay_id) AS total_icu_stays
FROM
  (SELECT subject_id,
          hadm_id,
          date_diff(date(outtime), date(intime), DAY) AS icutime,
          icustay_id
   FROM `physionet-data.mimiciii_clinical.icustays`) a
GROUP BY 2,
         1 -- total drugs admninisteres, avg num, total routes administetred, max administered at once

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
      FROM `physionet-data.mimiciii_clinical.inputevents_cv`
      GROUP BY linkorderid,
               2,
               1) a
   GROUP BY 2,
            1) c
RIGHT JOIN
  (SELECT subject_id,
          hadm_id,
          count(orderid) AS total_input_drugs,
          count(DISTINCT originalroute) AS tot_routes
   FROM `physionet-data.mimiciii_clinical.inputevents_cv`
   GROUP BY 2,
            1) b ON C.hadm_id = b.hadm_id
SELECT subject_id,
       hadm_id,
       patientweight
FROM `physionet-data.mimiciii_clinical.inputevents_mv`
GROUP BY 2,
         1,
         3 -- microbiology events, organism type, num orgamisms

SELECT hadm_id,
       count(DISTINCT org_itemid) AS tot_org
FROM `physionet-data.mimiciii_clinical.microbiologyevents`
GROUP BY 1
SELECT a.*
FROM
  (SELECT subject_id,
          hadm_id,
          org_itemid,
          org_name,
          charttime,
          row_number() OVER (PARTITION BY hadm_id
                             ORDER BY charttime DESC) AS rownum
   FROM `physionet-data.mimiciii_clinical.microbiologyevents`
   WHERE ORG_ITEMID IS NOT NULL) a
WHERE a.rownum = 1