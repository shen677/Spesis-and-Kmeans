-- select
--      *
--   from eicuii.lab
--   where labname ~*
--   (
--       'bnp'
--   );


DROP TABLE IF EXISTS pivoted_lab1 CASCADE;
CREATE TABLE pivoted_lab1 as
with vw0 as
(
  select
      patientunitstayid
    , labname
    , labresultoffset
    , labresultrevisedoffset
  from eicuii.lab
  where labname in
  (
      'RBC'
    , 'Hgb'
		, 'Hct'
    , 'RDW'
    , 'MCH'
    , 'MCV'
    , 'MCHC'
    , 'platelets x 1000'
    , 'WBC x 1000' -- HCO3
    , '-basos'
    , '-eos'
    , '-lymphs'
    , '-monos'
    , '-polys'
    , 'albumin'
    , 'anion gap'
    , 'BUN'
    , 'calcium'
    , 'chloride'
    , 'sodium'
    -- Liver enzymes
    , 'potassium'
    , 'glucose','bedside glucose'
    , 'creatinine'
		, 'PT - INR'
		, 'PT'
		, 'PTT'
		, 'lactate'
		, 'troponin - I'
		, 'troponin - T'
		, 'BNP'
  )
  group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
  having count(distinct labresult)<=1
)
-- get the last lab to be revised
, vw1 as
(
  select
      lab.patientunitstayid
    , lab.labname
    , lab.labresultoffset
    , lab.labresultrevisedoffset
    , lab.labresult
    , ROW_NUMBER() OVER
        (
          PARTITION BY lab.patientunitstayid, lab.labname, lab.labresultoffset
          ORDER BY lab.labresultrevisedoffset DESC
        ) as rn
  from eicuii.lab
  inner join vw0
    ON  lab.patientunitstayid = vw0.patientunitstayid
    AND lab.labname = vw0.labname
    AND lab.labresultoffset = vw0.labresultoffset
    AND lab.labresultrevisedoffset = vw0.labresultrevisedoffset
  -- only valid lab values
  WHERE
       (lab.labname = 'albumin' and lab.labresult >= 0.5 and lab.labresult <= 6.5)
    OR (lab.labname = 'BUN' and lab.labresult >= 1 and lab.labresult <= 280)
    OR (lab.labname = 'calcium' and lab.labresult > 0 and lab.labresult <= 9999)
    OR (lab.labname = 'chloride' and lab.labresult > 0 and lab.labresult <= 9999)
    OR (lab.labname = 'creatinine' and lab.labresult >= 0.1 and lab.labresult <= 28.28)
    OR (lab.labname in ('bedside glucose', 'glucose') and lab.labresult >= 25 and lab.labresult <= 1500)
    OR (lab.labname = 'Hct' and lab.labresult >= 5 and lab.labresult <= 75)
    OR (lab.labname = 'Hgb' and lab.labresult >  0 and lab.labresult <= 9999)
    OR (lab.labname = 'PT - INR' and lab.labresult >= 0.5 and lab.labresult <= 15)
    OR (lab.labname = 'platelets x 1000' and lab.labresult >  0 and lab.labresult <= 9999)
    OR (lab.labname = 'potassium' and lab.labresult >= 0.05 and lab.labresult <= 12)
    OR (lab.labname = 'PTT' and lab.labresult >  0 and lab.labresult <= 500)
		OR (lab.labname = 'PT' and lab.labresult >  0 and lab.labresult <= 500)
    OR (lab.labname = 'sodium' and lab.labresult >= 90 and lab.labresult <= 215)
    OR (lab.labname = 'WBC x 1000' and lab.labresult > 0 and lab.labresult <= 100)
		OR (lab.labname = 'RBC' and lab.labresult > 0 and lab.labresult <= 100)
		OR (lab.labname = 'RDW' and lab.labresult > 0 and lab.labresult <= 200)
		OR (lab.labname = 'MCH' and lab.labresult > 0 and lab.labresult <= 200)
		OR (lab.labname = 'MCV' and lab.labresult > 0 and lab.labresult <= 400)
		OR (lab.labname = 'MCHC' and lab.labresult > 0 and lab.labresult <= 400)
		OR (lab.labname = '-basos' and lab.labresult >= 0 and lab.labresult <= 100)
		OR (lab.labname = '-eos' and lab.labresult >= 0 and lab.labresult <= 100)
		OR (lab.labname = '-lymphs' and lab.labresult >= 0 and lab.labresult <= 100)
		OR (lab.labname = '-monos' and lab.labresult >= 0 and lab.labresult <= 100)
		OR (lab.labname = '-polys' and lab.labresult >= 0 and lab.labresult <= 100)
		OR (lab.labname = 'anion gap' and lab.labresult >= 0 and lab.labresult <= 100)
		OR (lab.labname = 'lactate' and lab.labresult >= 0 and lab.labresult <= 100)
		OR (lab.labname = 'anion gap' and lab.labresult >= 0 and lab.labresult <= 100)
		OR (lab.labname = 'lactate' and lab.labresult >= 0 and lab.labresult <= 100)
		OR (lab.labname = 'troponin - I' and lab.labresult >= 0 and lab.labresult <= 100)
		OR (lab.labname = 'troponin - T' and lab.labresult >= 0 and lab.labresult <= 100)
		OR (lab.labname = 'BNP' and lab.labresult >= 0)
)
select
    patientunitstayid
  , labresultoffset as chartoffset
	, MAX(case when labname = 'RBC' then labresult else null end) as rbc
	, MAX(case when labname = 'Hgb' then labresult else null end) as hemoglobin
	, MAX(case when labname = 'Hct' then labresult else null end) as hematocrit
	, MAX(case when labname = 'RDW' then labresult else null end) as rdw
	, MAX(case when labname = 'MCH' then labresult else null end) as mch
	, MAX(case when labname = 'MCV' then labresult else null end) as mcv
	, MAX(case when labname = 'MCHC' then labresult else null end) as mchc
	, MAX(case when labname = 'platelets x 1000' then labresult else null end) as platelet
	, MAX(case when labname = 'WBC x 1000' then labresult else null end) as wbc
	, MAX(case when labname = '-basos' then labresult else null end) as basophils
	, MAX(case when labname = '-eos' then labresult else null end) as eosinophils
	, MAX(case when labname = '-lymphs' then labresult else null end) as lymphocytes
	, MAX(case when labname = '-monos' then labresult else null end) as monocytes
	, MAX(case when labname = '-polys' then labresult else null end) as neutrophils
  , MAX(case when labname = 'albumin' then labresult else null end) as albumin
	, MAX(case when labname = 'anion gap' then labresult else null end) as aniongap
  , MAX(case when labname = 'BUN' then labresult else null end) as bun
  , MAX(case when labname = 'calcium' then labresult else null end) as calcium
	, MAX(case when labname = 'chloride' then labresult else null end) as chloride
	, MAX(case when labname = 'sodium' then labresult else null end) as sodium
	, MAX(case when labname = 'potassium' then labresult else null end) as potassium
	, MAX(case when labname in ('bedside glucose', 'glucose') then labresult else null end) as glucose
  , MAX(case when labname = 'creatinine' then labresult else null end) as creatinine
	, MAX(case when labname = 'PT - INR' then labresult else null end) as inr
	, MAX(case when labname = 'PT' then labresult else null end) as pt
	, MAX(case when labname = 'PTT' then labresult else null end) as ptt
	, MAX(case when labname = 'lactate' then labresult else null end) as lactate
	, MAX(case when labname = 'troponin - I' then labresult else null end) as ctni
	, MAX(case when labname = 'troponin - T' then labresult else null end) as ctnt
	, MAX(case when labname = 'BNP' then labresult else null end) as bnp
	
from vw1
where rn = 1
group by patientunitstayid, labresultoffset
order by patientunitstayid, labresultoffset;


DROP TABLE IF EXISTS pivoted_charlson CASCADE;
CREATE TABLE pivoted_charlson as
select
patientunitstayid,
max(case when SUBSTR(icd9code, 1, 3) in ('410','412','I21','I22','I252') then 1 else 0 end) myocardial_infarct,
max(case when SUBSTR(icd9code, 1, 3) = '428' then 1
		when SUBSTR(icd9code, 1, 5) IN ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493') then 1
		when SUBSTR(icd9code, 1, 4) BETWEEN '4254' AND '4259' then 1
		else 0 end) congestive_heart_failure,
MAX(CASE WHEN 
		SUBSTR(icd9code, 1, 3) IN ('440','441')
		OR
		SUBSTR(icd9code, 1, 4) IN ('0930','4373','4471','5571','5579','V434')
		OR
		SUBSTR(icd9code, 1, 4) BETWEEN '4431' AND '4439'
		THEN 1 
		ELSE 0 END) AS peripheral_vascular_disease,
MAX(CASE WHEN 
		SUBSTR(icd9code, 1, 3) BETWEEN '430' AND '438'
		OR
		SUBSTR(icd9code, 1, 5) = '36234'
		THEN 1 
		ELSE 0 END) AS cerebrovascular_disease,
MAX(CASE WHEN 
		LOWER(diagnosisstring) LIKE '%dementia%' 
		THEN 1 
		ELSE 0 END) AS dementia
-- Chronic pulmonary disease
, MAX(CASE WHEN 
		SUBSTR(icd9code, 1, 3) BETWEEN '490' AND '505'
		OR
		SUBSTR(icd9code, 1, 4) IN ('4168','4169','5064','5081','5088')
		THEN 1 
		ELSE 0 END) AS chronic_pulmonary_disease
, MAX(CASE WHEN 
		LOWER(diagnosisstring) LIKE '%immunological diseases%'
		or LOWER(diagnosisstring) LIKE '%rheumatic%'
		THEN 1 
		ELSE 0 END) AS rheumatic_disease
, MAX(CASE WHEN 
		LOWER(diagnosisstring) LIKE '%ulcer%' and LOWER(diagnosisstring) LIKE '%gastrointestinal%'
		THEN 1 
		ELSE 0 END) AS peptic_ulcer_disease
, MAX(CASE WHEN 
		LOWER(diagnosisstring) LIKE '%hepatic disease%'
		THEN 1 
		ELSE 0 END) AS liver_disease
-- Diabetes
, MAX(CASE WHEN 
		LOWER(diagnosisstring) LIKE '%diabetes%' and LOWER(diagnosisstring) not LIKE '%insipidus%' or LOWER(diagnosisstring) LIKE '%glycuresis%'
		THEN 1 
		ELSE 0 END) AS diabetes

-- Hemiplegia or paraplegia
, MAX(CASE WHEN 
		LOWER(diagnosisstring) LIKE '%paraplegia%' or LOWER(diagnosisstring) LIKE '%hemiplegia%'
		THEN 1 
		ELSE 0 END) AS paraplegia

-- Renal disease
, MAX(CASE WHEN 
		SUBSTR(icd9code, 1, 3) IN ('582','585','586','V56')
		OR
		SUBSTR(icd9code, 1, 4) IN ('5880','V420','V451')
		OR
		SUBSTR(icd9code, 1, 4) BETWEEN '5830' AND '5837'
		OR
		SUBSTR(icd9code, 1, 5) IN ('40301','40311','40391','40402','40403','40412','40413','40492','40493')          

		THEN 1 
		ELSE 0 END) AS renal_disease

-- Any malignancy, including lymphoma and leukemia, except malignant neoplasm of skin
, MAX(CASE WHEN 
		SUBSTR(icd9code, 1, 3) BETWEEN '140' AND '172'
		OR
		SUBSTR(icd9code, 1, 4) BETWEEN '1740' AND '1958'
		OR
		SUBSTR(icd9code, 1, 3) BETWEEN '200' AND '208'
		OR
		SUBSTR(icd9code, 1, 4) = '2386'
		THEN 1 
		ELSE 0 END) AS malignant_cancer

-- Metastatic solid tumor
, MAX(CASE WHEN 
		SUBSTR(icd9code, 1, 3) IN ('196','197','198','199')
		THEN 1 
		ELSE 0 END) AS metastatic_solid_tumor

-- AIDS/HIV
, MAX(CASE WHEN 
		diagnosisstring ~* 'AIDS' AND diagnosisstring ~*'infectious diseases'
		THEN 1 
		ELSE 0 END) AS aids
, max(case when
		diagnosisstring ~*'mesenteric ischemia'
		then 1
		else 0 end) as mesenteric_ischemia
from eicuii.diagnosis
GROUP BY patientunitstayid;

select 
sum(myocardial_infarct) myocardial_infarct
, sum(congestive_heart_failure) congestive_heart_failure
, sum(peripheral_vascular_disease) peripheral_vascular_disease
, sum(cerebrovascular_disease) cerebrovascular_disease
, sum(dementia) dementia
, sum(chronic_pulmonary_disease) chronic_pulmonary_disease
, sum(rheumatic_disease) rheumatic_disease
, sum(peptic_ulcer_disease) peptic_ulcer_disease
, sum(liver_disease) liver_disease
, sum(diabetes) diabetes
, sum(paraplegia) paraplegia
, sum(renal_disease) renal_disease
, sum(malignant_cancer) malignant_cancer
, sum(metastatic_solid_tumor ) metastatic_solid_tumor
, sum(aids) aids
from pivoted_charlson；





with sofa2 as
(select patientunitstayid,sofatotal as sofa from sofa_results where sofatotal>=2)
,infect as
(
select patientunitstayid,
max(case when infectdiseaseassessment isnull then 0 else 1 end) infect
from eicuii.careplaninfectiousdisease
where cplinfectdiseaseoffset between -1440*10 and 1440*10
group by patientunitstayid
)
select sofa2.patientunitstayid,sofa,infect
from infect
join sofa2
on sofa2.patientunitstayid = infect.patientunitstayid


DROP TABLE IF EXISTS pivoted_sofa14 CASCADE;
CREATE TABLE pivoted_sofa14 as
with sepsis as
(
with t1 as (select apache_groups.patientunitstayid,sofatotal
from apache_groups 
join sofa_results
on apache_groups.patientunitstayid = sofa_results.patientunitstayid
where apachedxgroup ~* 'sepsis' and sofatotal>=2)
select DISTINCT patientunitstayid from t1
)

, cardiovascular as(
with t1 as -- MAP
(


with tt1 as
(
select patientunitstayid,
min( case when noninvasivemean is not null then noninvasivemean else null end) as map
from eicuii.vitalaperiodic
where observationoffset between 1440*14 and 1440*15
group by patientunitstayid
), tt2 as
(
select patientunitstayid,
min( case when systemicmean is not null then systemicmean else null end) as map
from eicuii.vitalperiodic
where observationoffset between 1440*14 and 1440*15
group by patientunitstayid
)


select pt.patientunitstayid, case when tt1.map is not null then tt1.map
when tt2.map is not null then tt2.map
else null end as map
from eicuii.patient pt
left outer join tt1
on tt1.patientunitstayid=pt.patientunitstayid
left outer join tt2
on tt2.patientunitstayid=pt.patientunitstayid
order by pt.patientunitstayid
)
, t2 as --DOPAMINE
(
select distinct  patientunitstayid, max(
case when lower(drugname) like '%(ml/hr)%' then round(cast(drugrate as numeric)/3,3) -- rate in ml/h * 1600 mcg/ml / 80 kg / 60 min, to convert in mcg/kg/min
when lower(drugname) like '%(mcg/kg/min)%' then cast(drugrate as numeric)
else null end ) as dopa
from eicuii.infusiondrug id
where lower(drugname) like '%dopamine%' and infusionoffset between 1440*14 and 1440*15 and drugrate ~ '^[0-9]{0,5}$' and drugrate<>'' and drugrate<>'.'
group by patientunitstayid
order by patientunitstayid


), t3 as  --NOREPI
(
select distinct patientunitstayid, max(case when lower(drugname) like '%(ml/hr)%' and drugrate<>''  and drugrate<>'.' then round(cast(drugrate as numeric)/300,3) -- rate in ml/h * 16 mcg/ml / 80 kg / 60 min, to convert in mcg/kg/min
when lower(drugname) like '%(mcg/min)%' and drugrate<>'' and drugrate<>'.'  then round(cast(drugrate as numeric)/80 ,3)-- divide by 80 kg
when lower(drugname) like '%(mcg/kg/min)%' and drugrate<>'' and drugrate<>'.' then cast(drugrate as numeric)
else null end ) as norepi


from eicuii.infusiondrug id
where lower(drugname) like '%norepinephrine%'  and infusionoffset between 1440*14 and 1440*15  and drugrate ~ '^[0-9]{0,5}$' and drugrate<>'' and drugrate<>'.'
group by patientunitstayid
order by patientunitstayid


), t4 as  --DOBUTAMINE
(
select distinct patientunitstayid, 1 as dobu
from eicuii.infusiondrug id
where lower(drugname) like '%dobutamin%' and drugrate <>'' and drugrate<>'.' and drugrate <>'0' and drugrate ~ '^[0-9]{0,5}$' and infusionoffset between 1440*14 and 1440*15
order by patientunitstayid
)

select pt.patientunitstayid, t1.map, t2.dopa, t3.norepi, t4.dobu,
(case when dopa>=15 or norepi>0.1 then 4
when dopa>5 or (norepi>0 and norepi <=0.1) then 3
when dopa<=5 or dobu > 0 then 2
when map <70 then 1
else 0 end) as sofa_cardiovascular 
from eicuii.patient pt
left outer join t1
on t1.patientunitstayid=pt.patientunitstayid
left outer join t2
on t2.patientunitstayid=pt.patientunitstayid
left outer join t3
on t3.patientunitstayid=pt.patientunitstayid
left outer join t4
on t4.patientunitstayid=pt.patientunitstayid
order by pt.patientunitstayid
)

,respiration as (
with tempo2 as 
(
with tempo1 as
(
with t1 as 
--FIO2 from respchart
(
select *
from
(
select distinct patientunitstayid, max(cast(respchartvalue as numeric)) as rcfio2
-- , max(case when respchartvaluelabel = 'FiO2' then respchartvalue else null end) as fiO2
from eicuii.respiratorycharting
where respchartoffset between 1440*14 and 1440*15 and respchartvalue <> '' and respchartvalue ~ '^[0-9]{0,2}$'
group by patientunitstayid
) as tempo
where rcfio2 >20 -- many values are liters per minute!
order by patientunitstayid


), t2 as 
--FIO2 from nursecharting
(
select distinct patientunitstayid, max(cast(nursingchartvalue as numeric)) as ncfio2
from eicuii.nursecharting nc
where lower(nursingchartcelltypevallabel) like '%fio2%' and nursingchartvalue ~ '^[0-9]{0,2}$' and nursingchartentryoffset between 1440*14 and 1440*15
group by patientunitstayid


), t3 as 
--sao2 from vitalperiodic
(
select patientunitstayid,
min( case when sao2 is not null then sao2 else null end) as sao2
from eicuii.vitalperiodic
where observationoffset between 1440*14 and 1440*15
group by patientunitstayid


), t4 as 
--pao2 from lab
(
select patientunitstayid,
min(case when lower(labname) like 'pao2%' then labresult else null end) as pao2
from eicuii.lab
where labresultoffset between 1440*14 and 1440*15
group by patientunitstayid


), t5 as 
--airway type combining 3 sources (1=invasive)


(




with t1 as 
--airway type from respcare (1=invasive) (by resp therapist!!)
(
select distinct patientunitstayid,
max(case when airwaytype in ('Oral ETT','Nasal ETT','Tracheostomy') then 1 else NULL end) as airway  -- either invasive airway or NULL
from eicuii.respiratorycare
where respcarestatusoffset between 1440*14 and 1440*15


group by patientunitstayid-- , respcarestatusoffset
-- order by patientunitstayid-- , respcarestatusoffset
),


t2 as 
--airway type from respcharting (1=invasive)
(
select distinct patientunitstayid, 1 as ventilator
from eicuii.respiratorycharting rc
where respchartvalue like '%ventilator%'
or respchartvalue like '%vent%'
or respchartvalue like '%bipap%'
or respchartvalue like '%840%'
or respchartvalue like '%cpap%'
or respchartvalue like '%drager%'
or respchartvalue like 'mv%'
or respchartvalue like '%servo%'
or respchartvalue like '%peep%'
and respchartoffset between 1440*14 and 1440*15
group by patientunitstayid
-- order by patientunitstayid
),


t3 as 
--airway type from treatment (1=invasive)


(
select distinct patientunitstayid, max(case when treatmentstring in
('pulmonary|ventilation and oxygenation|mechanical ventilation',
'pulmonary|ventilation and oxygenation|tracheal suctioning',
'pulmonary|ventilation and oxygenation|ventilator weaning',
'pulmonary|ventilation and oxygenation|mechanical ventilation|assist controlled',
'pulmonary|radiologic procedures / bronchoscopy|endotracheal tube',
'pulmonary|ventilation and oxygenation|oxygen therapy (> 60%)',
'pulmonary|ventilation and oxygenation|mechanical ventilation|tidal volume 6-10 ml/kg',
'pulmonary|ventilation and oxygenation|mechanical ventilation|volume controlled',
'surgery|pulmonary therapies|mechanical ventilation',
'pulmonary|surgery / incision and drainage of thorax|tracheostomy',
'pulmonary|ventilation and oxygenation|mechanical ventilation|synchronized intermittent',
'pulmonary|surgery / incision and drainage of thorax|tracheostomy|performed during current admission for ventilatory support',
'pulmonary|ventilation and oxygenation|ventilator weaning|active',
'pulmonary|ventilation and oxygenation|mechanical ventilation|pressure controlled',
'pulmonary|ventilation and oxygenation|mechanical ventilation|pressure support',
'pulmonary|ventilation and oxygenation|ventilator weaning|slow',
'surgery|pulmonary therapies|ventilator weaning',
'surgery|pulmonary therapies|tracheal suctioning',
'pulmonary|radiologic procedures / bronchoscopy|reintubation',
'pulmonary|ventilation and oxygenation|lung recruitment maneuver',
'pulmonary|surgery / incision and drainage of thorax|tracheostomy|planned',
'surgery|pulmonary therapies|ventilator weaning|rapid',
'pulmonary|ventilation and oxygenation|prone position',
'pulmonary|surgery / incision and drainage of thorax|tracheostomy|conventional',
'pulmonary|ventilation and oxygenation|mechanical ventilation|permissive hypercapnea',
'surgery|pulmonary therapies|mechanical ventilation|synchronized intermittent',
'pulmonary|medications|neuromuscular blocking agent',
'surgery|pulmonary therapies|mechanical ventilation|assist controlled',
'pulmonary|ventilation and oxygenation|mechanical ventilation|volume assured',
'surgery|pulmonary therapies|mechanical ventilation|tidal volume 6-10 ml/kg',
'surgery|pulmonary therapies|mechanical ventilation|pressure support',
'pulmonary|ventilation and oxygenation|non-invasive ventilation',
'pulmonary|ventilation and oxygenation|non-invasive ventilation|face mask',
'pulmonary|ventilation and oxygenation|non-invasive ventilation|nasal mask',
'pulmonary|ventilation and oxygenation|mechanical ventilation|non-invasive ventilation',
'pulmonary|ventilation and oxygenation|mechanical ventilation|non-invasive ventilation|face mask',
'surgery|pulmonary therapies|non-invasive ventilation',
'surgery|pulmonary therapies|non-invasive ventilation|face mask',
'pulmonary|ventilation and oxygenation|mechanical ventilation|non-invasive ventilation|nasal mask',
'surgery|pulmonary therapies|non-invasive ventilation|nasal mask',
'surgery|pulmonary therapies|mechanical ventilation|non-invasive ventilation',
'surgery|pulmonary therapies|mechanical ventilation|non-invasive ventilation|face mask'
) then 1  else NULL end) as interface   -- either ETT/NiV or NULL
from eicuii.treatment
where treatmentoffset between 1440*14 and 1440*15
group by patientunitstayid-- , treatmentoffset, interface
order by patientunitstayid-- , treatmentoffset
), t4 as
(
select distinct patientunitstayid,
max(case when cplitemvalue like '%Intubated%' then 1 else NULL end) as airway  -- either invasive airway or NULL
from eicuii.careplangeneral
where cplitemoffset between 1440*14 and 1440*15
group by patientunitstayid -- , respcarestatusoffset


)


select pt.patientunitstayid,
case when t1.airway is not null or t2.ventilator is not null or t3.interface is not null or t4.airway is not null then 1 else null end as mechvent --summarize
from eicuii.patient pt
left outer join t1
on t1.patientunitstayid=pt.patientunitstayid
left outer join t2
on t2.patientunitstayid=pt.patientunitstayid
left outer join t3
on t3.patientunitstayid=pt.patientunitstayid
left outer join t4
on t4.patientunitstayid=pt.patientunitstayid order by pt.patientunitstayid




)


select pt.patientunitstayid, t3.sao2, t4.pao2, 
(case when t1.rcfio2>20 then t1.rcfio2 when t2.ncfio2 >20 then t2.ncfio2  when t1.rcfio2=1 or t2.ncfio2=1 then 100 else null end) as fio2, t5.mechvent
from eicuii.patient pt
left outer join t1
on t1.patientunitstayid=pt.patientunitstayid
left outer join t2
on t2.patientunitstayid=pt.patientunitstayid
left outer join t3
on t3.patientunitstayid=pt.patientunitstayid
left outer join t4
on t4.patientunitstayid=pt.patientunitstayid
left outer join t5
on t5.patientunitstayid=pt.patientunitstayid
-- order by pt.patientunitstayid
)


select *, -- coalesce(fio2,nullif(fio2,0),21) as fn, nullif(fio2,0) as nullifzero, coalesce(coalesce(nullif(fio2,0),21),fio2,21) as ifzero21 ,
coalesce(pao2,100)/coalesce(coalesce(nullif(fio2,0),21),fio2,21) as pf, coalesce(sao2,100)/coalesce(coalesce(nullif(fio2,0),21),fio2,21) as sf
from tempo1
)


select patientunitstayid, 
(case when pf <1 or sf <0.67 then 4  --COMPUTE SOFA RESPI
when pf between 1 and 2 or sf between 0.67 and 1.41 then 3
when pf between 2 and 3 or sf between 1.42 and 2.2 then 2
when pf between 3 and 4 or sf between 2.21 and 3.01 then 1
when pf > 4 or sf> 3.01 then 0 else 0 end ) as SOFA_respiration
from tempo2
order by patientunitstayid
)
, renal as
(
with t1 as --CREATININE
(
select pt.patientunitstayid,
max(case when lower(labname) like 'creatin%' then labresult else null end) as creat
from eicuii.patient pt
left outer join eicuii.lab
on pt.patientunitstayid=eicuii.lab.patientunitstayid
where labresultoffset between 1440*14 and 1440*15
group by pt.patientunitstayid


), t2 as --UO
(


with uotemp as
(
select patientunitstayid,
case when dayz=1 then sum(outputtotal) else null end as uod1
from
(


select distinct patientunitstayid, intakeoutputoffset, outputtotal,
(CASE
WHEN  (intakeoutputoffset) between 1440*14 and 1440*15 THEN 1
else null
end) as dayz
from eicuii.intakeoutput
where intakeoutputoffset between 1440*14 and 1440*15
order by patientunitstayid, intakeoutputoffset


) as temp
group by patientunitstayid, temp.dayz
)


select pt.patientunitstayid,
max(case when uod1 is not null then uod1 else null end) as UO
from eicuii.patient pt
left outer join uotemp
on uotemp.patientunitstayid=pt.patientunitstayid
group by pt.patientunitstayid


)


select pt.patientunitstayid, -- t1.creat, t2.uo,
(case 
--COMPUTE SOFA RENAL
when uo <200 or creat>5 then 4
when uo <500 or creat >3.5 then 3
when creat between 2 and 3.5 then 2
when creat between 1.2 and 2 then 1
else 0
end) as sofa_renal
from eicuii.patient pt
left outer join t1
on t1.patientunitstayid=pt.patientunitstayid
left outer join t2
on t2.patientunitstayid=pt.patientunitstayid
order by pt.patientunitstayid
-- group by pt.patientunitstayid, t1.creat, t2.uo
)
,sofa3others as
(
with t1 as --GCS
(
select patientunitstayid, sum(cast(physicalexamvalue as numeric)) as gcs
from eicuii.physicalexam pe
where (lower(physicalexampath) like '%gcs/eyes%'
or lower(physicalexampath) like '%gcs/verbal%'
or lower(physicalexampath) like '%gcs/motor%')
and physicalexamoffset between 1440*14 and 1440*15
group by patientunitstayid, physicalexamoffset
), t2 as
(
select pt.patientunitstayid,
max(case when lower(labname) like 'total bili%' then labresult else null end) as bili, --BILI
min(case when lower(labname) like 'platelet%' then labresult else null end) as plt --PLATELETS
from eicuii.patient pt
left outer join eicuii.lab
on pt.patientunitstayid=eicuii.lab.patientunitstayid
where labresultoffset between 1440*14 and 1440*15
group by pt.patientunitstayid
)


select distinct pt.patientunitstayid, min(t1.gcs) as gcs, max(t2.bili) as bili, min(t2.plt) as plt,
max(case when plt<20 then 4
when plt<50 then 3
when plt<100 then 2
when plt<150 then 1
else 0 end) as sofa_coagulation,
max(case when bili>12 then 4
when bili>6 then 3
when bili>2 then 2
when bili>1.2 then 1
else 0 end) as sofa_liver,
max(case when gcs=15 then 0
when gcs>=13 then 1
when gcs>=10 then 2
when gcs>=6 then 3
when gcs>=3 then 4
else 0 end) as sofa_cns
from eicuii.patient pt
left outer join t1
on t1.patientunitstayid=pt.patientunitstayid
left outer join t2
on t2.patientunitstayid=pt.patientunitstayid
group by pt.patientunitstayid, t1.gcs, t2.bili, t2.plt
order by pt.patientunitstayid
)
, sofa_re as
(
select sepsis.patientunitstayid,sofa_cardiovascular,SOFA_respiration sofa_respiration,sofa_renal,sofa_coagulation,sofa_liver,sofa_cns,
(sofa_cardiovascular+SOFA_respiration+sofa_renal+sofa_coagulation+sofa_liver+sofa_cns) sofa
from sepsis
left join cardiovascular on sepsis.patientunitstayid=cardiovascular.patientunitstayid
left join respiration on sepsis.patientunitstayid=respiration.patientunitstayid
left join renal on sepsis.patientunitstayid=renal.patientunitstayid
left join sofa3others on sepsis.patientunitstayid=sofa3others.patientunitstayid
)
select *
from sofa_re；
--where sofa_cardiovascular>=1 or SOFA_respiration>=2 or sofa_renal>=2 or sofa_coagulation>=2 or sofa_liver>=2 or sofa_cns>=2


create materialized view if not exists eicu_cci as 
with sepsis as
(
with t1 as (select apache_groups.patientunitstayid,sofatotal
from apache_groups 
join sofa_results
on apache_groups.patientunitstayid = sofa_results.patientunitstayid
where apachedxgroup ~* 'sepsis' and sofatotal>=2)
, t2 as(
select DISTINCT patientunitstayid from t1)
select icustay_detail.* from t2 left join icustay_detail on t2.patientunitstayid=icustay_detail.patientunitstayid
)

,bg as
(
select sepsis.patientunitstayid,round(avg(pao2),2) po2_avg,round(min(pao2),2) po2_min,round(max(pao2),2) po2_max,
round(avg(paco2),2) pco2_avg,round(min(paco2),2) pco2_min,round(max(paco2),2) pco2_max,
round(avg(fio2),2) fio2_avg,round(min(fio2),2) fio2_min,round(max(fio2),2) fio2_max,
round(avg(aniongap),2) aniongap_avg,round(min(aniongap),2) aniongap_min,round(max(aniongap),2) aniongap_max,
round(avg(baseexcess),2) baseexcess_avg,round(min(baseexcess),2) baseexcess_min,round(max(baseexcess),2) baseexcess_max,
round(avg(ph),2) ph_avg,round(min(ph),2) ph_min,round(max(ph),2) ph_max
from sepsis left join pivoted_bg on  sepsis.patientunitstayid=pivoted_bg.patientunitstayid 
where (chartoffset-hospitaladmitoffset)<=1440*3
GROUP BY sepsis.patientunitstayid
)
,vit as
(
with t1 as
(
select patientunitstayid,chartoffset,
heartrate heart_rate,
(case when ibp_systolic isnull then nibp_systolic else ibp_systolic end) sbp,
(case when ibp_diastolic isnull then nibp_diastolic else ibp_diastolic end) dbp,
(case when ibp_mean isnull then nibp_mean else ibp_mean end) mbp,
respiratoryrate resp_rate,
temperature
from pivoted_vital
)
select sepsis.patientunitstayid,
avg(heart_rate) heart_rate_avg,min(heart_rate) heart_rate_min,max(heart_rate) heart_rate_max,
avg(sbp) sbp_avg,min(sbp) sbp_min,max(sbp) sbp_max,
avg(dbp) dbp_avg,min(dbp) dbp_min,max(dbp) dbp_max,
avg(mbp) mbp_avg,min(mbp) mbp_min,max(mbp) mbp_max,
avg(resp_rate) resp_rate_avg,min(resp_rate) resp_rate_min,max(resp_rate) resp_rate_max,
avg(temperature) temperature_avg,min(temperature) temperature_min,max(temperature) temperature_max
from sepsis left join t1 on  sepsis.patientunitstayid=t1.patientunitstayid 
where (chartoffset-hospitaladmitoffset)<=1440*3
GROUP BY sepsis.patientunitstayid
)
, lab as
(
select sepsis.patientunitstayid,
avg(rbc) rbc_avg,min(rbc) rbc_min,max(rbc) rbc_max,
avg(hemoglobin) hemoglobin_avg,min(hemoglobin) hemoglobin_min,max(hemoglobin) hemoglobin_max,
avg(hematocrit) hematocrit_avg,min(hematocrit) hematocrit_min,max(hematocrit) hematocrit_max,
avg(rdw) rdw_avg,min(rdw) rdw_min,max(rdw) rdw_max,
avg(mch) mch_avg,min(mch) mch_min,max(mch) mch_max,
avg(mcv) mcv_avg,min(mcv) mcv_min,max(mcv) mcv_max,
avg(mchc) mchc_avg,min(mchc) mchc_min,max(mchc) mchc_max,
avg(platelet) platelet_avg,min(platelet) platelet_min,max(platelet) platelet_max,
avg(wbc) wbc_avg,min(wbc) wbc_min,max(wbc) wbc_max,
avg(basophils) basophils_avg,min(basophils) basophils_min,max(basophils) basophils_max,
avg(eosinophils) eosinophils_avg,min(eosinophils) eosinophils_min,max(eosinophils) eosinophils_max,
avg(lymphocytes) lymphocytes_avg,min(lymphocytes) lymphocytes_min,max(lymphocytes) lymphocytes_max,
avg(monocytes) monocytes_avg,min(monocytes) monocytes_min,max(monocytes) monocytes_max,
avg(neutrophils) neutrophils_avg,min(neutrophils) neutrophils_min,max(neutrophils) neutrophils_max,
avg(albumin) albumin_avg,min(albumin) albumin_min,max(albumin) albumin_max,
avg(aniongap) aniongap_avg,min(aniongap) aniongap_min,max(aniongap) aniongap_max,
avg(bun) bun_avg,min(bun) bun_min,max(bun) bun_max,
avg(calcium) calcium_avg,min(calcium) calcium_min,max(calcium) calcium_max,
avg(chloride) chloride_avg,min(chloride) chloride_min,max(chloride) chloride_max,
avg(sodium) sodium_avg,min(sodium) sodium_min,max(sodium) sodium_max,
avg(potassium) potassium_avg,min(potassium) potassium_min,max(potassium) potassium_max,
avg(glucose) glucose_avg,min(glucose) glucose_min,max(glucose) glucose_max,
avg(creatinine) creatinine_avg,min(creatinine) creatinine_min,max(creatinine) creatinine_max,
avg(inr) inr_avg,min(inr) inr_min,max(inr) inr_max,
avg(pt) pt_avg,min(pt) pt_min,max(pt) pt_max,
avg(ptt) ptt_avg,min(ptt) ptt_min,max(ptt) ptt_max
from sepsis left join pivoted_lab1 on  sepsis.patientunitstayid=pivoted_lab1.patientunitstayid 
where (chartoffset-hospitaladmitoffset)<=1440*3
GROUP BY sepsis.patientunitstayid
)
, charlson1 as
(
with ag AS
(
    SELECT 
        patientunitstayid
        , age
        , CASE WHEN age <= 40 THEN 0
    WHEN age <= 50 THEN 1
    WHEN age <= 60 THEN 2
    WHEN age <= 70 THEN 3
    ELSE 4 END AS age_score
    FROM icustay_detail
)
select pivoted_charlson.*-- age_score
--     + myocardial_infarct + congestive_heart_failure + peripheral_vascular_disease
--     + cerebrovascular_disease + dementia + chronic_pulmonary_disease
--     + rheumatic_disease + peptic_ulcer_disease
--     + GREATEST(mild_liver_disease, 3*severe_liver_disease)
--     + GREATEST(2*diabetes_with_cc, diabetes_without_cc)
--     + GREATEST(2*malignant_cancer, 6*metastatic_solid_tumor)
--     + 2*paraplegia + 2*renal_disease 
--     + 6*aids
--     AS charlson_comorbidity_index
from sepsis
left join pivoted_charlson on  sepsis.patientunitstayid=pivoted_charlson.patientunitstayidleft join ag on  sepsis.patientunitstayid=ag.patientunitstayid
where  pivoted_charlson.patientunitstayid is not null
)
,charlson as
(select patientunitstayid
, case when myocardial_infarct isnull then 0 else myocardial_infarct end myocardial_infarct --心梗
, case when congestive_heart_failure isnull then 0 else congestive_heart_failure end congestive_heart_failure  --慢性心衰
, case when chronic_pulmonary_disease isnull then 0 else chronic_pulmonary_disease end chronic_pulmonary_disease  --慢性肺部
, case when liver_disease isnull then 0 else liver_disease end liver_disease  -- 肝病
, case when renal_disease isnull then 0 else renal_disease end renal_disease  -- 肾病
, case when diabetes isnull then 0 else diabetes end diabetes  -- 糖尿病
, case when peripheral_vascular_disease isnull then 0 else peripheral_vascular_disease end peripheral_vascular_disease  -- 外周血管疾病
, case when cerebrovascular_disease isnull then 0 else cerebrovascular_disease end cerebrovascular_disease  -- 脑血管疾病
, case when dementia isnull then 0 else dementia end dementia  -- 痴呆
, case when rheumatic_disease isnull then 0 else rheumatic_disease end rheumatic_disease  -- 风湿免疫疾病
, case when peptic_ulcer_disease isnull then 0 else peptic_ulcer_disease end peptic_ulcer_disease  -- 消化道溃疡
, case when paraplegia isnull then 0 else paraplegia end paraplegia  -- 截瘫
, case when malignant_cancer isnull then 0 else malignant_cancer end malignant_cancer  -- 恶心非实体瘤
, case when metastatic_solid_tumor isnull then 0 else metastatic_solid_tumor end metastatic_solid_tumor  -- 恶心实体瘤
, case when aids isnull then 0 else aids end aids  --艾滋
, case when mesenteric_ischemia isnull then 0 else mesenteric_ischemia end mesenteric_ischemia --肠缺血
from charlson1
)
, pingfen as
(
with t_gcs as
(SELECT patientunitstayid,min(gcs) gcs from pivoted_gcs GROUP BY patientunitstayid)
, t_sofa as 
(SELECT patientunitstayid,max(sofatotal) sofa from sofa_results GROUP BY patientunitstayid)
, t_apsiii as 
(SELECT patientunitstayid,max(acutephysiologyscore) apsiii from eicuii.apachepatientresult GROUP BY patientunitstayid)
, t_oasis as 
(SELECT patientunitstayid,max(oasis) oasis from pivoted_oasis GROUP BY patientunitstayid)
select sepsis.patientunitstayid,
apsiii,
sofa,
oasis,
gcs
from sepsis
left join t_apsiii on sepsis.patientunitstayid = t_apsiii.patientunitstayid
left join t_sofa on sepsis.patientunitstayid = t_sofa.patientunitstayid
left join t_gcs on sepsis.patientunitstayid = t_gcs.patientunitstayid
left join t_oasis on sepsis.patientunitstayid = t_oasis.patientunitstayid
)
,med as
(
select sepsis.patientunitstayid,
max(dopamine) dopamine_max,avg(dopamine) dopamine_avg,
max(epinephrine) epinephrine_max,avg(epinephrine) epinephrine_avg,
max(norepinephrine) norepinephrine_max,avg(norepinephrine) norepinephrine_avg,
max(phenylephrine) phenylephrine_max,avg(phenylephrine) phenylephrine_avg,
max(vasopressin) vasopressin_max,avg(vasopressin) vasopressin_avg,
max(dobutamine) dobutamine_max,avg(dobutamine) dobutamine_avg,
max(milrinone) milrinone_max,avg(milrinone) milrinone_avg
from sepsis
left join pivoted_med2 on sepsis.patientunitstayid = pivoted_med2.patientunitstayid
GROUP BY sepsis.patientunitstayid
)
, crrt as
(select sepsis.patientunitstayid,
max(case when LOWER(treatmentstring) LIKE '%rrt%'
    OR LOWER(treatmentstring) LIKE '%dialysis%'
    OR LOWER(treatmentstring) LIKE '%ultrafiltration%'
    OR LOWER(treatmentstring) LIKE '%cavhd%' 
    OR LOWER(treatmentstring) LIKE '%cvvh%' 
    OR LOWER(treatmentstring) LIKE '%sled%'
    AND Lower(treatmentstring) NOT LIKE '%chronic%' then 1 else 0 end) crrt
from sepsis
left join eicuii.treatment on sepsis.patientunitstayid = treatment.patientunitstayid
GROUP BY sepsis.patientunitstayid
)
, vent as 
(
with t1 as
(select sepsis.patientunitstayid,
min(respcarestatusoffset) respcarestatusoffset_min
from sepsis
left join eicuii.respiratorycare on sepsis.patientunitstayid = respiratorycare.patientunitstayid
GROUP BY sepsis.patientunitstayid
)
select patientunitstayid, case when respcarestatusoffset_min isnull then 0 else 1 end ventilation from t1
)
, cci as 
(
select sepsis.patientunitstayid,icu_los_days,
case when (sofa_cardiovascular>=1 or SOFA_respiration>=2 or sofa_renal>=2 or sofa_coagulation>=2 or sofa_liver>=2 or sofa_cns>=2) and icu_los_days>= 14  then 1 else 0 end cci
from sepsis 
left join pivoted_sofa14 on sepsis.patientunitstayid = pivoted_sofa14.patientunitstayid
)
, acu as
(
select
patientunitstayid,
case when icu_los_days<=3 and hosp_mort=1 then 1 else 0 end acute_sepsis
from sepsis
)
,med1 as
(
select vas.* from pivoted_med2 vas 
join (select patientunitstayid,min(infusionoffset) infusionoffset_min from pivoted_med2 GROUP BY patientunitstayid) t1 on
vas.patientunitstayid = t1.patientunitstayid and infusionoffset_min=infusionoffset
ORDER BY patientunitstayid,infusionoffset
)
-- select med.patientunitstayid,avg(potassium) glucose1
-- from pivoted_lab1 c1
-- right join med on  med.patientunitstayid = c1.patientunitstayid
-- where glucose is not null and (chartoffset-infusionoffset)/1440 between 0.00001 and 1
-- GROUP BY med.patientunitstayid
, xgbj as
(
with potassiumall as
(
select sepsis.patientunitstayid,avg(potassium) potassiumall
from pivoted_lab1 c1
right join sepsis on  sepsis.patientunitstayid = c1.patientunitstayid
where potassium is not null
GROUP BY sepsis.patientunitstayid
)
, potassium3ed as
(
select med1.patientunitstayid,avg(potassium) potassium3ed
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where potassium is not null and (chartoffset-infusionoffset)/1440 between -3 and -0.00001
GROUP BY med1.patientunitstayid
)
, potassium1 as
(
select med1.patientunitstayid,avg(potassium) potassium1
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where potassium is not null and (chartoffset-infusionoffset)/1440 between 0.00001 and 1
GROUP BY med1.patientunitstayid
)
, glucoseall as
(
select sepsis.patientunitstayid,avg(potassium) glucoseall
from pivoted_lab1 c1
right join sepsis on  sepsis.patientunitstayid = c1.patientunitstayid
where glucose is not null
GROUP BY sepsis.patientunitstayid
)
, glucose3ed as
(
select med1.patientunitstayid,avg(potassium) glucose3ed
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where glucose is not null and (chartoffset-infusionoffset)/1440 between -3 and -0.00001
GROUP BY med1.patientunitstayid
)
, glucose1 as
(
select med1.patientunitstayid,avg(potassium) glucose1
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where glucose is not null and (chartoffset-infusionoffset)/1440 between 0.00001 and 1
GROUP BY med1.patientunitstayid
)
, lactateall as
(
select sepsis.patientunitstayid,avg(potassium) lactateall
from pivoted_lab1 c1
right join sepsis on  sepsis.patientunitstayid = c1.patientunitstayid
where lactate is not null
GROUP BY sepsis.patientunitstayid
)
, lactate3ed as
(
select med1.patientunitstayid,avg(potassium) lactate3ed
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where lactate is not null and (chartoffset-infusionoffset)/1440 between -3 and -0.00001
GROUP BY med1.patientunitstayid
)
, lactate1 as
(
select med1.patientunitstayid,avg(potassium) lactate1
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where lactate is not null and (chartoffset-infusionoffset)/1440 between 0.00001 and 1
GROUP BY med1.patientunitstayid
)
, ctniall as
(
select sepsis.patientunitstayid,avg(ctni) ctniall
from pivoted_lab1 c1
right join sepsis on  sepsis.patientunitstayid = c1.patientunitstayid
where ctni is not null
GROUP BY sepsis.patientunitstayid
)
, ctni3ed as
(
select med1.patientunitstayid,avg(ctni) ctni3ed
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where ctni is not null and (chartoffset-infusionoffset)/1440 between -3 and -0.00001
GROUP BY med1.patientunitstayid
)
, ctni1 as
(
select med1.patientunitstayid,avg(ctni) ctni1
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where ctni is not null and (chartoffset-infusionoffset)/1440 between 0.00001 and 1
GROUP BY med1.patientunitstayid
)
, ctntall as
(
select sepsis.patientunitstayid,avg(ctnt) ctntall
from pivoted_lab1 c1
right join sepsis on  sepsis.patientunitstayid = c1.patientunitstayid
where ctnt is not null
GROUP BY sepsis.patientunitstayid
)
, ctnt3ed as
(
select med1.patientunitstayid,avg(ctnt) ctnt3ed
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where ctnt is not null and (chartoffset-infusionoffset)/1440 between -3 and -0.00001
GROUP BY med1.patientunitstayid
)
, ctnt1 as
(
select med1.patientunitstayid,avg(ctnt) ctnt1
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where ctnt is not null and (chartoffset-infusionoffset)/1440 between 0.00001 and 1
GROUP BY med1.patientunitstayid
)
, bunall as
(
select sepsis.patientunitstayid,avg(bun) bunall
from pivoted_lab1 c1
right join sepsis on  sepsis.patientunitstayid = c1.patientunitstayid
where bun is not null
GROUP BY sepsis.patientunitstayid
)
, bun3ed as
(
select med1.patientunitstayid,avg(bun) bun3ed
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where bun is not null and (chartoffset-infusionoffset)/1440 between -3 and -0.00001
GROUP BY med1.patientunitstayid
)
, bun1 as
(
select med1.patientunitstayid,avg(bun) bun1
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where bun is not null and (chartoffset-infusionoffset)/1440 between 0.00001 and 1
GROUP BY med1.patientunitstayid
)
, bnpall as
(
select sepsis.patientunitstayid,avg(bnp) bnpall
from pivoted_lab1 c1
right join sepsis on  sepsis.patientunitstayid = c1.patientunitstayid
where bnp is not null
GROUP BY sepsis.patientunitstayid
)
, bnp3ed as
(
select med1.patientunitstayid,avg(bnp) bnp3ed
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where bnp is not null and (chartoffset-infusionoffset)/1440 between -3 and -0.00001
GROUP BY med1.patientunitstayid
)
, bnp1 as
(
select med1.patientunitstayid,avg(bnp) bnp1
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where bnp is not null and (chartoffset-infusionoffset)/1440 between 0.00001 and 1
GROUP BY med1.patientunitstayid
)
, creatinineall as
(
select sepsis.patientunitstayid,avg(creatinine) creatinineall
from pivoted_lab1 c1
right join sepsis on  sepsis.patientunitstayid = c1.patientunitstayid
where creatinine is not null
GROUP BY sepsis.patientunitstayid
)
, creatinine3ed as
(
select med1.patientunitstayid,avg(creatinine) creatinine3ed
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where creatinine is not null and (chartoffset-infusionoffset)/1440 between -3 and -0.00001
GROUP BY med1.patientunitstayid
)
, creatinine1 as
(
select med1.patientunitstayid,avg(creatinine) creatinine1
from pivoted_lab1 c1
right join med1 on  med1.patientunitstayid = c1.patientunitstayid
where creatinine is not null and (chartoffset-infusionoffset)/1440 between 0.00001 and 1
GROUP BY med1.patientunitstayid
)
select sepsis.patientunitstayid,glucoseall,glucose3ed,glucose1,potassiumall,potassium3ed,potassium1,lactateall,lactate3ed,lactate1,ctniall,ctni3ed,ctni1,ctntall,ctnt3ed,ctnt1,bnpall,bnp3ed,bnp1,bunall,bun3ed,bun1,creatinineall,creatinine3ed,creatinine1
from sepsis 
left join potassiumall on sepsis.patientunitstayid=potassiumall.patientunitstayid
left join potassium3ed on sepsis.patientunitstayid=potassium3ed.patientunitstayid
left join potassium1 on sepsis.patientunitstayid=potassium1.patientunitstayid
left join glucoseall on sepsis.patientunitstayid=glucoseall.patientunitstayid
left join glucose3ed on sepsis.patientunitstayid=glucose3ed.patientunitstayid
left join glucose1 on sepsis.patientunitstayid=glucose1.patientunitstayid
left join lactateall on sepsis.patientunitstayid=lactateall.patientunitstayid
left join lactate3ed on sepsis.patientunitstayid=lactate3ed.patientunitstayid
left join lactate1 on sepsis.patientunitstayid=lactate1.patientunitstayid
left join ctniall on sepsis.patientunitstayid=ctniall.patientunitstayid
left join ctni3ed on sepsis.patientunitstayid=ctni3ed.patientunitstayid
left join ctni1 on sepsis.patientunitstayid=ctni1.patientunitstayid
left join ctntall on sepsis.patientunitstayid=ctntall.patientunitstayid
left join ctnt3ed on sepsis.patientunitstayid=ctnt3ed.patientunitstayid
left join ctnt1 on sepsis.patientunitstayid=ctnt1.patientunitstayid
left join bunall on sepsis.patientunitstayid=bunall.patientunitstayid
left join bun3ed on sepsis.patientunitstayid=bun3ed.patientunitstayid
left join bun1 on sepsis.patientunitstayid=bun1.patientunitstayid
left join bnpall on sepsis.patientunitstayid=bnpall.patientunitstayid
left join bnp3ed on sepsis.patientunitstayid=bnp3ed.patientunitstayid
left join bnp1 on sepsis.patientunitstayid=bnp1.patientunitstayid
left join creatinineall on sepsis.patientunitstayid=creatinineall.patientunitstayid
left join creatinine3ed on sepsis.patientunitstayid=creatinine3ed.patientunitstayid
left join creatinine1 on sepsis.patientunitstayid=creatinine1.patientunitstayid
)
, arrhythmias as
(
select sepsis.patientunitstayid,
MAX(CASE WHEN 
		diagnosisstring ~*'arrhythmias'
		THEN 1 
		ELSE 0 END) AS arrhythmias
from eicuii.diagnosis dig
right join sepsis on  sepsis.patientunitstayid = dig.patientunitstayid
GROUP BY sepsis.patientunitstayid
)
, af as
(
select sepsis.patientunitstayid,
MAX(CASE WHEN 
		diagnosisstring ~*'atrial fibrillation' and LOWER(diagnosisstring) not LIKE '%measures%'
		THEN 1 
		ELSE 0 END) AS af
from eicuii.diagnosis dig
right join sepsis on  sepsis.patientunitstayid = dig.patientunitstayid
GROUP BY sepsis.patientunitstayid
)
, vf as
(
select sepsis.patientunitstayid,
MAX(CASE WHEN 
		diagnosisstring ~*'ventricular fibrillation' and LOWER(diagnosisstring) not LIKE '%cardiac arrest%'
		THEN 1 
		ELSE 0 END) AS vf
from eicuii.diagnosis dig
right join sepsis on  sepsis.patientunitstayid = dig.patientunitstayid
GROUP BY sepsis.patientunitstayid
)
, sepsis_cci_1 as
(
select
sepsis.*,
-- vit
heart_rate_avg,heart_rate_min,heart_rate_max,
sbp_avg,sbp_min,sbp_max,
dbp_avg,dbp_min,dbp_max,
mbp_avg,mbp_min,mbp_max,
resp_rate_avg,resp_rate_min,resp_rate_max,
temperature_avg,temperature_min,temperature_max,
--bg
po2_avg,po2_min,po2_max,
pco2_avg,pco2_min,pco2_max,
fio2_avg,fio2_min,fio2_max,

baseexcess_avg,baseexcess_min,baseexcess_max,
--bl
rbc_avg,rbc_min,rbc_max,
hemoglobin_avg,hemoglobin_min,hemoglobin_max,
hematocrit_avg,hematocrit_min,hematocrit_max,
rdw_avg,rdw_min,rdw_max,
mch_avg,mch_min,mch_max,
mcv_avg,mcv_min,mcv_max,
mchc_avg,mchc_min,mchc_max,
platelet_avg,platelet_min,platelet_max,
wbc_avg,wbc_min,wbc_max,
basophils_avg,basophils_min,basophils_max,
eosinophils_avg,eosinophils_min,eosinophils_max,
lymphocytes_avg,lymphocytes_min,lymphocytes_max,
monocytes_avg,monocytes_min, monocytes_max,
neutrophils_avg,neutrophils_min,neutrophils_max,
--ch
albumin_avg,albumin_min,albumin_max,
lab.aniongap_avg,lab.aniongap_min,lab.aniongap_max,
bun_avg,bun_min,bun_max,
calcium_avg,calcium_min,calcium_max,
chloride_avg,chloride_min,chloride_max,
sodium_avg,sodium_min,sodium_max,
potassium_avg,potassium_min,potassium_max,
glucose_avg,glucose_min,glucose_max,
creatinine_avg,creatinine_min,creatinine_max,
--co
inr_avg,inr_min,inr_max,
pt_avg,pt_min,pt_max,
ptt_avg,ptt_min,ptt_max
--con
,myocardial_infarct --心梗
,congestive_heart_failure  --慢性心衰
,chronic_pulmonary_disease  --慢性肺部
,liver_disease  -- 肝病
,renal_disease  -- 肾病
,diabetes  -- 糖尿病
, case when cerebrovascular_disease+dementia>=2 then 1 else  cerebrovascular_disease+dementia end nervous_system_disease --神经系统疾病
, case when malignant_cancer+metastatic_solid_tumor>=2 then 1 else malignant_cancer+metastatic_solid_tumor end malignant_tumor -- 恶性肿瘤
, peripheral_vascular_disease  -- 外周血管疾病
, cerebrovascular_disease  -- 脑血管疾病
, dementia  -- 痴呆
, rheumatic_disease  -- 风湿免疫疾病
, peptic_ulcer_disease  -- 消化道溃疡
, paraplegia  -- 截瘫
, malignant_cancer  -- 恶心非实体瘤
, metastatic_solid_tumor  -- 恶心实体瘤
, aids  --艾滋
-- , charlson_comorbidity_index, -- Calculate the Charlson Comorbidity Score using the original weights from Charlson, 1987.
--eval
,sofa,
apsiii,
--lods,
oasis,
gcs,
--vas
case when dopamine_max isnull then 0 else dopamine_max end dopamine_max,
case when dopamine_avg isnull then 0 else dopamine_avg end dopamine_avg,
case when epinephrine_max isnull then 0 else epinephrine_max end  epinephrine_max,
case when epinephrine_avg isnull then 0 else epinephrine_avg end  epinephrine_avg,
case when norepinephrine_max isnull then 0 else norepinephrine_max end norepinephrine_max,
case when norepinephrine_avg isnull then 0 else norepinephrine_avg end norepinephrine_avg,
case when phenylephrine_max isnull then 0 else phenylephrine_max end phenylephrine_max,
case when phenylephrine_avg isnull then 0 else phenylephrine_avg end phenylephrine_avg,
case when vasopressin_max isnull then 0 else vasopressin_max end vasopressin_max,
case when vasopressin_avg isnull then 0 else vasopressin_avg end vasopressin_avg,
case when dobutamine_max isnull then 0 else dobutamine_max end dobutamine_max,
case when dobutamine_avg isnull then 0 else dobutamine_avg end dobutamine_avg,
case when milrinone_max isnull then 0 else milrinone_max end milrinone_max,
case when milrinone_avg isnull then 0 else milrinone_avg end milrinone_avg,
crrt,
ventilation,

acute_sepsis,
case when cci isnull then 0 else cci end cci,
infusionoffset starttime,arrhythmias,af,vf,mesenteric_ischemia,dopamine::FLOAT,epinephrine,norepinephrine,phenylephrine,vasopressin,dobutamine,milrinone,glucoseall,glucose3ed,glucose1,potassiumall,potassium3ed,potassium1,lactateall,lactate3ed,lactate1,ctniall,ctni3ed,ctni1,ctntall,ctnt3ed,ctnt1,bnpall,bnp3ed,bnp1,bunall,bun3ed,bun1,creatinineall,creatinine3ed,creatinine1
from sepsis
left join bg on sepsis.patientunitstayid = bg.patientunitstayid
left join vit on sepsis.patientunitstayid = vit.patientunitstayid
left join lab on sepsis.patientunitstayid = lab.patientunitstayid
left join charlson on sepsis.patientunitstayid = charlson.patientunitstayid
left join pingfen on sepsis.patientunitstayid = pingfen.patientunitstayid
left join med on sepsis.patientunitstayid = med.patientunitstayid
left join crrt on sepsis.patientunitstayid = crrt.patientunitstayid
left join vent on sepsis.patientunitstayid = vent.patientunitstayid
left join cci on sepsis.patientunitstayid = cci.patientunitstayid
left join acu on sepsis.patientunitstayid = acu.patientunitstayid
left join med1 on sepsis.patientunitstayid=med1.patientunitstayid
left join xgbj on sepsis.patientunitstayid=xgbj.patientunitstayid
left join arrhythmias on sepsis.patientunitstayid=arrhythmias.patientunitstayid
left join vf on sepsis.patientunitstayid=vf.patientunitstayid
left join af on sepsis.patientunitstayid=af.patientunitstayid
)
select DISTINCT * from sepsis_cci_1 where age >= 18
;

-- select DISTINCT labname from eicuii.lab where labname ~* 'pt'
-- -- RBC Oxyhemoglobin(要除10转为g/dL) RDW MCH MCV MCHC -basos -eos -lymphs -monos -polys platelets x 1000 WBC x 1000
-- --albumin   anion gap BUN calcium  chloride sodium potassium glucose bedside glucose creatinine
-- -- PT - INR   PT   PTT
-- select * from eicuii.lab where labname = 'anion gap'
-- select avg(labresult) from eicuii.lab where labname = 'ionized calcium'
-- select * from eicuii.lab where patientunitstayid = 141168
-- 
-- select DISTINCT diagnosisstring from eicuii.diagnosis where SUBSTR(icd9code, 1, 3) in ('410','412','I21','I22','I252')

-- select DISTINCT treatmentstring from eicuii.treatment where treatmentstring~* 'dialysis'
-- 
-- 
-- DROP TABLE IF EXISTS pivoted_charlson CASCADE;
-- CREATE TABLE pivoted_charlson as
-- select
-- patientunitstayid,
-- max(case when SUBSTR(icd9code, 1, 3) in ('410','412','I21','I22','I252') then 1 else 0 end) myocardial_infarct,
-- max(case when SUBSTR(icd9code, 1, 3) = '428' then 1
-- 		when SUBSTR(icd9code, 1, 5) IN ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493') then 1
-- 		when SUBSTR(icd9code, 1, 4) BETWEEN '4254' AND '4259' then 1
-- 		else 0 end) congestive_heart_failure,
-- MAX(CASE WHEN 
-- 		SUBSTR(icd9code, 1, 3) IN ('440','441')
-- 		OR
-- 		SUBSTR(icd9code, 1, 4) IN ('0930','4373','4471','5571','5579','V434')
-- 		OR
-- 		SUBSTR(icd9code, 1, 4) BETWEEN '4431' AND '4439'
-- 		THEN 1 
-- 		ELSE 0 END) AS peripheral_vascular_disease,
-- MAX(CASE WHEN 
-- 		SUBSTR(icd9code, 1, 3) BETWEEN '430' AND '438'
-- 		OR
-- 		SUBSTR(icd9code, 1, 5) = '36234'
-- 		THEN 1 
-- 		ELSE 0 END) AS cerebrovascular_disease,
-- MAX(CASE WHEN 
-- 		LOWER(diagnosisstring) LIKE '%dementia%' 
-- 		THEN 1 
-- 		ELSE 0 END) AS dementia
-- -- Chronic pulmonary disease
-- , MAX(CASE WHEN 
-- 		SUBSTR(icd9code, 1, 3) BETWEEN '490' AND '505'
-- 		OR
-- 		SUBSTR(icd9code, 1, 4) IN ('4168','4169','5064','5081','5088')
-- 		THEN 1 
-- 		ELSE 0 END) AS chronic_pulmonary_disease
-- , MAX(CASE WHEN 
-- 		LOWER(diagnosisstring) LIKE '%immunological diseases%'
-- 		or LOWER(diagnosisstring) LIKE '%rheumatic%'
-- 		THEN 1 
-- 		ELSE 0 END) AS rheumatic_disease
-- , MAX(CASE WHEN 
-- 		LOWER(diagnosisstring) LIKE '%ulcer%' and LOWER(diagnosisstring) LIKE '%gastrointestinal%'
-- 		THEN 1 
-- 		ELSE 0 END) AS peptic_ulcer_disease
-- , MAX(CASE WHEN 
-- 		LOWER(diagnosisstring) LIKE '%hepatic disease%'
-- 		THEN 1 
-- 		ELSE 0 END) AS liver_disease
-- -- Diabetes
-- , MAX(CASE WHEN 
-- 		LOWER(diagnosisstring) LIKE '%diabetes%' and LOWER(diagnosisstring) not LIKE '%insipidus%' or LOWER(diagnosisstring) LIKE '%glycuresis%'
-- 		THEN 1 
-- 		ELSE 0 END) AS diabetes
-- 
-- -- Hemiplegia or paraplegia
-- , MAX(CASE WHEN 
-- 		LOWER(diagnosisstring) LIKE '%paraplegia%' or LOWER(diagnosisstring) LIKE '%hemiplegia%'
-- 		THEN 1 
-- 		ELSE 0 END) AS paraplegia
-- 
-- -- Renal disease
-- , MAX(CASE WHEN 
-- 		SUBSTR(icd9code, 1, 3) IN ('582','585','586','V56')
-- 		OR
-- 		SUBSTR(icd9code, 1, 4) IN ('5880','V420','V451')
-- 		OR
-- 		SUBSTR(icd9code, 1, 4) BETWEEN '5830' AND '5837'
-- 		OR
-- 		SUBSTR(icd9code, 1, 5) IN ('40301','40311','40391','40402','40403','40412','40413','40492','40493')          
-- 
-- 		THEN 1 
-- 		ELSE 0 END) AS renal_disease
-- 
-- -- Any malignancy, including lymphoma and leukemia, except malignant neoplasm of skin
-- , MAX(CASE WHEN 
-- 		SUBSTR(icd9code, 1, 3) BETWEEN '140' AND '172'
-- 		OR
-- 		SUBSTR(icd9code, 1, 4) BETWEEN '1740' AND '1958'
-- 		OR
-- 		SUBSTR(icd9code, 1, 3) BETWEEN '200' AND '208'
-- 		OR
-- 		SUBSTR(icd9code, 1, 4) = '2386'
-- 		THEN 1 
-- 		ELSE 0 END) AS malignant_cancer
-- 
-- -- Metastatic solid tumor
-- , MAX(CASE WHEN 
-- 		SUBSTR(icd9code, 1, 3) IN ('196','197','198','199')
-- 		THEN 1 
-- 		ELSE 0 END) AS metastatic_solid_tumor
-- 
-- -- AIDS/HIV
-- , MAX(CASE WHEN 
-- 		diagnosisstring ~* 'AIDS' AND diagnosisstring ~*'infectious diseases'
-- 		THEN 1 
-- 		ELSE 0 END) AS aids
-- , max(case when
-- 		diagnosisstring ~*'mesenteric ischemia'
-- 		then 1
-- 		else 0 end) as mesenteric_ischemia
-- from eicuii.diagnosis
-- GROUP BY patientunitstayid;
-- 
-- select 
-- sum(myocardial_infarct) myocardial_infarct
-- , sum(congestive_heart_failure) congestive_heart_failure
-- , sum(peripheral_vascular_disease) peripheral_vascular_disease
-- , sum(cerebrovascular_disease) cerebrovascular_disease
-- , sum(dementia) dementia
-- , sum(chronic_pulmonary_disease) chronic_pulmonary_disease
-- , sum(rheumatic_disease) rheumatic_disease
-- , sum(peptic_ulcer_disease) peptic_ulcer_disease
-- , sum(liver_disease) liver_disease
-- , sum(diabetes) diabetes
-- , sum(paraplegia) paraplegia
-- , sum(renal_disease) renal_disease
-- , sum(malignant_cancer) malignant_cancer
-- , sum(metastatic_solid_tumor ) metastatic_solid_tumor
-- , sum(aids) aids
-- from pivoted_charlson
-- 
-- select DISTINCT diagnosisstring from eicuii.diagnosis where SUBSTR(icd9code, 1, 3) IN ('042','043','044')
-- select DISTINCT diagnosisstring from eicuii.diagnosis where diagnosisstring ~*'ischemia'
-- select DISTINCT diagnosisstring from eicuii.diagnosis where diagnosisstring ~*'atrial fibrillation' and LOWER(diagnosisstring) not LIKE '%measures%'
-- select DISTINCT diagnosisstring from eicuii.diagnosis where diagnosisstring ~*'ventricular fibrillation' and LOWER(diagnosisstring) not LIKE '%cardiac arrest%'
-- 
-- select DISTINCT diagnosisstring from eicuii.diagnosis where LOWER(diagnosisstring) LIKE '%AIDS%' and LOWER(diagnosisstring) LIKE '%gastrointestinal%'
-- select * from eicuii.diagnosis



