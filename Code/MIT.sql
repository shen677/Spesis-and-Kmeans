DROP TABLE IF EXISTS pivoted_bg CASCADE;
CREATE TABLE pivoted_bg as
-- get blood gas measures
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
        'paO2'
      , 'paCO2'
      , 'pH'
      , 'FiO2'
      , 'anion gap'
      , 'Base Deficit'
      , 'Base Excess'
      , 'PEEP'
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
  WHERE
     (lab.labname = 'paO2' and lab.labresult >= 15 and lab.labresult <= 720)
  OR (lab.labname = 'paCO2' and lab.labresult >= 5 and lab.labresult <= 250)
  OR (lab.labname = 'pH' and lab.labresult >= 6.5 and lab.labresult <= 8.5)
  OR (lab.labname = 'FiO2' and lab.labresult >= 0.2 and lab.labresult <= 1.0)
  -- we will fix fio2 units later
  OR (lab.labname = 'FiO2' and lab.labresult >= 20 and lab.labresult <= 100)
  OR (lab.labname = 'anion gap' and lab.labresult >= 0 and lab.labresult <= 300)
  OR (lab.labname = 'Base Deficit' and lab.labresult >= -100 and lab.labresult <= 100)
  OR (lab.labname = 'Base Excess' and lab.labresult >= -100 and lab.labresult <= 100)
  OR (lab.labname = 'PEEP' and lab.labresult >= 0 and lab.labresult <= 60)
)
select
    patientunitstayid
  , labresultoffset as chartoffset
  -- the aggregate (max()) only ever applies to 1 value due to the where clause
  , MAX(case
        when labname != 'FiO2' then null
        when labresult >= 20 then labresult/100.0
      else labresult end) as fio2
  , MAX(case when labname = 'paO2' then labresult else null end) as pao2
  , MAX(case when labname = 'paCO2' then labresult else null end) as paco2
  , MAX(case when labname = 'pH' then labresult else null end) as pH
  , MAX(case when labname = 'anion gap' then labresult else null end) as aniongap
  , MAX(case when labname = 'Base Deficit' then labresult else null end) as basedeficit
  , MAX(case when labname = 'Base Excess' then labresult else null end) as baseexcess
  , MAX(case when labname = 'PEEP' then labresult else null end) as peep
from vw1
where rn = 1
group by patientunitstayid, labresultoffset
order by patientunitstayid, labresultoffset;


DROP TABLE IF EXISTS pivoted_gcs CASCADE;
CREATE TABLE pivoted_gcs as
with nc as
(
select
  patientunitstayid
  , nursingchartoffset as chartoffset
  , min(case
      when nursingchartcelltypevallabel = 'Glasgow coma score'
       and nursingchartcelltypevalname = 'GCS Total'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      when nursingchartcelltypevallabel = 'Score (Glasgow Coma Scale)'
       and nursingchartcelltypevalname = 'Value'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end)
    as gcs
  , min(case
      when nursingchartcelltypevallabel = 'Glasgow coma score'
       and nursingchartcelltypevalname = 'Motor'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end)
    as gcsmotor
  , min(case
      when nursingchartcelltypevallabel = 'Glasgow coma score'
       and nursingchartcelltypevalname = 'Verbal'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end)
    as gcsverbal
  , min(case
      when nursingchartcelltypevallabel = 'Glasgow coma score'
       and nursingchartcelltypevalname = 'Eyes'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end)
    as gcseyes
  from eicuii.nursecharting
  -- speed up by only looking at a subset of charted data
  where nursingchartcelltypecat in
  (
    'Scores', 'Other Vital Signs and Infusions'
  )
  group by patientunitstayid, nursingchartoffset
)
-- apply some preprocessing to fields
, ncproc AS
(
  select
    patientunitstayid
  , chartoffset
  , case when gcs > 2 and gcs < 16 then gcs else null end as gcs
  , gcsmotor, gcsverbal, gcseyes
  from nc
)
select
  patientunitstayid
  , chartoffset
  , gcs
  , gcsmotor, gcsverbal, gcseyes
FROM ncproc
WHERE gcs IS NOT NULL
OR gcsmotor IS NOT NULL
OR gcsverbal IS NOT NULL
OR gcseyes IS NOT NULL
ORDER BY patientunitstayid;



-- Extract a subset of infusions
-- NOTE: I couldn't find warfarin/coumadin.

DROP TABLE IF EXISTS pivoted_infusion CASCADE;
CREATE TABLE pivoted_infusion as
with vw0 as
(
  select
    patientunitstayid
    , infusionoffset
    -- TODO: need dopamine rate
    , max(case when drugname in
              (
                   'Dopamine'
                 , 'Dopamine ()'
                 , 'DOPamine MAX 800 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)'
                 , 'Dopamine (mcg/hr)'
                 , 'Dopamine (mcg/kg/hr)'
                 , 'dopamine (mcg/kg/min)'
                 , 'Dopamine (mcg/kg/min)'
                 , 'Dopamine (mcg/min)'
                 , 'Dopamine (mg/hr)'
                 , 'Dopamine (ml/hr)'
                 , 'Dopamine (nanograms/kg/min)'
                 , 'DOPamine STD 15 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)'
                 , 'DOPamine STD 400 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)'
                 , 'DOPamine STD 400 mg Dextrose 5% 500 ml  Premix (mcg/kg/min)'
                 , 'Dopamine (Unknown)'
              )
              -- note: no rows found for inotropin
                then 1
              else null end
            ) as dopamine

    -- this like statement is pretty reliable - no false positives when I checked
    -- also catches the brand name dobutrex
    , max(case when lower(drugname) like '%dobu%' then 1 else null end) as dobutamine
    , max(case
              when drugname in
              (
                 'Norepinephrine'
               , 'Norepinephrine ()'
               , 'Norepinephrine MAX 32 mg Dextrose 5% 250 ml (mcg/min)'
               , 'Norepinephrine MAX 32 mg Dextrose 5% 500 ml (mcg/min)'
               , 'Norepinephrine (mcg/hr)'
               , 'Norepinephrine (mcg/kg/hr)'
               , 'Norepinephrine (mcg/kg/min)'
               , 'Norepinephrine (mcg/min)'
               , 'Norepinephrine (mg/hr)'
               , 'Norepinephrine (mg/kg/min)'
               , 'Norepinephrine (mg/min)'
               , 'Norepinephrine (ml/hr)'
               , 'Norepinephrine STD 32 mg Dextrose 5% 282 ml (mcg/min)'
               , 'Norepinephrine STD 32 mg Dextrose 5% 500 ml (mcg/min)'
               , 'Norepinephrine STD 4 mg Dextrose 5% 250 ml (mcg/min)'
               , 'Norepinephrine STD 4 mg Dextrose 5% 500 ml (mcg/min)'
               , 'Norepinephrine STD 8 mg Dextrose 5% 250 ml (mcg/min)'
               , 'Norepinephrine STD 8 mg Dextrose 5% 500 ml (mcg/min)'
               , 'Norepinephrine (units/min)'
               , 'Norepinephrine (Unknown)'
               , 'norepinephrine Volume (ml)'
               , 'norepinephrine Volume (ml) (ml/hr)'
               -- levophed
              , 'Levophed (mcg/kg/min)'
              , 'levophed  (mcg/min)'
              , 'levophed (mcg/min)'
              , 'Levophed (mcg/min)'
              , 'Levophed (mg/hr)'
              , 'levophed (ml/hr)'
              , 'Levophed (ml/hr)'
              , 'NSS with LEVO (ml/hr)'
              , 'NSS w/ levo/vaso (ml/hr)'
              )
          then 1 else 0 end) as norepinephrine
    , max(case
          when drugname in
          (
             'Phenylephrine'
           , 'Phenylephrine ()'
           , 'Phenylephrine  MAX 100 mg Sodium Chloride 0.9% 250 ml (mcg/min)'
           , 'Phenylephrine (mcg/hr)'
           , 'Phenylephrine (mcg/kg/min)'
           , 'Phenylephrine (mcg/kg/min) (mcg/kg/min)'
           , 'Phenylephrine (mcg/min)'
           , 'Phenylephrine (mcg/min) (mcg/min)'
           , 'Phenylephrine (mg/hr)'
           , 'Phenylephrine (mg/kg/min)'
           , 'Phenylephrine (ml/hr)'
           , 'Phenylephrine  STD 20 mg Sodium Chloride 0.9% 250 ml (mcg/min)'
           , 'Phenylephrine  STD 20 mg Sodium Chloride 0.9% 500 ml (mcg/min)'
           , 'Volume (ml) Phenylephrine'
           , 'Volume (ml) Phenylephrine ()'
           -- neosynephrine is a synonym
           , 'neo-synephrine (mcg/min)'
           , 'neosynephrine (mcg/min)'
           , 'Neosynephrine (mcg/min)'
           , 'Neo Synephrine (mcg/min)'
           , 'Neo-Synephrine (mcg/min)'
           , 'NeoSynephrine (mcg/min)'
           , 'NEO-SYNEPHRINE (mcg/min)'
           , 'Neosynephrine (ml/hr)'
           , 'neosynsprine'
           , 'neosynsprine (mcg/kg/hr)'
          )
        then 1 else 0 end) as phenylephrine
    , max(case
            when drugname in
            (
                 'EPI (mcg/min)'
               , 'Epinepherine (mcg/min)'
               , 'Epinephrine'
               , 'Epinephrine ()'
               , 'EPINEPHrine(Adrenalin)MAX 30 mg Sodium Chloride 0.9% 250 ml (mcg/min)'
               , 'EPINEPHrine(Adrenalin)STD 4 mg Sodium Chloride 0.9% 250 ml (mcg/min)'
               , 'EPINEPHrine(Adrenalin)STD 4 mg Sodium Chloride 0.9% 500 ml (mcg/min)'
               , 'EPINEPHrine(Adrenalin)STD 7 mg Sodium Chloride 0.9% 250 ml (mcg/min)'
               , 'Epinephrine (mcg/hr)'
               , 'Epinephrine (mcg/kg/min)'
               , 'Epinephrine (mcg/min)'
               , 'Epinephrine (mg/hr)'
               , 'Epinephrine (mg/kg/min)'
               , 'Epinephrine (ml/hr)'
            ) then 1 else 0 end)
          as epinephrine
    , max(case
            when drugname in
            (
                'Vasopressin'
              , 'Vasopressin ()'
              , 'Vasopressin 20 Units Sodium Chloride 0.9% 100 ml (units/hr)'
              , 'Vasopressin 20 Units Sodium Chloride 0.9% 250 ml (units/hr)'
              , 'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/hr)'
              , 'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/kg/hr)'
              , 'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/min)'
              , 'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (Unknown)'
              , 'Vasopressin 40 Units Sodium Chloride 0.9% 200 ml (units/min)'
              , 'Vasopressin (mcg/kg/min)'
              , 'Vasopressin (mcg/min)'
              , 'Vasopressin (mg/hr)'
              , 'Vasopressin (mg/min)'
              , 'vasopressin (ml/hr)'
              , 'Vasopressin (ml/hr)'
              , 'Vasopressin (units/hr)'
              , 'Vasopressin (units/kg/min)'
              , 'vasopressin (units/min)'
              , 'Vasopressin (units/min)'
              , 'VAsopressin (units/min)'
              , 'Vasopressin (Unknown)'
            ) then 1 else 0 end)
          as vasopressin
    , max(case when drugname in
      (
           'Milrinone'
         , 'Milrinone ()'
         , 'Milrinone (mcg/kg/hr)'
         , 'Milrinone (mcg/kg/min)'
         , 'Milrinone (ml/hr)'
         , 'Milrinone (Primacor) 40 mg Dextrose 5% 200 ml (mcg/kg/min)'
         , 'Milronone (mcg/kg/min)'
         , 'primacore (mcg/kg/min)'
      ) then 1 else 0 end)
      as milrinone
    , max(case when drugname in
      (
          'Hepain (ml/hr)'
        , 'Heparin'
        , 'Heparin ()'
        , 'Heparin 25,000 Unit/D5w 250 ml (ml/hr)'
        , 'Heparin 25000 Units Dextrose 5% 500 ml  Premix (units/hr)'
        , 'Heparin 25000 Units Dextrose 5% 500 ml  Premix (units/kg/hr)'
        , 'Heparin 25000 Units Dextrose 5% 950 ml  Premix (units/kg/hr)'
        , 'HEPARIN #2 (units/hr)'
        , 'Heparin 8000u/1L NS (ml/hr)'
        , 'Heparin-EKOS (units/hr)'
        , 'Heparin/Femoral Sheath   (units/hr)'
        , 'Heparin (mcg/kg/hr)'
        , 'Heparin (mcg/kg/min)'
        , 'Heparin (ml/hr)'
        , 'heparin (units/hr)'
        , 'Heparin (units/hr)'
        , 'HEPARIN (units/hr)'
        , 'Heparin (units/kg/hr)'
        , 'Heparin (Unknown)'
        , 'Heparin via sheath (units/hr)'
        , 'Left  Heparin (units/hr)'
        , 'NSS carrier heparin (ml/hr)'
        , 'S-Heparin (units/hr)'
        , 'Volume (ml) Heparin-heparin 25,000 units in 0.45 % sodium chloride 500 mL infusion'
        , 'Volume (ml) Heparin-heparin 25,000 units in 0.45 % sodium chloride 500 mL infusion (ml/hr)'
        , 'Volume (ml) Heparin-heparin 25,000 units in dextrose 500 mL infusion'
        , 'Volume (ml) Heparin-heparin 25,000 units in dextrose 500 mL infusion (ml/hr)'
        , 'Volume (ml) Heparin-heparin infusion 2 units/mL in 0.9% sodium chloride (ARTERIAL LINE)'
        , 'Volume (ml) Heparin-heparin infusion 2 units/mL in 0.9% sodium chloride (ARTERIAL LINE) (ml/hr)'
      ) then 1 else 0 end)
      as heparin
  from eicuii.infusiondrug
  group by patientunitstayid, infusionoffset
)
select
  patientunitstayid
  , infusionoffset as chartoffset
  , dopamine::SMALLINT as dopamine
  , dobutamine::SMALLINT as dobutamine
  , norepinephrine::SMALLINT as norepinephrine
  , phenylephrine::SMALLINT as phenylephrine
  , epinephrine::SMALLINT as epinephrine
  , vasopressin::SMALLINT as vasopressin
  , milrinone::SMALLINT as milrinone
  , heparin::SMALLINT as heparin
from vw0
-- at least one of our drugs should be non-zero
where dopamine = 1
OR dobutamine = 1
OR norepinephrine = 1
OR phenylephrine = 1
OR epinephrine = 1
OR vasopressin = 1
OR milrinone = 1
OR heparin = 1
order by patientunitstayid, infusionoffset;



DROP TABLE IF EXISTS pivoted_lab CASCADE;
CREATE TABLE pivoted_lab as
-- remove duplicate labs if they exist at the same time
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
      'albumin'
    , 'total bilirubin'
    , 'BUN'
    , 'calcium'
    , 'chloride'
    , 'creatinine'
    , 'bedside glucose', 'glucose'
    , 'bicarbonate' -- HCO3
    , 'Total CO2'
    , 'Hct'
    , 'Hgb'
    , 'PT - INR'
    , 'PTT'
    , 'lactate'
    , 'platelets x 1000'
    , 'potassium'
    , 'sodium'
    , 'WBC x 1000'
    , '-bands'
    -- Liver enzymes
    , 'ALT (SGPT)'
    , 'AST (SGOT)'
    , 'alkaline phos.'
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
    OR (lab.labname = 'total bilirubin' and lab.labresult >= 0.2 and lab.labresult <= 70.175)
    OR (lab.labname = 'BUN' and lab.labresult >= 1 and lab.labresult <= 280)
    OR (lab.labname = 'calcium' and lab.labresult > 0 and lab.labresult <= 9999)
    OR (lab.labname = 'chloride' and lab.labresult > 0 and lab.labresult <= 9999)
    OR (lab.labname = 'creatinine' and lab.labresult >= 0.1 and lab.labresult <= 28.28)
    OR (lab.labname in ('bedside glucose', 'glucose') and lab.labresult >= 25 and lab.labresult <= 1500)
    OR (lab.labname = 'bicarbonate' and lab.labresult >= 0 and lab.labresult <= 9999)
    OR (lab.labname = 'Total CO2' and lab.labresult >= 0 and lab.labresult <= 9999)
    -- will convert hct unit to fraction later
    OR (lab.labname = 'Hct' and lab.labresult >= 5 and lab.labresult <= 75)
    OR (lab.labname = 'Hgb' and lab.labresult >  0 and lab.labresult <= 9999)
    OR (lab.labname = 'PT - INR' and lab.labresult >= 0.5 and lab.labresult <= 15)
    OR (lab.labname = 'lactate' and lab.labresult >= 0.1 and lab.labresult <= 30)
    OR (lab.labname = 'platelets x 1000' and lab.labresult >  0 and lab.labresult <= 9999)
    OR (lab.labname = 'potassium' and lab.labresult >= 0.05 and lab.labresult <= 12)
    OR (lab.labname = 'PTT' and lab.labresult >  0 and lab.labresult <= 500)
    OR (lab.labname = 'sodium' and lab.labresult >= 90 and lab.labresult <= 215)
    OR (lab.labname = 'WBC x 1000' and lab.labresult > 0 and lab.labresult <= 100)
    OR (lab.labname = '-bands' and lab.labresult >= 0 and lab.labresult <= 100)
    OR (lab.labname = 'ALT (SGPT)' and lab.labresult > 0)
    OR (lab.labname = 'AST (SGOT)' and lab.labresult > 0)
    OR (lab.labname = 'alkaline phos.' and lab.labresult > 0)
)
select
    patientunitstayid
  , labresultoffset as chartoffset
  , MAX(case when labname = 'albumin' then labresult else null end) as albumin
  , MAX(case when labname = 'total bilirubin' then labresult else null end) as bilirubin
  , MAX(case when labname = 'BUN' then labresult else null end) as BUN
  , MAX(case when labname = 'calcium' then labresult else null end) as calcium
  , MAX(case when labname = 'chloride' then labresult else null end) as chloride
  , MAX(case when labname = 'creatinine' then labresult else null end) as creatinine
  , MAX(case when labname in ('bedside glucose', 'glucose') then labresult else null end) as glucose
  , MAX(case when labname = 'bicarbonate' then labresult else null end) as bicarbonate
  , MAX(case when labname = 'Total CO2' then labresult else null end) as TotalCO2
  , MAX(case when labname = 'Hct' then labresult else null end) as hematocrit
  , MAX(case when labname = 'Hgb' then labresult else null end) as hemoglobin
  , MAX(case when labname = 'PT - INR' then labresult else null end) as INR
  , MAX(case when labname = 'lactate' then labresult else null end) as lactate
  , MAX(case when labname = 'platelets x 1000' then labresult else null end) as platelets
  , MAX(case when labname = 'potassium' then labresult else null end) as potassium
  , MAX(case when labname = 'PTT' then labresult else null end) as ptt
  , MAX(case when labname = 'sodium' then labresult else null end) as sodium
  , MAX(case when labname = 'WBC x 1000' then labresult else null end) as wbc
  , MAX(case when labname = '-bands' then labresult else null end) as bands
  , MAX(case when labname = 'ALT (SGPT)' then labresult else null end) as alt
  , MAX(case when labname = 'AST (SGOT)' then labresult else null end) as ast
  , MAX(case when labname = 'alkaline phos.' then labresult else null end) as alp
from vw1
where rn = 1
group by patientunitstayid, labresultoffset
order by patientunitstayid, labresultoffset;



DROP TABLE IF EXISTS pivoted_med CASCADE;
CREATE TABLE pivoted_med as
-- remove duplicate labs if they exist at the same time
with vw0 as
(
  select
    patientunitstayid
    -- due to issue in ETL, times of 0 should likely be null
    , case when drugorderoffset = 0 then null else drugorderoffset end as drugorderoffset
    , case when drugstartoffset = 0 then null else drugstartoffset end as drugstartoffset
    , case when drugstopoffset = 0 then null else drugstopoffset end as drugstopoffset

    -- assign our own identifier based off HICL codes
    -- the following codes have multiple drugs: 35779, 1874, 189
    , case
        when drughiclseqno in (37410, 36346, 2051) then 'norepinephrine'
        when drughiclseqno in (37407, 39089, 36437, 34361, 2050) then 'epinephrine'
        when drughiclseqno in (8777, 40) then 'dobutamine'
        when drughiclseqno in (2060, 2059) then 'dopamine'
        when drughiclseqno in (37028, 35517, 35587, 2087) then 'phenylephrine'
        when drughiclseqno in (38884, 38883, 2839) then 'vasopressin'
        when drughiclseqno in (9744) then 'milrinone'
        when drughiclseqno in (39654, 9545, 2807, 33442, 8643, 33314, 2808, 2810) then 'heparin'
        when drughiclseqno in (2812, 24859) then 'warfarin'
        -- now do missing HICL
        when drughiclseqno is null
          and lower(drugname) like '%heparin%' then 'heparin'
        when drughiclseqno is null
          and (lower(drugname) like '%warfarin%' OR lower(drugname) like '%coumadin%') then 'warfarin'

        when drughiclseqno is null
          and lower(drugname) like '%dobutamine%' then 'dobutamine'
        when drughiclseqno is null
          and lower(drugname) like '%dobutrex%' then 'dobutamine'
        when drughiclseqno is null
          and lower(drugname) like '%norepinephrine%' then 'norepinephrine'
        when drughiclseqno is null
          and lower(drugname) like '%levophed%' then 'norepinephrine'
        when drughiclseqno is null
          and lower(drugname) like 'epinephrine%' then 'epinephrine'
        when drughiclseqno is null
          and lower(drugname) like '%phenylephrine%' then 'phenylephrine'
        when drughiclseqno is null
          and lower(drugname) like '%neosynephrine%' then 'neosynephrine'
        when drughiclseqno is null
          and lower(drugname) like '%vasopressin%' then 'vasopressin'
        when drughiclseqno is null
          and lower(drugname) like '%milrinone%' then 'milrinone'
      else null end
        as drugname_structured

    -- raw identifiers
    , drugname, drughiclseqno, gtc

    -- delivery info
    , dosage, routeadmin, prn
    -- , loadingdose
  from eicuii.medication m
  -- only non-zero dosages
  where dosage is not null
  -- not cancelled
  and drugordercancelled = 'No'
)
select
    patientunitstayid
  , drugorderoffset
  , drugstartoffset as chartoffset
  , drugstopoffset
  , max(case when drugname_structured = 'norepinephrine' then 1 else 0 end)::SMALLINT as norepinephrine
  , max(case when drugname_structured = 'epinephrine' then 1 else 0 end)::SMALLINT as epinephrine
  , max(case when drugname_structured = 'dopamine' then 1 else 0 end)::SMALLINT as dopamine
  , max(case when drugname_structured = 'dobutamine' then 1 else 0 end)::SMALLINT as dobutamine
  , max(case when drugname_structured = 'phenylephrine' then 1 else 0 end)::SMALLINT as phenylephrine
  , max(case when drugname_structured = 'vasopressin' then 1 else 0 end)::SMALLINT as vasopressin
  , max(case when drugname_structured = 'milrinone' then 1 else 0 end)::SMALLINT as milrinone
  , max(case when drugname_structured = 'heparin' then 1 else 0 end)::SMALLINT as heparin
  , max(case when drugname_structured = 'warfarin' then 1 else 0 end)::SMALLINT as warfarin
from vw0
WHERE
  -- have to have a start time
  drugstartoffset is not null
GROUP BY
  patientunitstayid, drugorderoffset, drugstartoffset, drugstopoffset
ORDER BY
  patientunitstayid, drugstartoffset, drugstopoffset, drugorderoffset;
	
	
	
DROP TABLE IF EXISTS pivoted_o2 CASCADE;
CREATE TABLE pivoted_o2 as
-- create columns with only numeric data
with nc as
(
select
    patientunitstayid
  , nursingchartoffset
  , nursingchartentryoffset
  , case
        WHEN nursingchartcelltypevallabel = 'O2 L/%'
        AND  nursingchartcelltypevalname = 'O2 L/%'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as o2_flow
  , case
        WHEN nursingchartcelltypevallabel = 'O2 Admin Device'
        AND  nursingchartcelltypevalname = 'O2 Admin Device'
          then nursingchartvalue
      else null end
    as o2_device
  , case
        WHEN nursingchartcelltypevallabel = 'End Tidal CO2'
        AND  nursingchartcelltypevalname = 'End Tidal CO2'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as etco2
  from eicuii.nursecharting
  -- speed up by only looking at a subset of charted data
  where nursingchartcelltypecat = 'Vital Signs'
)
select
  patientunitstayid
, nursingchartoffset as chartoffset
, nursingchartentryoffset as entryoffset
, AVG(CASE WHEN o2_flow >= 0 AND o2_flow <= 100 THEN o2_flow ELSE NULL END) AS o2_flow
, MAX(o2_device) AS o2_device
, AVG(CASE WHEN etco2 >= 0 AND etco2 <= 1000 THEN etco2 ELSE NULL END) AS etco2
from nc
WHERE o2_flow    IS NOT NULL
   OR o2_device  IS NOT NULL
   OR etco2      IS NOT NULL
group by patientunitstayid, nursingchartoffset, nursingchartentryoffset
order by patientunitstayid, nursingchartoffset, nursingchartentryoffset;


-- This script duplicates the nurse charting table, making the following changes:
--  "major" vital signs -> pivoted_vital
--  "minor" vital signs -> pivoted_vital_other
DROP TABLE IF EXISTS pivoted_vital CASCADE;
CREATE TABLE pivoted_vital as
-- create columns with only numeric data
with nc as
(
select
    patientunitstayid
  , nursingchartoffset
  , nursingchartentryoffset
  , case
      when nursingchartcelltypevallabel = 'Heart Rate'
       and nursingchartcelltypevalname = 'Heart Rate'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as heartrate
  , case
      when nursingchartcelltypevallabel = 'Respiratory Rate'
       and nursingchartcelltypevalname = 'Respiratory Rate'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as RespiratoryRate
  , case
      when nursingchartcelltypevallabel = 'O2 Saturation'
       and nursingchartcelltypevalname = 'O2 Saturation'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as o2saturation
  , case
      when nursingchartcelltypevallabel = 'Non-Invasive BP'
       and nursingchartcelltypevalname = 'Non-Invasive BP Systolic'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as nibp_systolic
  , case
      when nursingchartcelltypevallabel = 'Non-Invasive BP'
       and nursingchartcelltypevalname = 'Non-Invasive BP Diastolic'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as nibp_diastolic
  , case
      when nursingchartcelltypevallabel = 'Non-Invasive BP'
       and nursingchartcelltypevalname = 'Non-Invasive BP Mean'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as nibp_mean
  , case
      when nursingchartcelltypevallabel = 'Temperature'
       and nursingchartcelltypevalname = 'Temperature (C)'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as temperature
  , case
      when nursingchartcelltypevallabel = 'Temperature'
       and nursingchartcelltypevalname = 'Temperature Location'
          then nursingchartvalue
      else null end
    as TemperatureLocation
  , case
      when nursingchartcelltypevallabel = 'Invasive BP'
       and nursingchartcelltypevalname = 'Invasive BP Systolic'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as ibp_systolic
  , case
      when nursingchartcelltypevallabel = 'Invasive BP'
       and nursingchartcelltypevalname = 'Invasive BP Diastolic'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as ibp_diastolic
  , case
      when nursingchartcelltypevallabel = 'Invasive BP'
       and nursingchartcelltypevalname = 'Invasive BP Mean'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      -- other map fields
      when nursingchartcelltypevallabel = 'MAP (mmHg)'
       and nursingchartcelltypevalname = 'Value'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      when nursingchartcelltypevallabel = 'Arterial Line MAP (mmHg)'
       and nursingchartcelltypevalname = 'Value'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as ibp_mean
  from eicuii.nursecharting
  -- speed up by only looking at a subset of charted data
  where nursingchartcelltypecat in
  (
    'Vital Signs','Scores','Other Vital Signs and Infusions'
  )
)
select
  patientunitstayid
, nursingchartoffset as chartoffset
, nursingchartentryoffset as entryoffset
, avg(case when heartrate >= 25 and heartrate <= 225 then heartrate else null end) as heartrate
, avg(case when RespiratoryRate >= 0 and RespiratoryRate <= 60 then RespiratoryRate else null end) as RespiratoryRate
, avg(case when o2saturation >= 0 and o2saturation <= 100 then o2saturation else null end) as spo2
, avg(case when nibp_systolic >= 25 and nibp_systolic <= 250 then nibp_systolic else null end) as nibp_systolic
, avg(case when nibp_diastolic >= 1 and nibp_diastolic <= 200 then nibp_diastolic else null end) as nibp_diastolic
, avg(case when nibp_mean >= 1 and nibp_mean <= 250 then nibp_mean else null end) as nibp_mean
, avg(case when temperature >= 25 and temperature <= 46 then temperature else null end) as temperature
, max(temperaturelocation) as temperaturelocation
, avg(case when ibp_systolic >= 1 and ibp_systolic <= 300 then ibp_systolic else null end) as ibp_systolic
, avg(case when ibp_diastolic >= 1 and ibp_diastolic <= 200 then ibp_diastolic else null end) as ibp_diastolic
, avg(case when ibp_mean >= 1 and ibp_mean <= 250 then ibp_mean else null end) as ibp_mean
from nc
WHERE heartrate IS NOT NULL
OR RespiratoryRate IS NOT NULL
OR o2saturation IS NOT NULL
OR nibp_systolic IS NOT NULL
OR nibp_diastolic IS NOT NULL
OR nibp_mean IS NOT NULL
OR temperature IS NOT NULL
OR temperaturelocation IS NOT NULL
OR ibp_systolic IS NOT NULL
OR ibp_diastolic IS NOT NULL
OR ibp_mean IS NOT NULL
group by patientunitstayid, nursingchartoffset, nursingchartentryoffset
order by patientunitstayid, nursingchartoffset, nursingchartentryoffset;



DROP TABLE IF EXISTS pivoted_uo CASCADE;
CREATE TABLE pivoted_uo AS
with uo as
(
select
  patientunitstayid
  , intakeoutputoffset
  , outputtotal
  , cellvaluenumeric
  , case
    when cellpath not like 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|%' then 0
    when cellpath in
    (
      'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine' -- most data is here
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|3 way foley'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|3 Way Foley'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Actual Urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Adjusted total UO NOC end shift'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|BRP (urine)'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|BRP (Urine)'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|condome cath urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|diaper urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|inc of urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incontient urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incontient urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incontient Urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incontinence of urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incontinence-urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incontinence/ voids urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incontinent of urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|INCONTINENT OF URINE'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incontinent UOP'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incontinent urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incontinent (urine)'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incontinent Urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incontinent urine counts'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incont of urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incont. of urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incont. of urine count'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incont. of urine count'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incont urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incont. urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incont. urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incont. Urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|inc urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|inc. urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Inc. urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Inc Urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|indwelling foley'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Indwelling Foley'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Catheter-Foley'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Catheterization Urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath UOP'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|straight cath urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath Urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|strait cath Urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Suprapubic Urine Output'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|true urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|True Urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|True Urine out'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|unmeasured urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Unmeasured Urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|unmeasured urine output'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urethal Catheter'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urethral Catheter'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|urinary output 7AM - 7 PM'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|urinary output 7AM-7PM'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|URINE'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|URINE'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|URINE CATHETER'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Intermittent/Straight Cath (mL)'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|straightcath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|straight cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight  cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight  Cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath''d'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|straight cath daily'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|straight cathed'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cathed'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Catheter-Foley'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight catheterization'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Catheterization Urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Catheter Output'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Catheter-Straight Catheter'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|straight cath ml''s'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight cath ml''s'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath Q6hrs'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight caths'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath UOP'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|straight cath urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath Urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-straight cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine-straight cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Straight Cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Condom Catheter'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|condom catheter'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|condome cath urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|condom cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Condom Cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|CONDOM CATHETER OUTPUT'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine via condom catheter'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine-foley'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine- foley'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine- Foley'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine foley catheter'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine, L neph:'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine (measured)'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|urine output'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-external catheter'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-foley'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-Foley'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-Foley'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-FOLEY'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-Foley cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-FOLEY CATH'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-foley catheter'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-Foley Catheter'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-FOLEY CATHETER'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-Foley Output'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-Fpley'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-Ileoconduit'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-left nephrostomy'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-Left Nephrostomy'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-Left Nephrostomy Tube'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-LEFT PCN TUBE'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-L Nephrostomy'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-L Nephrostomy Tube'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-Nephrostomy'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-right nephrostomy'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-RIGHT Nephrouretero Stent Urine Output'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-R nephrostomy'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-R Nephrostomy'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-R. Nephrostomy'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-R Nephrostomy Tube'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-Rt Nephrectomy'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-stent'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-straight cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-suprapubic'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-Texas Cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-Urine'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output-Urine Output'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine, R neph:'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine-straight cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Straight Cath'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|urine (void)'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine- void'
    , 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine, void:'
    ) then 1
    when cellpath ilike 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|foley%'
    AND lower(cellpath) not like '%pacu%'
    AND lower(cellpath) not like '%or%'
    AND lower(cellpath) not like '%ir%'
      then 1
    when cellpath like 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Output%Urinary Catheter%' then 1
    when cellpath like 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Output%Urethral Catheter%' then 1
    when cellpath like 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output (mL)%' then 1
    when cellpath like 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Output%External Urethral%' then 1
    when cellpath like 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urinary Catheter Output%' then 1
  else 0 end as cellpath_is_uo
from eicuii.intakeoutput
)
select
  patientunitstayid
  , intakeoutputoffset as chartoffset
  , max(outputtotal) as outputtotal
  , sum(cellvaluenumeric) as urineoutput
from uo
where uo.cellpath_is_uo = 1
and cellvaluenumeric is not null
group by patientunitstayid, intakeoutputoffset
order by patientunitstayid, intakeoutputoffset;



-- ------------------------------------------------------------------
-- Title: Oxford Acute Severity of Illness Score (OASIS)
-- This query extracts the Oxford acute severity of illness score.
-- This score is a measure of severity of illness for patients in the ICU.
-- The score is calculated on the first day of each ICU patients' stay.
-- OASIS score was originally created for MIMIC
-- This script creates a pivoted table containing the OASIS score in eICU 
-- ------------------------------------------------------------------

-- Authors:
-- Tristan Struja, MD, MSc, MPH (ORCID 0000-0003-0199-0184) and João Matos, MS (ORICD 0000-0002-0312-1647)

-- Reference for OASIS:
--    Johnson, Alistair EW, Andrew A. Kramer, and Gari D. Clifford.
--    "A new severity of illness scale using a subset of acute physiology and chronic health evaluation data elements shows comparable predictive accuracy*."
--    Critical care medicine 41, no. 7 (2013): 1711-1718.
-- https://alistairewj.github.io/project/oasis/

-- Variables used in OASIS (first 24h only):
--  Heart rate, MAP, Temperature, Respiratory rate
--  (sourced FROM `physionet-data.eicu_crd_derived.pivoted_vital`)
--  GCS
--  (sourced FROM `physionet-data.eicu_crd_derived.pivoted_vital` and `physionet-data.eicu_crd_derived.physicalexam`)
--  Urine output 
--  (sourced  FROM `physionet-data.eicu_crd_derived.pivoted_uo`)
--  Pre-ICU in-hospital length of stay 
--  (sourced FROM `physionet-data.eicu_crd.patient`)
--  Age 
--  (sourced FROM `physionet-data.eicu_crd.patient`)
--  Elective surgery 
--  (sourced FROM `physionet-data.eicu_crd.patient` and `physionet-data.eicu_crd.apachepredvar`)
--  Ventilation status 
--  (sourced FROM `physionet-data.eicu_crd_derived.ventilation_events`, `physionet-data.eicu_crd.apacheapsvar`, 
--   `physionet-data.eicu_crd.apachepredvar`, and `physionet-data.eicu_crd.respiratorycare`)

-- Regarding missing values:
-- Elective stay: If there is no information on surgery in an elective stay, we assumed all cases to be -> "no elective surgery"
-- There are a lot of missing values, especially for urine output. Hence, we have created 2 OASIS summary scores:
-- 1) No imputation, values as is with missings. 2) Imputation in case of NULL values, with 0's (common approach for severity of illness scores)

-- Note:
--  The score is calculated for *all* ICU patients, with the assumption that the user will subselect appropriate patientunitstayid.

DROP TABLE IF EXISTS pivoted_oasis CASCADE;
CREATE TABLE pivoted_oasis as
WITH 

-- Pre-ICU stay LOS -> directly convert from minutes to hours
pre_icu_los_data AS (
SELECT patientunitstayid AS pid_LOS
  ,CASE
      WHEN hospitaladmitoffset > (-0.17*60) THEN 5
      WHEN hospitaladmitoffset BETWEEN (-4.94*60) AND (-0.17*60) THEN 3
      WHEN hospitaladmitoffset BETWEEN (-24*60) AND (-4.94*60) THEN 0
      WHEN hospitaladmitoffset BETWEEN (-311.80*60) AND (-24.0*60) THEN 2
      WHEN hospitaladmitoffset < (-311.80*60) THEN 1
      ELSE NULL
      END AS pre_icu_los_oasis
    FROM eicuii.patient
)
-- Heart rate 
, heartrate_oasis AS (
SELECT patientunitstayid AS pid_HR
  , CASE
    WHEN MIN(heartrate) < 33 THEN 4
    WHEN MAX(heartrate) BETWEEN 33 AND 88 THEN 0
    WHEN MAX(heartrate) BETWEEN 89 AND 106 THEN 1
    WHEN MAX(heartrate) BETWEEN 107 AND 125 THEN 3
    WHEN MAX(heartrate) > 125 THEN 6
    ELSE NULL
    END AS heartrate_oasis
  FROM "public".pivoted_vital
  WHERE (chartoffset > 0 AND chartoffset <= 1440) -- consider only first 24h
    AND heartrate IS NOT NULL
  GROUP BY pid_HR
)

-- Mean arterial pressure
, map_oasis AS (
  SELECT patientunitstayid AS pid_MAP
  , CASE
    WHEN MIN(ibp_mean) < 20.65 THEN 4
    WHEN MIN(ibp_mean) BETWEEN 20.65 AND 50.99 THEN 3
    WHEN MIN(ibp_mean) BETWEEN 51 AND 61.32 THEN 2
    WHEN MIN(ibp_mean) BETWEEN 61.33 AND 143.44 THEN 0
    WHEN MAX(ibp_mean) >143.44 THEN 3
    
    WHEN MIN(nibp_mean) < 20.65 THEN 4
    WHEN MIN(nibp_mean) BETWEEN 20.65 AND 50.99 THEN 3
    WHEN MIN(nibp_mean) BETWEEN 51 AND 61.32 THEN 2
    WHEN MIN(nibp_mean) BETWEEN 61.33 AND 143.44 THEN 0
    WHEN MAX(nibp_mean) >143.44 THEN 3
    ELSE NULL
    END AS map_oasis
  FROM "public".pivoted_vital
  WHERE (chartoffset > 0 AND chartoffset <= 1440) -- consider only first 24h
  GROUP BY pid_MAP
)

-- Respiratory rate
, respiratoryrate_oasis AS (
SELECT patientunitstayid AS pid_RR
  , CASE
    WHEN MIN(respiratoryrate) < 6 THEN 10
    WHEN MIN(respiratoryrate) BETWEEN 6 AND 12 THEN 1
    WHEN MIN(respiratoryrate) BETWEEN 13 AND 22 THEN 0
    WHEN MAX(respiratoryrate) BETWEEN 23 AND 30 THEN 1
    WHEN MAX(respiratoryrate) BETWEEN 31 AND 44 THEN 6
    WHEN MAX(respiratoryrate) > 44 THEN 9
    ELSE NULL
    END AS respiratoryrate_oasis

  FROM "public".pivoted_vital
  WHERE (chartoffset > 0 AND chartoffset <= 1440) -- consider only first 24h
    AND respiratoryrate IS NOT NULL
  GROUP BY pid_RR
)

-- Temperature 
, temperature_oasis AS (
  SELECT patientunitstayid AS pid_temp
  , CASE
    WHEN MIN(temperature) < 33.22 THEN 3
    WHEN MIN(temperature) BETWEEN 33.22 AND 35.93 THEN 4
    WHEN MAX(temperature) BETWEEN 33.22 AND 35.93 THEN 4
    WHEN MIN(temperature) BETWEEN 35.94 AND 36.39 THEN 2
    WHEN MAX(temperature) BETWEEN 36.40 AND 36.88 THEN 0
    WHEN MAX(temperature) BETWEEN 36.89 AND 39.88 THEN 2
    WHEN MAX(temperature) >39.88 THEN 6
    ELSE NULL
    END AS temperature_oasis
  FROM "public".pivoted_vital
  WHERE (chartoffset > 0 AND chartoffset <= 1440) -- consider only first 24h
    AND temperature IS NOT NULL
  GROUP BY pid_temp
)

-- Age 
-- Change age from string to integer
, age_numeric AS (
with t1 as(
  SELECT patientunitstayid 
  , CASE
    WHEN age = '> 89' THEN '91'
    when age = '' then null
		else age
    END AS age_num
    FROM eicuii.patient)
select patientunitstayid,CAST(age_num AS INTEGER) age_num from t1
)

-- Get the information itself in a second step
, age_oasis AS (
    SELECT patientunitstayid AS pid_age
    , CASE
    WHEN MAX(age_num) < 24 THEN 0
    WHEN MAX(age_num) BETWEEN 24 AND 53 THEN 3
    WHEN MAX(age_num) BETWEEN 54 AND 77 THEN 6
    WHEN MAX(age_num) BETWEEN 78 AND 89 THEN 9
    WHEN MAX(age_num) > 89 THEN 7
    ELSE NULL
    END AS age_oasis
    FROM age_numeric
    GROUP BY pid_age
)

-- GCS, Glasgow Coma Scale
-- Merge information from two tables into one
, merged_gcs AS (
    
  SELECT pat_gcs.patientunitstayid, physicalexam.gcs1, pivoted_gcs.gcs2
  FROM eicuii.patient AS pat_gcs
   
    LEFT JOIN(
      SELECT patientunitstayid, MIN(CAST(physicalexamvalue AS NUMERIC)) AS gcs1
      FROM eicuii.physicalexam
      WHERE  (
      (physicalExamPath LIKE 'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/_' OR
       physicalExamPath LIKE 'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/__')
      AND (physicalexamoffset > 0 AND physicalexamoffset <= 1440) -- consider only first 24h
      AND physicalexamvalue IS NOT NULL)
  GROUP BY patientunitstayid
  )
  AS physicalexam
  ON physicalexam.patientunitstayid = pat_gcs.patientunitstayid

    LEFT JOIN(
      SELECT pivoted_gcs.patientunitstayid, pivoted_gcs.gcs as gcs2
      FROM pivoted_gcs AS pivoted_gcs
      WHERE (chartoffset > 0 AND chartoffset <= 1440) -- consider only first 24h
    )
  AS pivoted_gcs
  ON pivoted_gcs.patientunitstayid = pat_gcs.patientunitstayid
)

-- Only keep minimal gcs from merged_gcs table
, minimal_gcs AS (
    SELECT patientunitstayid, COALESCE(gcs1, gcs2) AS gcs_min 
    FROM merged_gcs
)

-- Call merged_gcs table in one go
, gcs_oasis AS (
    SELECT patientunitstayid AS pid_gcs
    , CASE
    WHEN gcs_min < 8 THEN 10
    WHEN gcs_min BETWEEN 8 AND 13 THEN 4
    WHEN gcs_min = 14 THEN 3
    WHEN gcs_min = 15 THEN 0
    ELSE NULL
    END AS gcs_oasis
    FROM minimal_gcs
    --WHERE (chartoffset > 0 AND chartoffset <= 1440) -- already considered in step above
)

-- Elective admission

-- Mapping
-- Assume emergency admission if patient came from
-- Emergency Department
-- Assume elective admission if patient from other place, e.g. operating room, floor, Direct Admit, Chest Pain Center, Other Hospital, Observation, etc.
, elective_surgery AS (

    -- 1: pat table as base for patientunitstayid  
    SELECT pat.patientunitstayid, electivesurgery1
      , CASE
      WHEN unitAdmitSource LIKE 'Emergency Department' THEN 0
      ELSE 1
      END AS adm_elective1
      FROM eicuii.patient AS pat

    -- 2: apachepredvar table
    LEFT JOIN (
    SELECT apache.patientunitstayid, electivesurgery AS electivesurgery1
    FROM eicuii.apachepredvar AS apache
    )
    AS apache
    ON pat.patientunitstayid = apache.patientunitstayid

)

, electivesurgery_oasis AS (
  SELECT patientunitstayid AS pid_adm
  , CASE
    WHEN electivesurgery1 = 0 THEN 6
    WHEN electivesurgery1 IS NULL THEN 6
    WHEN adm_elective1 = 0 THEN 6
    ELSE 0
    END AS electivesurgery_oasis
  FROM elective_surgery
)


-- Urine output
, merged_uo AS (
  
  -- pat table as base for patientunitstayid 
  SELECT pat.patientunitstayid, COALESCE(pivoted_uo.urineoutput, apache_urine.urine) AS uo_comb -- consider pivoted_uo first, if missing -> apacheapsvar
  FROM eicuii.patient AS pat
  
  -- Join information from pivoted_uo table
  LEFT JOIN(
  SELECT patientunitstayid AS pid_uo, SUM(urineoutput) AS urineoutput
  FROM pivoted_uo 
  WHERE (chartoffset > 0 AND chartoffset <= 1440) -- consider only first 24h
  AND urineoutput > 0 AND urineoutput IS NOT NULL -- ignore biologically implausible values <0
  GROUP BY pid_uo
  ) AS pivoted_uo
  ON pivoted_uo.pid_uo = pat.patientunitstayid

  -- Join information from apacheapsvar table
  LEFT JOIN(
  SELECT patientunitstayid AS pid_auo, urine
  FROM eicuii.apacheapsvar 
  WHERE urine > 0 AND urine IS NOT NULL -- ignore biologically implausible values <0
  ) AS apache_urine
  ON apache_urine.pid_auo = pat.patientunitstayid

)

-- Call merged_uo table for score computation
, urineoutput_oasis AS (
  SELECT merged_uo.patientunitstayid AS pid_urine, merged_uo.uo_comb
  , CASE
    WHEN uo_comb <671 THEN 10
    WHEN uo_comb BETWEEN 671 AND 1426.99 THEN 5
    WHEN uo_comb BETWEEN 1427 AND 2543.99 THEN 1
    WHEN uo_comb BETWEEN 2544 AND 6896 THEN 0
    WHEN uo_comb >6896 THEN 8
    ELSE NULL
    END AS urineoutput_oasis
  FROM merged_uo
)

-- Ventiliation -> Note: This information is stored in 5 tables
-- Create unified vent_table first
, merged_vent AS (

    -- 1: use patient table as base 
    SELECT pat.patientunitstayid, vent_2, vent_3, vent_4--, vent_1, vent_2, vent_3, vent_4
    FROM eicuii.patient AS pat

--     -- 2: ventilation_events table
--       LEFT JOIN(
--         SELECT patientunitstayid,
--         MAX( CASE WHEN event = 'mechvent start' OR event = 'mechvent end' THEN 1
--         ELSE NULL
--         END) as vent_1
--         FROM eicuii.ventilation_events AS vent_events
--         GROUP BY patientunitstayid
--   )
--   AS vent_events
--   ON vent_events.patientunitstayid = pat.patientunitstayid 

    -- 3: apacheapsvar table
    LEFT JOIN(
      SELECT patientunitstayid, intubated as vent_2
      FROM eicuii.apacheapsvar AS apacheapsvar
      WHERE (intubated = 1)
  )
  AS apacheapsvar
  ON apacheapsvar.patientunitstayid = pat.patientunitstayid 
  
    -- 4: apachepredvar table
    LEFT JOIN(
      SELECT patientunitstayid, oobintubday1 as vent_3
      FROM eicuii.apachepredvar AS apachepredvar
      WHERE (oobintubday1 = 1)
  )
  AS apachepredvar
  ON apachepredvar.patientunitstayid = pat.patientunitstayid 
    
    
    -- 5: respiratory care table 
    LEFT JOIN(
      SELECT patientunitstayid, 
      CASE
      WHEN COUNT(airwaytype) >= 1 THEN 1
      WHEN COUNT(airwaysize) >= 1 THEN 1
      WHEN COUNT(airwayposition) >= 1 THEN 1
      WHEN COUNT(cuffpressure) >= 1 THEN 1
      WHEN COUNT(setapneatv) >= 1 THEN 1
      ELSE NULL
      END AS vent_4
      FROM eicuii.respiratorycare AS resp_care
      WHERE (respCareStatusOffset > 0 AND respCareStatusOffset <= 1440)
      GROUP BY patientunitstayid
  )
  AS resp_care
  ON resp_care.patientunitstayid = pat.patientunitstayid 
)

-- Call merged vent table in one go
, vent_oasis AS (
    SELECT patientunitstayid AS pid_vent
    , CASE
    --WHEN vent_1 = 1 THEN 9
    WHEN vent_2 = 1 THEN 9
    WHEN vent_3 = 1 THEN 9
    WHEN vent_4 = 1 THEN 9
    ELSE 0
    END AS vent_oasis
    FROM merged_vent
    --WHERE (chartoffset > 0 AND chartoffset <= 1440) -- already considered in step above
)
	
, cohort_oasis AS (
  SELECT cohort.patientunitstayid, 
  pre_icu_los_data.pre_icu_los_oasis, 
  age_oasis.age_oasis, 
  gcs_oasis.gcs_oasis,
  heartrate_oasis.heartrate_oasis,
  map_oasis.map_oasis,
  respiratoryrate_oasis.respiratoryrate_oasis,
  temperature_oasis.temperature_oasis,
  urineoutput_oasis.urineoutput_oasis,
  vent_oasis.vent_oasis,
  electivesurgery_oasis.electivesurgery_oasis
  FROM eicuii.patient AS cohort

  LEFT JOIN pre_icu_los_data
  ON cohort.patientunitstayid = pre_icu_los_data.pid_LOS

  LEFT JOIN age_oasis
  ON cohort.patientunitstayid = age_oasis.pid_age 

  LEFT JOIN gcs_oasis
  ON cohort.patientunitstayid = gcs_oasis.pid_gcs 

  LEFT JOIN heartrate_oasis
  ON cohort.patientunitstayid = heartrate_oasis.pid_HR 

  LEFT JOIN map_oasis
  ON cohort.patientunitstayid = map_oasis.pid_MAP 

  LEFT JOIN respiratoryrate_oasis
  ON cohort.patientunitstayid = respiratoryrate_oasis.pid_RR

  LEFT JOIN temperature_oasis
  ON cohort.patientunitstayid = temperature_oasis.pid_temp

  LEFT JOIN urineoutput_oasis
  ON cohort.patientunitstayid = urineoutput_oasis.pid_urine

  LEFT JOIN vent_oasis
  ON cohort.patientunitstayid = vent_oasis.pid_vent

  LEFT JOIN electivesurgery_oasis
  ON cohort.patientunitstayid = electivesurgery_oasis.pid_adm

)

, score_impute AS (

SELECT cohort_oasis.*,
	case when pre_icu_los_oasis isnull then 0 else pre_icu_los_oasis end AS pre_icu_los_oasis_imp,
	case when age_oasis isnull then 0 else age_oasis end AS age_oasis_imp,
	case when gcs_oasis isnull then 0 else gcs_oasis end AS gcs_oasis_imp,
	case when heartrate_oasis isnull then 0 else heartrate_oasis end AS heartrate_oasis_imp,
	case when map_oasis isnull then 0 else map_oasis end AS map_oasis_imp,
	case when respiratoryrate_oasis isnull then 0 else respiratoryrate_oasis end AS respiratoryrate_oasis_imp,
	case when temperature_oasis isnull then 0 else temperature_oasis end AS temperature_oasis_imp,
	case when urineoutput_oasis isnull then 0 else urineoutput_oasis end AS urineoutput_oasis_imp,
	case when vent_oasis isnull then 0 else vent_oasis end AS vent_oasis_imp,
	case when electivesurgery_oasis isnull then 0 else electivesurgery_oasis end AS electivesurgery_oasis_imp
--   IFNULL(pre_icu_los_oasis, 0) AS pre_icu_los_oasis_imp,
--   IFNULL(age_oasis, 0) AS age_oasis_imp, 
--   IFNULL(gcs_oasis, 0) AS gcs_oasis_imp, 
--   IFNULL(heartrate_oasis, 0) AS heartrate_oasis_imp,
--   IFNULL(map_oasis, 0) AS map_oasis_imp,
--   IFNULL(respiratoryrate_oasis, 0) AS respiratoryrate_oasis_imp,
--   IFNULL(temperature_oasis, 0) AS temperature_oasis_imp, 
--   IFNULL(urineoutput_oasis, 0) AS urineoutput_oasis_imp, 
--   IFNULL(vent_oasis, 0) AS vent_oasis_imp, 
--   IFNULL(electivesurgery_oasis, 0) AS electivesurgery_oasis_imp

FROM cohort_oasis
)

--Compute overall score
-- oasis_null -> only cases where all components have a Non-NULL value
-- oasis_imp -> Imputation in case of NULL values, with 0's (common approach for severity of illness scores)
, score AS (
SELECT patientunitstayid, 
    MAX(pre_icu_los_oasis) AS pre_icu_los_oasis,
    MAX(age_oasis) AS age_oasis,
    MAX(gcs_oasis) AS gcs_oasis,
    MAX(heartrate_oasis) AS heartrate_oasis,
    MAX(map_oasis) AS map_oasis,
    MAX(respiratoryrate_oasis) AS respiratoryrate_oasis,
    MAX(temperature_oasis) AS temperature_oasis,
    MAX(urineoutput_oasis) AS urineoutput_oasis,
    MAX(vent_oasis) AS vent_oasis,
    MAX(electivesurgery_oasis) AS electivesurgery_oasis,
    MAX(pre_icu_los_oasis + 
        age_oasis + 
        gcs_oasis + 
        heartrate_oasis + 
        map_oasis + 
        respiratoryrate_oasis + 
        temperature_oasis + 
        urineoutput_oasis + 
        vent_oasis + 
        electivesurgery_oasis) AS oasis_null,
  
  MAX(pre_icu_los_oasis_imp) AS pre_icu_los_oasis_imp,
  MAX(age_oasis_imp) AS age_oasis_imp, 
  MAX(gcs_oasis_imp) AS gcs_oasis_imp, 
  MAX(heartrate_oasis_imp) AS heartrate_oasis_imp,
  MAX(map_oasis_imp) AS map_oasis_imp,
  MAX(respiratoryrate_oasis_imp) AS respiratoryrate_oasis_imp,
  MAX(temperature_oasis_imp) AS temperature_oasis_imp, 
  MAX(urineoutput_oasis_imp) AS urineoutput_oasis_imp, 
  MAX(vent_oasis_imp) AS vent_oasis_imp,
  MAX(electivesurgery_oasis_imp) AS electivesurgery_oasis_imp, 
  MAX(pre_icu_los_oasis_imp + 
      age_oasis_imp + 
      gcs_oasis_imp + 
      heartrate_oasis_imp + 
      map_oasis_imp + 
      respiratoryrate_oasis_imp + 
      temperature_oasis_imp + 
      urineoutput_oasis_imp + 
      vent_oasis_imp + 
      electivesurgery_oasis_imp) AS oasis_imp

FROM score_impute
GROUP BY patientunitstayid

)

-- Final statement to generate view
-- Note: single components contain NULL values, but not final OASIS score (NULL's replaced by 0, see above)
-- Code for above columns is retrained as convienience for user wanting to modify the view for other puroposes
SELECT patientunitstayid, 
pre_icu_los_oasis,
age_oasis,
gcs_oasis,
heartrate_oasis,
map_oasis,
respiratoryrate_oasis,
temperature_oasis,
urineoutput_oasis,
vent_oasis,
electivesurgery_oasis,
oasis_imp AS oasis
-- Calculate the probability of in-hospital mortality
, 1 / (1 + exp(- (-6.1746 + 0.1275*(oasis_imp) ))) AS oasis_prob

FROM score
;



DROP TABLE IF EXISTS pivoted_score CASCADE;
CREATE TABLE pivoted_score as
-- create columns with only numeric data
with nc as
(
select
    patientunitstayid
  , nursingchartoffset
  , nursingchartentryoffset
  , case
    when nursingchartcelltypecat = 'Scores'
     and nursingchartcelltypevallabel = 'Glasgow coma score'
     and nursingchartcelltypevalname = 'GCS Total'
     and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
     and nursingchartvalue not in ('-','.')
        then cast(nursingchartvalue as numeric)
    when nursingchartcelltypecat = 'Other Vital Signs and Infusions'
     and nursingchartcelltypevallabel = 'Score (Glasgow Coma Scale)'
     and nursingchartcelltypevalname = 'Value'
     and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
     and nursingchartvalue not in ('-','.')
        then cast(nursingchartvalue as numeric)
    else null end
  as gcs
  -- components of GCS
  , case
    when nursingchartcelltypecat = 'Scores'
     and nursingchartcelltypevallabel = 'Glasgow coma score'
     and nursingchartcelltypevalname = 'Motor'
     and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
     and nursingchartvalue not in ('-','.')
        then cast(nursingchartvalue as numeric)
    when nursingchartcelltypecat = 'Other Vital Signs and Infusions'
     and nursingchartcelltypevallabel = 'Best Motor Response'
        then case
          when nursingchartvalue in ('1', '1-->(M1) none', 'Flaccid') then 1
          when nursingchartvalue in ('2', '2-->(M2) extension to pain', 'Abnormal extension') then 2
          when nursingchartvalue in ('3', '3-->(M3) flexion to pain', 'Abnormal flexion') then 3
          when nursingchartvalue in ('4', '4-->(M4) withdraws from pain', 'Withdraws') then 4
          when nursingchartvalue in ('5', '5-->(M5) localizes pain', 'Localizes to noxious stimuli') then 5
          when nursingchartvalue in ('6','6-->(M6) obeys commands', 'Obeys simple commands') then 6
        else null end
    else null end
  as gcs_motor
  , case
    when nursingchartcelltypecat = 'Scores'
     and nursingchartcelltypevallabel = 'Glasgow coma score'
     and nursingchartcelltypevalname = 'Verbal'
     and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
     and nursingchartvalue not in ('-','.')
        then cast(nursingchartvalue as numeric)
    when nursingchartcelltypecat = 'Other Vital Signs and Infusions'
     and nursingchartcelltypevallabel = 'Best Verbal Response'
        then case
          -- when nursingchartvalue in ('Trached or intubated') then 0
          when nursingchartvalue in ('1', '1-->(V1) none', 'None', 'Clearly unresponsive') then 1
          when nursingchartvalue in ('2', '2-->(V2) incomprehensible speech', 'Incomprehensible sounds') then 2
          when nursingchartvalue in ('3', '3-->(V3) inappropriate words', 'Inappropriate words') then 3
          when nursingchartvalue in ('4', '4-->(V4) confused', 'Confused') then 4
          when nursingchartvalue in ('5', '5-->(V5) oriented', 'Oriented',
                                    'Orientation/ability to communicate questionable',
                                    'Clearly oriented/can indicate needs') then 5
        else null end
    else null end
  as gcs_verbal
  , case
    when nursingchartcelltypecat = 'Scores'
     and nursingchartcelltypevallabel = 'Glasgow coma score'
     and nursingchartcelltypevalname = 'Eyes'
     and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
     and nursingchartvalue not in ('-','.')
        then cast(nursingchartvalue as numeric)
    when nursingchartcelltypecat = 'Other Vital Signs and Infusions'
     and nursingchartcelltypevallabel = 'Best Eye Response'
        then case
          when nursingchartvalue in ('1', '1-->(E1) none') then 1
          when nursingchartvalue in ('2', '2-->(E2) to pain') then 2
          when nursingchartvalue in ('3', '3-->(E3) to speech') then 3
          when nursingchartvalue in ('4', '4-->(E4) spontaneous') then 4
        else null end
    else null end
  as gcs_eyes
  -- unable/other misc info
  , case
    when nursingchartcelltypecat = 'Scores'
     and nursingchartcelltypevallabel = 'Glasgow coma score'
     and nursingchartcelltypevalname = 'GCS Total'
     and nursingchartvalue = 'Unable to score due to medication'
        then 1
    else null end
  as gcs_unable
  , case
    when nursingchartcelltypecat = 'Other Vital Signs and Infusions'
     and nursingchartcelltypevallabel = 'Best Verbal Response'
     and nursingchartvalue = 'Trached or intubated'
        then 1
    else null end
  as gcs_intub
  -- fall risk
  , case
    when nursingchartcelltypecat = 'Scores'
     and nursingchartcelltypevallabel = 'Fall Risk'
     and nursingchartcelltypevalname = 'Fall Risk'
        then case
          when nursingchartvalue = 'Low' then 1
          when nursingchartvalue = 'Medium' then 2
          when nursingchartvalue = 'High' then 3
        else null end
    else null end::numeric
  as fall_risk
  -- delirium
  , case
    when nursingchartcelltypecat = 'Scores'
     and nursingchartcelltypevallabel = 'Delirium Scale/Score'
     and nursingchartcelltypevalname = 'Delirium Scale'
        then nursingchartvalue
    else null end
  as delirium_scale
  , case
    when nursingchartcelltypecat = 'Scores'
     and nursingchartcelltypevallabel = 'Delirium Scale/Score'
     and nursingchartcelltypevalname = 'Delirium Score'
        then case
          when nursingchartvalue in ('No', 'NO') then 0
          when nursingchartvalue in ('Yes', 'YES') then 1
          when nursingchartvalue = 'N/A' then NULL
        else cast(nursingchartvalue as numeric) end
    else null end
  as delirium_score
  -- sedation
  , case
    when nursingchartcelltypecat = 'Scores'
     and nursingchartcelltypevallabel = 'Sedation Scale/Score/Goal'
     and nursingchartcelltypevalname = 'Sedation Scale'
        then nursingchartvalue
    else null end
  as sedation_scale
  , case
    when nursingchartcelltypecat = 'Scores'
     and nursingchartcelltypevallabel = 'Sedation Scale/Score/Goal'
     and nursingchartcelltypevalname = 'Sedation Score'
        then cast(nursingchartvalue as numeric)
    else null end
  as sedation_score
  , case
    when nursingchartcelltypecat = 'Scores'
     and nursingchartcelltypevallabel = 'Sedation Scale/Score/Goal'
     and nursingchartcelltypevalname = 'Sedation Goal'
        then cast(nursingchartvalue as numeric)
    else null end
  as sedation_goal
  -- pain
  , case
    when nursingchartcelltypecat = 'Scores'
     and nursingchartcelltypevallabel = 'Pain Score/Goal'
     and nursingchartcelltypevalname = 'Pain Score'
        then cast(nursingchartvalue as numeric)
    else null end
  as pain_score
  , case
    when nursingchartcelltypecat = 'Scores'
     and nursingchartcelltypevallabel = 'Pain Score/Goal'
     and nursingchartcelltypevalname = 'Pain Goal'
        then cast(nursingchartvalue as numeric)
    else null end
  as pain_goal
  from eicuii.nursecharting
  -- speed up by only looking at a subset of charted data
  where nursingchartcelltypecat IN
  (
        'Scores'
      , 'Other Vital Signs and Infusions'
  )
)
select
  patientunitstayid
, nursingchartoffset as chartoffset
, nursingchartentryoffset as entryoffset
, AVG(gcs) as gcs
, AVG(gcs_motor) as gcs_motor
, AVG(gcs_verbal) as gcs_verbal
, AVG(gcs_eyes) as gcs_eyes
, MAX(gcs_unable) as gcs_unable
, MAX(gcs_intub) as gcs_intub
, AVG(fall_risk) as fall_risk
, MAX(delirium_scale) as delirium_scale
, AVG(delirium_score) as delirium_score
, MAX(sedation_scale) as sedation_scale
, AVG(sedation_score) as sedation_score
, AVG(sedation_goal) as sedation_goal
, AVG(pain_score) as pain_score
, AVG(pain_goal) as pain_goal
from nc
WHERE gcs IS NOT NULL
OR gcs_motor IS NOT NULL
OR gcs_verbal IS NOT NULL
OR gcs_eyes IS NOT NULL
OR gcs_unable IS NOT NULL
OR gcs_intub IS NOT NULL
OR fall_risk IS NOT NULL
OR delirium_scale IS NOT NULL
OR delirium_score IS NOT NULL
OR sedation_scale IS NOT NULL
OR sedation_score IS NOT NULL
OR sedation_goal IS NOT NULL
OR pain_score IS NOT NULL
OR pain_goal IS NOT NULL
group by patientunitstayid, nursingchartoffset, nursingchartentryoffset
order by patientunitstayid, nursingchartoffset, nursingchartentryoffset;




DROP TABLE IF EXISTS pivoted_treatment_vasopressor CASCADE;
CREATE TABLE pivoted_treatment_vasopressor AS
with tr as
(
  select
    patientunitstayid
   , treatmentoffset as chartoffset
   , max(case when treatmentstring in
   (
     'toxicology|drug overdose|vasopressors|vasopressin' --                                                                   |    23
   , 'toxicology|drug overdose|vasopressors|phenylephrine (Neosynephrine)' --                                                 |    21
   , 'toxicology|drug overdose|vasopressors|norepinephrine > 0.1 micrograms/kg/min' --                                        |    62
   , 'toxicology|drug overdose|vasopressors|norepinephrine <= 0.1 micrograms/kg/min' --                                       |    29
   , 'toxicology|drug overdose|vasopressors|epinephrine > 0.1 micrograms/kg/min' --                                           |     6
   , 'toxicology|drug overdose|vasopressors|epinephrine <= 0.1 micrograms/kg/min' --                                          |     2
   , 'toxicology|drug overdose|vasopressors|dopamine 5-15 micrograms/kg/min' --                                               |     7
   , 'toxicology|drug overdose|vasopressors|dopamine >15 micrograms/kg/min' --                                                |     3
   , 'toxicology|drug overdose|vasopressors' --                                                                               |    30
   , 'surgery|cardiac therapies|vasopressors|vasopressin' --                                                                  |   356
   , 'surgery|cardiac therapies|vasopressors|phenylephrine (Neosynephrine)' --                                                |  1000
   , 'surgery|cardiac therapies|vasopressors|norepinephrine > 0.1 micrograms/kg/min' --                                       |   390
   , 'surgery|cardiac therapies|vasopressors|norepinephrine <= 0.1 micrograms/kg/min' --                                      |   347
   , 'surgery|cardiac therapies|vasopressors|epinephrine > 0.1 micrograms/kg/min' --                                          |   117
   , 'surgery|cardiac therapies|vasopressors|epinephrine <= 0.1 micrograms/kg/min' --                                         |   178
   , 'surgery|cardiac therapies|vasopressors|dopamine  5-15 micrograms/kg/min' --                                             |   274
   , 'surgery|cardiac therapies|vasopressors|dopamine >15 micrograms/kg/min' --                                               |    23
   , 'surgery|cardiac therapies|vasopressors' --                                                                              |   596
   , 'renal|electrolyte correction|treatment of hypernatremia|vasopressin' --                                                 |     7
   , 'neurologic|therapy for controlling cerebral perfusion pressure|vasopressors|phenylephrine (Neosynephrine)' --           |   321
   , 'neurologic|therapy for controlling cerebral perfusion pressure|vasopressors|norepinephrine > 0.1 micrograms/kg/min' --  |   348
   , 'neurologic|therapy for controlling cerebral perfusion pressure|vasopressors|norepinephrine <= 0.1 micrograms/kg/min' -- |   374
   , 'neurologic|therapy for controlling cerebral perfusion pressure|vasopressors|epinephrine > 0.1 micrograms/kg/min' --     |    21
   , 'neurologic|therapy for controlling cerebral perfusion pressure|vasopressors|epinephrine <= 0.1 micrograms/kg/min' --    |   199
   , 'neurologic|therapy for controlling cerebral perfusion pressure|vasopressors|dopamine 5-15 micrograms/kg/min' --         |   277
   , 'neurologic|therapy for controlling cerebral perfusion pressure|vasopressors|dopamine > 15 micrograms/kg/min' --         |    20
   , 'neurologic|therapy for controlling cerebral perfusion pressure|vasopressors' --                                         |   172
   , 'gastrointestinal|medications|hormonal therapy (for varices)|vasopressin' --                                             |   964
   , 'cardiovascular|shock|vasopressors|vasopressin' --                                                                       | 11082
   , 'cardiovascular|shock|vasopressors|phenylephrine (Neosynephrine)' --                                                     | 13189
   , 'cardiovascular|shock|vasopressors|norepinephrine > 0.1 micrograms/kg/min' --                                            | 24174
   , 'cardiovascular|shock|vasopressors|norepinephrine <= 0.1 micrograms/kg/min' --                                           | 17467
   , 'cardiovascular|shock|vasopressors|epinephrine > 0.1 micrograms/kg/min' --                                               |  2410
   , 'cardiovascular|shock|vasopressors|epinephrine <= 0.1 micrograms/kg/min' --                                              |  2384
   , 'cardiovascular|shock|vasopressors|dopamine  5-15 micrograms/kg/min' --                                                  |  4822
   , 'cardiovascular|shock|vasopressors|dopamine >15 micrograms/kg/min' --                                                    |  1102
   , 'cardiovascular|shock|vasopressors' --                                                                                   |  9335
   , 'toxicology|drug overdose|agent specific therapy|beta blockers overdose|dopamine' --                             |    66
   , 'cardiovascular|ventricular dysfunction|inotropic agent|norepinephrine > 0.1 micrograms/kg/min' --                       |   537
   , 'cardiovascular|ventricular dysfunction|inotropic agent|norepinephrine <= 0.1 micrograms/kg/min' --                      |   411
   , 'cardiovascular|ventricular dysfunction|inotropic agent|epinephrine > 0.1 micrograms/kg/min' --                          |   274
   , 'cardiovascular|ventricular dysfunction|inotropic agent|epinephrine <= 0.1 micrograms/kg/min' --                         |   456
   , 'cardiovascular|shock|inotropic agent|norepinephrine > 0.1 micrograms/kg/min' --                                         |  1940
   , 'cardiovascular|shock|inotropic agent|norepinephrine <= 0.1 micrograms/kg/min' --                                        |  1262
   , 'cardiovascular|shock|inotropic agent|epinephrine > 0.1 micrograms/kg/min' --                                            |   477
   , 'cardiovascular|shock|inotropic agent|epinephrine <= 0.1 micrograms/kg/min' --                                           |   505
   , 'cardiovascular|shock|inotropic agent|dopamine <= 5 micrograms/kg/min' --                                        |  1103
   , 'cardiovascular|shock|inotropic agent|dopamine  5-15 micrograms/kg/min' --                                       |  1156
   , 'cardiovascular|shock|inotropic agent|dopamine >15 micrograms/kg/min' --                                         |   144
   , 'surgery|cardiac therapies|inotropic agent|dopamine <= 5 micrograms/kg/min' --                                   |   171
   , 'surgery|cardiac therapies|inotropic agent|dopamine  5-15 micrograms/kg/min' --                                  |    93
   , 'surgery|cardiac therapies|inotropic agent|dopamine >15 micrograms/kg/min' --                                    |     3
   , 'cardiovascular|myocardial ischemia / infarction|inotropic agent|norepinephrine > 0.1 micrograms/kg/min' --              |   688
   , 'cardiovascular|myocardial ischemia / infarction|inotropic agent|norepinephrine <= 0.1 micrograms/kg/min' --             |   670
   , 'cardiovascular|myocardial ischemia / infarction|inotropic agent|epinephrine > 0.1 micrograms/kg/min' --                 |   381
   , 'cardiovascular|myocardial ischemia / infarction|inotropic agent|epinephrine <= 0.1 micrograms/kg/min' --                |   357
   , 'cardiovascular|ventricular dysfunction|inotropic agent|dopamine <= 5 micrograms/kg/min' --                      |   886
   , 'cardiovascular|ventricular dysfunction|inotropic agent|dopamine  5-15 micrograms/kg/min' --                     |   649
   , 'cardiovascular|ventricular dysfunction|inotropic agent|dopamine >15 micrograms/kg/min' --                       |    86
   , 'cardiovascular|myocardial ischemia / infarction|inotropic agent|dopamine <= 5 micrograms/kg/min' --             |   346
   , 'cardiovascular|myocardial ischemia / infarction|inotropic agent|dopamine  5-15 micrograms/kg/min' --            |   520
   , 'cardiovascular|myocardial ischemia / infarction|inotropic agent|dopamine >15 micrograms/kg/min' --              |    54
  ) then 1 else 0 end)::SMALLINT as vasopressor
  from eicuii.treatment
  group by patientunitstayid, treatmentoffset
)
select
  patientunitstayid, chartoffset, vasopressor
from tr
where vasopressor = 1
order by patientunitstayid, chartoffset;


-- This script groups together like vital signs on the same row
--  "major" vital signs (frequently measured) -> pivoted_vital
--  "minor" vital signs (infrequently measured) -> pivoted_vital_other
DROP TABLE IF EXISTS pivoted_vital_other CASCADE;
CREATE TABLE pivoted_vital_other as
-- create columns with only numeric data
with nc as
(
select
    patientunitstayid
  , nursingchartoffset
  , nursingchartentryoffset
  -- pivot data - choose column names for consistency with vitalperiodic
  , case
        WHEN nursingchartcelltypevallabel = 'PA'
        AND  nursingchartcelltypevalname = 'PA Systolic'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as pasystolic
  , case
        WHEN nursingchartcelltypevallabel = 'PA'
        AND  nursingchartcelltypevalname = 'PA Diastolic'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as padiastolic
  , case
        WHEN nursingchartcelltypevallabel = 'PA'
        AND  nursingchartcelltypevalname = 'PA Mean'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as pamean
  , case
        WHEN nursingchartcelltypevallabel = 'SV'
        AND  nursingchartcelltypevalname = 'SV'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as sv
  , case
        WHEN nursingchartcelltypevallabel = 'CO'
        AND  nursingchartcelltypevalname = 'CO'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as co
  , case
        WHEN nursingchartcelltypevallabel = 'SVR'
        AND  nursingchartcelltypevalname = 'SVR'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as svr
  , case
        WHEN nursingchartcelltypevallabel = 'ICP'
        AND  nursingchartcelltypevalname = 'ICP'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as icp
  , case
        WHEN nursingchartcelltypevallabel = 'CI'
        AND  nursingchartcelltypevalname = 'CI'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as ci
  , case
        WHEN nursingchartcelltypevallabel = 'SVRI'
        AND  nursingchartcelltypevalname = 'SVRI'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as svri
  , case
        WHEN nursingchartcelltypevallabel = 'CPP'
        AND  nursingchartcelltypevalname = 'CPP'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as cpp
  , case
        WHEN nursingchartcelltypevallabel = 'SVO2'
        AND  nursingchartcelltypevalname = 'SVO2'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as svo2
  , case
        WHEN nursingchartcelltypevallabel = 'PAOP'
        AND  nursingchartcelltypevalname = 'PAOP'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as paop
  , case
        WHEN nursingchartcelltypevallabel = 'PVR'
        AND  nursingchartcelltypevalname = 'PVR'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as pvr
  , case
        WHEN nursingchartcelltypevallabel = 'PVRI'
        AND  nursingchartcelltypevalname = 'PVRI'
        -- verify it's numeric
        AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as pvri
  , case
      WHEN nursingchartcelltypevallabel = 'IAP'
      AND  nursingchartcelltypevalname = 'IAP'
      -- verify it's numeric
      AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$' and nursingchartvalue not in ('-','.')
        then cast(nursingchartvalue as numeric)
    else null end
  as iap
  from eicuii.nursecharting
  -- speed up by only looking at a subset of charted data
  where nursingchartcelltypecat = 'Vital Signs'
)
select
  patientunitstayid
, nursingchartoffset as chartoffset
, nursingchartentryoffset as entryoffset
, AVG(CASE WHEN pasystolic >= 0 AND pasystolic <= 1000 THEN pasystolic ELSE NULL END) AS pasystolic
, AVG(CASE WHEN padiastolic >= 0 AND padiastolic <= 1000 THEN padiastolic ELSE NULL END) AS padiastolic
, AVG(CASE WHEN pamean >= 0 AND pamean <= 1000 THEN pamean ELSE NULL END) AS pamean
, AVG(CASE WHEN sv >= 0 AND sv <= 1000 THEN sv ELSE NULL END) AS sv
, AVG(CASE WHEN co >= 0 AND co <= 1000 THEN co ELSE NULL END) AS co
, AVG(CASE WHEN svr >= 0 AND svr <= 1000 THEN svr ELSE NULL END) AS svr
, AVG(CASE WHEN icp >= 0 AND icp <= 1000 THEN icp ELSE NULL END) AS icp
, AVG(CASE WHEN ci >= 0 AND ci <= 1000 THEN ci ELSE NULL END) AS ci
, AVG(CASE WHEN svri >= 0 AND svri <= 1000 THEN svri ELSE NULL END) AS svri
, AVG(CASE WHEN cpp >= 0 AND cpp <= 1000 THEN cpp ELSE NULL END) AS cpp
, AVG(CASE WHEN svo2 >= 0 AND svo2 <= 1000 THEN svo2 ELSE NULL END) AS svo2
, AVG(CASE WHEN paop >= 0 AND paop <= 1000 THEN paop ELSE NULL END) AS paop
, AVG(CASE WHEN pvr >= 0 AND pvr <= 1000 THEN pvr ELSE NULL END) AS pvr
, AVG(CASE WHEN pvri >= 0 AND pvri <= 1000 THEN pvri ELSE NULL END) AS pvri
, AVG(CASE WHEN iap >= 0 AND iap <= 1000 THEN iap ELSE NULL END) AS iap
from nc
WHERE pasystolic IS NOT NULL
OR padiastolic IS NOT NULL
OR pamean IS NOT NULL
OR sv IS NOT NULL
OR co IS NOT NULL
OR svr IS NOT NULL
OR icp IS NOT NULL
OR ci IS NOT NULL
OR svri IS NOT NULL
OR cpp IS NOT NULL
OR svo2 IS NOT NULL
OR paop IS NOT NULL
OR pvr IS NOT NULL
OR pvri IS NOT NULL
OR iap IS NOT NULL
group by patientunitstayid, nursingchartoffset, nursingchartentryoffset
order by patientunitstayid, nursingchartoffset, nursingchartentryoffset;



DROP TABLE IF EXISTS pivoted_weight CASCADE;
CREATE TABLE pivoted_weight as
WITH htwt as
(
SELECT
  patientunitstayid
  , hospitaladmitoffset as chartoffset
  , admissionheight as height
  , admissionweight as weight
  , CASE
    -- CHECK weight vs. height are swapped
    WHEN  admissionweight >= 100
      AND admissionheight >  25 AND admissionheight <= 100
      AND abs(admissionheight-admissionweight) >= 20
    THEN 'swap'
    END AS method
  FROM eicuii.patient
)
, htwt_fixed as
(
  SELECT
    patientunitstayid
    , chartoffset
    , 'admit' as weight_type
    , CASE
      WHEN method = 'swap' THEN weight
      WHEN height <= 0.30 THEN NULL
      WHEN height <= 2.5 THEN height*100
      WHEN height <= 10 THEN NULL
      WHEN height <= 25 THEN height*10
      -- CHECK weight in both columns
      WHEN height <= 100 AND abs(height-weight) < 20 THEN NULL
      WHEN height  > 250 THEN NULL
      ELSE height END as height_fixed
    , CASE
      WHEN method = 'swap' THEN height
      WHEN weight <= 20 THEN NULL
      WHEN weight  > 300 THEN NULL
      ELSE weight
      END as weight_fixed
    from htwt
)
-- extract weight from the charted data
, wt1 AS
(
  select
    patientunitstayid, nursingchartoffset as chartoffset
    -- all of the below weights are measured in kg
    , CASE WHEN nursingchartcelltypevallabel IN
        (
            'Admission Weight', 'Admit weight'
        ) THEN 'admit'
    ELSE 'daily' END AS weight_type
    , CAST(nursingchartvalue as NUMERIC) as weight
  from eicuii.nursecharting
  where nursingchartcelltypecat = 'Other Vital Signs and Infusions'
  and nursingchartcelltypevallabel in
  (
      'Admission Weight'
    , 'Admit weight'
    , 'WEIGHT in Kg'
  )
  -- ensure that nursingchartvalue is numeric
  and nursingchartvalue ~ '^([0-9]+\.?[0-9]*|\.[0-9]+)$'
  and nursingchartoffset < 60*24
)
-- weight from intake/output table
, wt2 as
(
  select
    patientunitstayid, intakeoutputoffset as chartoffset
    , 'daily' as weight_type
    , MAX(
        CASE WHEN cellpath = 'flowsheet|Flowsheet Cell Labels|I&O|Weight|Bodyweight (kg)'
        then cellvaluenumeric
      else NULL END
    ) AS weight_kg
    -- there are ~300 extra (lb) measurements compared to kg, so we include both
    , MAX(
        CASE WHEN cellpath = 'flowsheet|Flowsheet Cell Labels|I&O|Weight|Bodyweight (lb)'
        then cellvaluenumeric*0.453592
      else NULL END
    ) AS weight_kg2
  FROM eicuii.intakeoutput
  WHERE CELLPATH IN
  ( 'flowsheet|Flowsheet Cell Labels|I&O|Weight|Bodyweight (kg)'
  , 'flowsheet|Flowsheet Cell Labels|I&O|Weight|Bodyweight (lb)'
  )
  and INTAKEOUTPUTOFFSET < 60*24
  GROUP BY patientunitstayid, intakeoutputoffset
)
-- weight from infusiondrug
, wt3 as
(
  select
    patientunitstayid, infusionoffset as chartoffset
    , 'daily' as weight_type
    , cast(patientweight as NUMERIC) as weight
  from eicuii.infusiondrug
  where patientweight is not null
  and infusionoffset < 60*24
)
-- combine together all weights
SELECT patientunitstayid, chartoffset, 'patient' as source_table, weight_type, weight_fixed as weight
FROM htwt_fixed
WHERE weight_fixed IS NOT NULL
UNION ALL
SELECT patientunitstayid, chartoffset, 'nursecharting' as source_table, weight_type, weight
FROM wt1
WHERE weight IS NOT NULL
UNION ALL
SELECT patientunitstayid, chartoffset, 'intakeoutput' as source_table, weight_type, COALESCE(weight_kg, weight_kg2) as weight
FROM wt2
WHERE weight_kg IS NOT NULL
OR weight_kg2 IS NOT NULL
UNION ALL
SELECT patientunitstayid, chartoffset, 'infusiondrug' as source_table, weight_type, weight
FROM wt3
WHERE weight IS NOT NULL
ORDER BY 1, 2, 3;