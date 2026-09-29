from common import *

def operation(r):
    c,v,t=r.icd_code,r.icd_version,r.long_title.lower()
    if v=='9':
        if c.startswith('361'): return 'CABG'
        if c.startswith(('351','352')) or c in ['3501','3502','3503','3504','3531','3532','3533','3534','3535','3539']:return 'surgical_valve'
        if c in ['3551','3553','3581','3582','3583','3584','3591','3592','3593','3594','3595','3732','3733','3735']:return 'open_cardiac_repair'
        if c=='3751':return 'heart_transplant'
        if c in ['3752','3766']:return 'implanted_VAD_or_artificial_heart'
        return ''
    if 'open approach' not in t:return ''
    if 'bypass coronary artery' in t:return 'CABG'
    if 'valve' in t and any(x in t for x in ['replacement','repair','supplement','restriction','release']):return 'surgical_valve'
    if 'transplantation of heart' in t:return 'heart_transplant'
    if c.startswith('02') and any(x in t for x in ['repair of','supplement','replacement of','excision of','resection of','reposition of']) and any(x in t for x in ['atrium','ventricle','septum','heart']):return 'open_cardiac_repair'
    if c=='02HA0QZ':return 'implanted_VAD_or_artificial_heart'
    return ''

def main():
    d=read('hosp/d_icd_diagnoses.csv.gz',['icd_code','icd_version','long_title'])
    d['cohort']=''
    d.loc[(d.icd_version.eq('9')&d.icd_code.str.startswith('410'))|(d.icd_version.eq('10')&d.icd_code.str.startswith(('I21','I22'))),'cohort']='AMI'
    d.loc[d.long_title.str.contains('acute',case=False)&d.long_title.str.contains('heart failure',case=False),'cohort']='AHF'
    d.loc[(d.icd_version.eq('9')&d.icd_code.eq('78551'))|(d.icd_version.eq('10')&d.icd_code.eq('R570')),'cohort']='CS'
    d=d[d.cohort.ne('')];write('evidence/diagnosis_codes_frozen.csv',d)
    dx=pd.concat(list(filtered_chunks('hosp/diagnoses_icd.csv.gz',set(d.icd_code),['subject_id','hadm_id','seq_num','icd_code','icd_version'],key='icd_code')),ignore_index=True)
    dx=dx.merge(d,on=['icd_code','icd_version'],how='inner')
    flags=dx[['hadm_id','cohort']].drop_duplicates()
    p=read('hosp/patients.csv.gz',['subject_id','anchor_age','anchor_year'])
    s=read('icu/icustays.csv.gz',['subject_id','hadm_id','stay_id','intime','outtime'])
    s['intime']=pd.to_datetime(s.intime);s['outtime']=pd.to_datetime(s.outtime)
    s=s.merge(p,on='subject_id',validate='many_to_one')
    s['age']=pd.to_numeric(s.anchor_age)+s.intime.dt.year-pd.to_numeric(s.anchor_year)
    s=s[s.age.ge(18)].sort_values(['intime','stay_id']).copy()
    write('work/adult_stays.csv',s)
    dp=read('hosp/d_icd_procedures.csv.gz',['icd_code','icd_version','long_title'])
    dp['operation_class']=dp.apply(operation,axis=1)
    dp=dp[dp.operation_class.ne('')];write('evidence/surgery_codes_frozen.csv',dp)
    pro=pd.concat(list(filtered_chunks('hosp/procedures_icd.csv.gz',set(dp.icd_code),['subject_id','hadm_id','chartdate','icd_code','icd_version'],key='icd_code')),ignore_index=True)
    pro=pro.merge(dp,on=['icd_code','icd_version'],how='inner');pro['chartdate']=pd.to_datetime(pro.chartdate)
    write('work/qualifying_operations.csv.gz',pro)
    op=s[['subject_id','hadm_id','stay_id','intime']].merge(pro,on=['subject_id','hadm_id'],how='inner')
    op['strict']=op.chartdate.lt(op.intime.dt.normalize())
    op['same_day']=op.chartdate.eq(op.intime.dt.normalize())
    write('audit/surgery_stay_timing.csv.gz',op)
    strict=set(op.loc[op.strict,'stay_id']);same=set(op.loc[op.same_day,'stay_id'])-strict
    memberships=[]
    for name in ['AMI','AHF','CS','Cardiac_surgery_strict','Adult_ICU','Cardiac_surgery_same_day_only','Cardiac_surgery_expanded']:
        if name in ['AMI','AHF','CS']:z=s[s.hadm_id.isin(flags.loc[flags.cohort.eq(name),'hadm_id'])].copy()
        elif name=='Cardiac_surgery_strict':z=s[s.stay_id.isin(strict)].copy()
        elif name=='Cardiac_surgery_same_day_only':z=s[s.stay_id.isin(same)].copy()
        elif name=='Cardiac_surgery_expanded':z=s[s.stay_id.isin(strict|same)].copy()
        else:z=s.copy()
        z['cohort']=name;z['patient_primary']=~z.duplicated('subject_id')
        memberships.append(z)
    c=pd.concat(memberships,ignore_index=True)
    write('work/cohort_memberships.csv.gz',c)
    out=[]
    for name,z in c.groupby('cohort',sort=False):
        q=z[z.patient_primary]
        out.append({'cohort':name,'N_patients':len(q),'N_all_stays':len(z),'unique_patients_all_stays':z.subject_id.nunique(),
            'primary_stays_duration_ge24h':int(((q.outtime-q.intime).dt.total_seconds()>=86400).sum()),
            'primary_stays_duration_ge48h':int(((q.outtime-q.intime).dt.total_seconds()>=172800).sum()),
            'primary_stays_duration_ge72h':int(((q.outtime-q.intime).dt.total_seconds()>=259200).sum())})
    write('results/cohort_denominators.csv',out)
    write('audit/surgery_operation_class_counts.csv',pro.groupby('operation_class').agg(N_records=('hadm_id','size'),N_hadm=('hadm_id','nunique'),N_patients=('subject_id','nunique')).reset_index())
    dump('audit/cohort_complete.json',{'complete':True,'adult_stays':len(s),'adult_patients':s.subject_id.nunique(),'outcomes_read':False})
    log(out)

if __name__=='__main__':main()
