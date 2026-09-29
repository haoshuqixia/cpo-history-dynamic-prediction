from common import *

WINDOWS=[24,48,72,'whole']
PAIR_IDS=set(['220052','225312','220181','228640','220235'])|set(SPEC.loc[SPEC.source.str.startswith('CO '),'itemid'])

def norm(u):
    return str(u).lower().replace(' ','').replace('.','').replace('^','').replace('²','2').replace('hour','hr')
ALIASES={
 'W':{'w','watt','watts'},'L/min':{'l/min'},'L/min/m2':{'l/min/m2'},
 'mmHg':{'mmhg'},'%':{'%','percent'},'mL':{'ml'},'mL/min':{'ml/min'},
 'mL/hr':{'ml/hr','ml/h'},'mL/beat':{'ml/beat'},'mL/kg':{'ml/kg'},
 'mL/m2':{'ml/m2'},'1/kOhm':{'1/kohm','/kohm','kohm-1'},
}

def units(values, spec):
    # Event units take precedence; fallback only to explicit dictionary/label unit.
    u=values.fillna('').astype(str).map(norm)
    fallback=norm(spec.get('unitname',''))
    if fallback in ['','none','nan'] and '%' in spec.get('label',''):fallback='%'
    inferred=u.eq('') & (fallback not in ['','none','nan'])
    u.loc[inferred]=fallback
    ok=u.isin(ALIASES[spec['canonical_unit']])
    return ok,inferred,u

def load_context():
    s=pd.read_csv(ROOT/'work/adult_stays.csv',dtype={'stay_id':str,'hadm_id':str,'subject_id':str},parse_dates=['intime','outtime'])
    c=pd.read_csv(ROOT/'work/cohort_memberships.csv.gz',dtype={'stay_id':str,'hadm_id':str,'subject_id':str})
    members={}
    for co,z in c.groupby('cohort',sort=False):
        members[(co,'patient')]=set(z.loc[z.patient_primary,'stay_id'])
        members[(co,'all_stays')]=set(z.stay_id)
    return s,members

def stats_by_stay(z):
    if z.empty:return pd.DataFrame(columns=['stay_id','n','span_h'])
    g=z.groupby('stay_id').agg(n=('charttime','nunique'),first=('charttime','min'),last=('charttime','max')).reset_index()
    g['span_h']=(g['last']-g['first']).dt.total_seconds()/3600
    return g[['stay_id','n','span_h']]

def aggregate_source(events,spec,members,out,stay_output=True,windows=None):
    for view,flag in [('raw','raw'),('numeric_only','numeric'),('QA_valid','valid')]:
        z=events.loc[events[flag].astype(bool)].drop_duplicates(['stay_id','charttime'])
        for win in (WINDOWS if windows is None else windows):
            w=z if win=='whole' else z[z.hours.le(win)]
            g=stats_by_stay(w)
            if stay_output and not g.empty:
                q=g.assign(source=spec['source'],itemid=spec['itemid'],view=view,window=str(win))
                path=ROOT/'work/per_stay_observability.csv.gz'
                q.to_csv(path,index=False,mode='a',header=not path.exists(),compression='gzip')
            for (co,unit),ids in members.items():
                q=g[g.stay_id.isin(ids)];den=len(ids);n=len(q)
                row={'cohort':co,'counting_unit':unit,'domain':spec['domain'],'itemid':spec['itemid'],'source':spec['source'],
                     'canonical_unit':spec['canonical_unit'],'view':view,'window':str(win),'denominator':den,'N_any':n,
                     'median_measurements':q.n.median() if n else np.nan,'median_span_h':q.span_h.median() if n else np.nan,
                     'median_measurements_all':np.median(np.r_[q.n.to_numpy(),np.zeros(den-n)]) if den else np.nan,
                     'median_span_all_h':np.median(np.r_[q.span_h.to_numpy(),np.zeros(den-n)]) if den else np.nan}
                for k in [2,3,4,6]:row[f'N_ge{k}']=int(q.n.ge(k).sum())
                for h in [6,12,24]:row[f'N_ge3_span{h}h']=int((q.n.ge(3)&q.span_h.ge(h)).sum())
                row['Gate']=gate(row['N_ge3_span6h'])
                out.append(row)

def process(x,spec,s):
    it=spec['itemid'];n=len(x)
    if 'valuenum' not in x:x['valuenum']=x['value']
    # Invalid numeric strings are not salvaged from free text.
    x['v']=pd.to_numeric(x.valuenum,errors='coerce')
    x['numeric']=np.isfinite(x.v)
    ok,infer,un=units(x.valueuom,spec)
    x['unit_ok']=ok;x['unit_inferred']=infer
    lo=float(spec['qa_lower']);hi=float(spec['qa_upper'])
    positive=spec['canonical_unit'] in ['W','L/min','L/min/m2','mL/beat','mL/m2','mL/kg','1/kOhm'] or spec['source'].startswith(('MAP','PaCO2'))
    x['bounds_ok']=x.v.le(hi)&(x.v.gt(lo) if positive else x.v.ge(lo))
    x['valid']=x.numeric&ok&x.bounds_ok;x['raw']=True
    x['charttime']=pd.to_datetime(x.charttime,errors='coerce')
    if 'storetime' in x:x['storetime']=pd.to_datetime(x.storetime,errors='coerce')
    u=x.groupby('valueuom',dropna=False).agg(N_raw_rows=('raw','size'),N_numeric_rows=('numeric','sum'),N_unit_compatible=('unit_ok','sum'),N_QA_valid_rows=('valid','sum')).reset_index()
    u['itemid']=it;u['source']=spec['source']
    x=x.merge(s[['stay_id','intime','outtime']],on='stay_id',how='inner',validate='many_to_one')
    x['hours']=(x.charttime-x.intime).dt.total_seconds()/3600
    x=x[x.charttime.ge(x.intime)&x.charttime.le(x.outtime)].copy()
    keys=['stay_id','charttime'];valid=x[x.valid]
    c=valid.groupby(keys).v.nunique()
    conflict=c[c.gt(1)].reset_index(name='different_valid_values')
    write(f'audit/conflicts_{it}.csv.gz',conflict)
    exact=x.duplicated(keys+['value','valuenum','valueuom']).sum()
    dup=x.duplicated(keys).sum()
    counts={'itemid':it,'source':spec['source'],'domain':spec['domain'],'raw_all_rows':n,'adult_within_stay_rows':len(x),
        'raw_unique_times':len(x.drop_duplicates(keys)),'exact_duplicate_rows':int(exact),'additional_same_time_rows':int(dup),
        'valid_unique_times':len(c),'conflicting_valid_times':len(conflict),'numeric_rows':int(x.numeric.sum()),
        'unit_incompatible_or_missing_rows':int((~x.unit_ok).sum()),'unit_inferred_rows':int(x.unit_inferred.sum()),
        'out_of_broad_bounds_rows':int((x.numeric&~x.bounds_ok).sum()),'qa_valid_rows':int(x.valid.sum())}
    if 'warning' in x:
        counts['warning_marked_rows']=int(x.warning.eq('1').sum())
    if 'storetime' in x:
        lag=(x.storetime-x.charttime).dt.total_seconds()/60
        counts.update({'store_lag_median_min':lag.median(),'store_lag_p90_min':lag.quantile(.9),'negative_store_lag_rows':int(lag.lt(0).sum())})
    q=valid.drop_duplicates(keys).sort_values(keys)
    gaps=q.groupby('stay_id').charttime.diff().dt.total_seconds()/3600
    counts.update({'measurement_gap_median_h':gaps.median(),'measurement_gap_p25_h':gaps.quantile(.25),'measurement_gap_p75_h':gaps.quantile(.75),'fraction_gaps_lt1h':gaps.lt(1).sum()/gaps.notna().sum() if gaps.notna().any() else np.nan})
    cols=[k for k in ['subject_id','stay_id','charttime','value','valuenum','valueuom','v','unit_ok','bounds_ok','warning'] if k in x]
    write(f'audit/qa_rejected_{it}.csv.gz',x.loc[~x.valid,cols])
    if it in PAIR_IDS or it=='lab_art_pco2':
        # Preserve all raw anchor timestamps for audit, plus a conflict-free valid flag for pairing.
        base=x.sort_values('charttime').drop_duplicates(keys).copy()
        # Use a valid row if available; raw-invalid first rows must not erase a valid same-time record.
        good=valid.drop_duplicates(keys)[keys+['v']].rename(columns={'v':'valid_v'})
        base=base.merge(good,on=keys,how='left',validate='one_to_one')
        base=base.merge(conflict,on=keys,how='left',validate='one_to_one')
        base['pair_valid']=base.valid_v.notna()&base.different_valid_values.isna()
        base['v']=base.valid_v
        # numeric-only values are for time-only potential pairing; never calculate a pressure difference from them.
        nt=x[x.numeric].groupby(keys).v.nunique().reset_index(name='numeric_nvalues')
        base=base.merge(nt,on=keys,how='left');base['pair_numeric']=base.numeric_nvalues.ge(1)
        base['itemid']=it
        base[['stay_id','charttime','hours','v','pair_valid','pair_numeric','itemid']].to_pickle(ROOT/f'work/pair_{it}.pkl')
    return x,counts,u

def lab_source(s):
    sp=pd.read_csv(ROOT/'work/raw_52033.csv.gz',dtype=str,keep_default_na=False)
    sp['kind']=sp.value.str.upper().str.strip().str.replace('.','',regex=False)
    write('audit/lab_specimen_vocabulary.csv',sp.groupby('kind').size().reset_index(name='rows'))
    sp['art']=sp.kind.isin(['ART','ARTERIAL'])
    sg=sp.groupby(['subject_id','specimen_id']).agg(nkind=('kind','nunique'),art=('art','all')).reset_index()
    sg=sg[sg.art&sg.nkind.eq(1)]
    x=pd.read_csv(ROOT/'work/raw_50818.csv.gz',dtype=str,keep_default_na=False)
    nr=len(x);x=x.merge(sg[['subject_id','specimen_id']],on=['subject_id','specimen_id'],how='inner',validate='many_to_one')
    nart=len(x);x['charttime']=pd.to_datetime(x.charttime,errors='coerce')
    # Subject and ICU time matching; known admission IDs must agree. Missing admission IDs can be resolved only uniquely.
    x=x.rename(columns={'hadm_id':'lab_hadm_id'})
    x=x.merge(s[['subject_id','hadm_id','stay_id','intime','outtime']],on='subject_id',how='inner')
    x=x[x.charttime.ge(x.intime)&x.charttime.le(x.outtime)&(x.lab_hadm_id.eq('')|x.lab_hadm_id.eq(x.hadm_id))]
    ambiguous=x.groupby('labevent_id').stay_id.nunique();bad=set(ambiguous[ambiguous.gt(1)].index)
    x=x[~x.labevent_id.isin(bad)].drop(columns=['intime','outtime'])
    dump('audit/paco2_specimen_selection.json',{'raw_pco2_rows':nr,'explicit_arterial_rows':nart,'mapped_within_adult_icu_rows':len(x),'ambiguous_labevent_ids_excluded':len(bad)})
    return x

def main():
    # This is a generated, reproducible ledger; never append a second run to the first.
    (ROOT/'work/per_stay_observability.csv.gz').unlink(missing_ok=True)
    s,members=load_context();d=pd.read_csv(ROOT/'evidence/variable_contract.csv',dtype=str,keep_default_na=False)
    out=[];qa=[];ua=[]
    for spec in d.to_dict('records'):
        p=ROOT/f'work/raw_{spec["itemid"]}.csv.gz'
        if p.exists():
            x=pd.read_csv(p,dtype=str,keep_default_na=False)
        else:x=pd.DataFrame(columns=['stay_id','charttime','value','valuenum','valueuom'])
        x,q,u=process(x,spec,s);qa.append(q);ua.append(u)
        if spec['domain']!='0':aggregate_source(x,spec,members,out)
        log(f'processed {spec["itemid"]} {spec["source"]}: {q["valid_unique_times"]:,} QA-valid times')
    spec={'itemid':'lab_art_pco2','source':'PaCO2 laboratory explicit arterial','domain':'5','canonical_unit':'mmHg','qa_lower':0,'qa_upper':200,'unitname':'mmHg','label':'pCO2 arterial specimen'}
    x,q,u=process(lab_source(s),spec,s);qa.append(q);ua.append(u);aggregate_source(x,spec,members,out)
    write('results/all_metrics_direct.csv',out);write('audit/source_record_qa.csv',qa);write('audit/unit_record_qa.csv',pd.concat(ua,ignore_index=True))
    dump('audit/aggregate_complete.json',{'complete':True,'sources':len(d)+1,'metric_rows':len(out)})

if __name__=='__main__':main()
