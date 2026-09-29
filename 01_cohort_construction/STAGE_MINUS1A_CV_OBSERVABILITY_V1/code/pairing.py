from aggregate import *

def nearest(a,b,minutes):
    cols=['stay_id','charttime','hours','anchor_v','partner_time','partner_itemid','partner_v','signed_delta_min','abs_delta_min']
    if a.empty or b.empty:return pd.DataFrame(columns=cols)
    aa=a[['stay_id','charttime','hours','v']].rename(columns={'v':'anchor_v'}).sort_values('charttime').drop_duplicates(['stay_id','charttime'])
    bb=b[['stay_id','charttime','itemid','v','priority']].sort_values(['charttime','priority','itemid']).drop_duplicates(['stay_id','charttime'])
    bb=bb.rename(columns={'charttime':'partner_time','itemid':'partner_itemid','v':'partner_v'})
    args=dict(left_on='charttime',right_on='partner_time',by='stay_id',tolerance=pd.Timedelta(minutes=minutes))
    left=pd.merge_asof(aa,bb,direction='backward',**args)
    right=pd.merge_asof(aa,bb,direction='forward',**args)
    dl=(left.partner_time-left.charttime).dt.total_seconds().abs()
    dr=(right.partner_time-right.charttime).dt.total_seconds().abs()
    choose=dr.notna()&(dl.isna()|dr.lt(dl)|(dr.eq(dl)&right.priority.lt(left.priority)))
    left.loc[choose,:]=right.loc[choose,:]
    z=left[left.partner_time.notna()].copy()
    z['signed_delta_min']=(z.partner_time-z.charttime).dt.total_seconds()/60
    z['abs_delta_min']=z.signed_delta_min.abs()
    return z[cols]

def unique_partner(z):
    return z.sort_values(['abs_delta_min','charttime']).drop_duplicates(['stay_id','partner_itemid','partner_time'])

def audit_pairs(p,spec,view,tol,members,rows,windows=None):
    for (co,unit),ids in members.items():
        z=p[p.stay_id.isin(ids)]
        for win in (WINDOWS if windows is None else windows):
            w=z if win=='whole' else z[z.hours.le(win)]
            reuse=w.duplicated(['stay_id','partner_itemid','partner_time']).sum()
            rows.append({'source':spec['source'],'cohort':co,'counting_unit':unit,'view':view,'tolerance_min':tol,'window':str(win),
              'N_pairs':len(w),'N_stays':w.stay_id.nunique(),'partner_reused_extra_pairs':int(reuse),
              'abs_delta_p25_min':w.abs_delta_min.quantile(.25),'abs_delta_median_min':w.abs_delta_min.median(),
              'abs_delta_p75_min':w.abs_delta_min.quantile(.75),'abs_delta_p90_min':w.abs_delta_min.quantile(.9),
              'signed_delta_median_min':w.signed_delta_min.median(),
              'N_exact':int(w.abs_delta_min.eq(0).sum()),'N_invasive_MAP':int(w.partner_itemid.isin(['220052','225312']).sum()),
              'N_noninvasive_MAP':int(w.partner_itemid.eq('220181').sum())})

def events_from_pairs(pairs,no_reuse=False):
    frames=[]
    for view,p in pairs.items():
        if no_reuse:p=unique_partner(p)
        z=p[['stay_id','charttime','hours']].copy();z[view]=True;frames.append(z)
    e=pd.concat(frames,ignore_index=True)
    if e.empty:return pd.DataFrame(columns=['stay_id','charttime','hours','raw','numeric','valid'])
    for c in ['raw','numeric','valid']:e[c]=e[c].eq(True)
    return e.groupby(['stay_id','charttime','hours'],as_index=False)[['raw','numeric','valid']].max()

def main():
    s,members=load_context();out=[];ar=[]
    maps=pd.concat([pd.read_pickle(ROOT/f'work/pair_{it}.pkl') for it in ['220052','225312','220181']],ignore_index=True)
    maps['priority']=maps.itemid.map({'220052':0,'225312':0,'220181':1})
    # Limit map tables to actual CO-anchor stays before grouping.
    configs=[]
    for spec in SPEC[SPEC.source.str.startswith('CO ')].to_dict('records'):
        a=pd.read_pickle(ROOT/f'work/pair_{spec["itemid"]}.pkl')
        configs.append((a,maps[maps.stay_id.isin(set(a.stay_id))].copy(),{
            'itemid':'recon_'+spec['itemid'],'source':'reconstructed CPO - '+spec['source'],'domain':1,'canonical_unit':'W'},True))
    del maps
    et=pd.read_pickle(ROOT/'work/pair_228640.pkl');et['priority']=0
    for it,name in [('lab_art_pco2','PaCO2 blood-sample to EtCO2'),('220235','PaCO2 charted to EtCO2 (not sampling time)')]:
        a=pd.read_pickle(ROOT/f'work/pair_{it}.pkl')
        configs.append((a,et,{'itemid':'pair_'+it,'source':name,'domain':5,'canonical_unit':'mmHg'},False))
    for a,b,spec,iscpo in configs:
        for tol in [0,15,30]:
          for win in WINDOWS:
            aw=a if win=='whole' else a[a.hours.le(win)]
            bw=b if win=='whole' else b[b.hours.le(win)]
            pairs={}
            for view,col in [('raw',None),('numeric','pair_numeric'),('valid','pair_valid')]:
                aa=aw if col is None else aw[aw[col]]
                bb=bw if col is None else bw[bw[col]]
                # PaCO2 time-only pairing still requires QA-valid arterial PaCO2 anchors;
                # only the EtCO2 side has a numeric-only, unit-unresolved relaxation.
                if not iscpo and view=='numeric':aa=aw[aw.pair_valid]
                p=nearest(aa,bb,tol)
                if iscpo and view=='valid':
                    p['CPO_W']=pd.to_numeric(p.anchor_v)*pd.to_numeric(p.partner_v)/451
                    p=p[p.CPO_W.gt(0)&p.CPO_W.le(20)]
                # Never calculate a PaCO2-EtCO2 difference here.
                pairs[view]=p
                audit_pairs(p,spec,view,tol,members,ar,windows=[win])
                if tol==15 and view in ['numeric','valid']:
                    write(f'work/pair_ledger_{spec["itemid"]}_{view}_15_{win}.csv.gz',p)
            for nonreuse in [False,True]:
                derived=dict(spec)
                suffix='' if tol==15 and not nonreuse else f' [tol{tol}min'+('; unique partner]' if nonreuse else ']')
                derived['source']+=suffix;derived['itemid']+=f'_tol{tol}_'+('unique' if nonreuse else 'reuse')
                e=events_from_pairs(pairs,nonreuse)
                aggregate_source(e,derived,members,out,stay_output=False,windows=[win])
          log(f'paired {spec["source"]} ±{tol} min: {len(pairs["valid"]):,} valid / {len(pairs["numeric"]):,} numeric time pairs (whole stay)')
    write('results/all_metrics_paired.csv',out);write('audit/pairing_timing_and_reuse.csv',ar)
    dump('audit/pairing_complete.json',{'complete':True,'metric_rows':len(out),'audit_rows':len(ar)})

if __name__=='__main__':main()
