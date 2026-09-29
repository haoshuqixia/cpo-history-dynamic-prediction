from common import *
import sys

def run(table):
    done=ROOT/f'audit/extract_{table}.json'
    if done.exists() and json.loads(done.read_text()).get('complete'):
        log(f'{table}: completed extraction reused; raw files not appended twice')
        return
    d=pd.read_csv(ROOT/'evidence/variable_contract.csv',dtype=str,keep_default_na=False)
    if table=='labevents':
        ids={'50818','52033'}
        cols=['labevent_id','subject_id','hadm_id','specimen_id','itemid','charttime','value','valuenum','valueuom']
        relative='hosp/labevents.csv.gz'
    else:
        ids=set(d.loc[d.linksto.eq(table),'itemid'])
        cols=['subject_id','hadm_id','stay_id','itemid','charttime','storetime','value','valueuom']
        if table=='chartevents': cols+=['valuenum','warning']
        relative='icu/'+table+'.csv.gz'
    counts={}; unit_counts={}; n=0
    for x in filtered_chunks(relative,ids,cols):
        n+=len(x)
        for it,y in x.groupby('itemid',sort=False):
            path=ROOT/f'work/raw_{it}.csv.gz'
            y.to_csv(path,mode='a',header=not path.exists(),index=False,compression='gzip')
            counts[it]=counts.get(it,0)+len(y)
            for unit,nn in y.groupby('valueuom',dropna=False).size().items():
                unit_counts[(it,unit)]=unit_counts.get((it,unit),0)+int(nn)
        log(f'{table}: {n:,} allowlisted rows extracted')
    dump(f'audit/extract_{table}.json',{'complete':True,'selected_records':n,'per_item':counts,'ids':sorted(ids),'columns':cols})
    write(f'audit/units_{table}.csv',[{'itemid':a,'valueuom':b,'rows':v} for (a,b),v in unit_counts.items()])

if __name__=='__main__':run(sys.argv[1])
