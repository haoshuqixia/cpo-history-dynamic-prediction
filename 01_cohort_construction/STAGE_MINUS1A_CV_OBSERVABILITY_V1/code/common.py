import os
from pathlib import Path
import csv, gzip, hashlib, json, subprocess, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
M=Path(os.environ["CPO_MIMIC_DIR"])
def write(path, x):
    path=ROOT/path; path.parent.mkdir(parents=True,exist_ok=True)
    (x if isinstance(x,pd.DataFrame) else pd.DataFrame(x)).to_csv(path,index=False)
def dump(path,x):
    (ROOT/path).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str))
def log(x): print(time.strftime('%H:%M:%S'),x,flush=True)
def read(path,cols=None): return pd.read_csv(M/path,usecols=cols,dtype=str,keep_default_na=False)
def gate(n): return 'A' if n>=500 else 'B' if n>=300 else 'C' if n>=150 else 'FAIL'

# itemid, domain, analytical source, canonical unit, broad pre-count QA bounds.
SPECS=[
 (229896,1,'direct CPO (Impella category)','W',0,20),
 (220088,2,'CO thermodilution','L/min',0,40),(224842,2,'CO CCO','L/min',0,40),
 (227543,2,'CO arterial','L/min',0,40),(228178,2,'CO PiCCO','L/min',0,40),
 (228369,2,'CO NICOM','L/min',0,40),(229897,2,'CO Impella','L/min',0,40),
 (226858,2,'CO PA-line insertion','L/min',0,40),(226859,2,'CI PA-line insertion','L/min/m2',0,25),
 (228177,2,'CI PiCCO','L/min/m2',0,25),(228368,2,'CI NICOM','L/min/m2',0,25),
 (220059,3,'PASP','mmHg',-20,250),(220060,3,'PADP','mmHg',-20,250),
 (220061,3,'mPAP','mmHg',-20,250),(223771,3,'PCWP','mmHg',-20,150),
 (224654,3,'PAEDP','mmHg',-20,150),(220074,3,'CVP','mmHg',-50,150),
 (229668,3,'PA diastolic signal','mmHg',-20,250),
 (223772,4,'SvO2','%',0,100),(225674,4,'Mixed Venous O2 Sat','%',0,100),
 (226541,4,'ScvO2 Central Venous O2 Sat legacy','%',0,100),
 (227549,4,'ScvO2 Presep','%',0,100),(227685,4,'ZCentral Venous O2 Sat','%',0,100),
 (227686,4,'Central Venous O2 Sat','%',0,100),
 (228640,5,'EtCO2','mmHg',0,200),
 (226588,6,'Chest Tube 1','mL',0,20000),(226589,6,'Chest Tube 2','mL',0,20000),
 (229413,6,'Chest Tube 3','mL',0,20000),(229414,6,'Chest Tube 4','mL',0,20000),
 (226592,6,'Mediastinal drainage','mL',0,20000),(226612,6,'Pericardial drainage','mL',0,20000),
 (224191,7,'CRRT hourly patient fluid removal','mL',0,20000),
 (224144,7,'CRRT blood flow','mL/min',0,3000),
 (224153,7,'CRRT replacement rate','mL/hr',0,100000),
 (224154,7,'CRRT dialysate rate','mL/hr',0,100000),
 (228005,7,'CRRT prefilter replacement rate','mL/hr',0,100000),
 (228006,7,'CRRT postfilter replacement rate','mL/hr',0,100000),
 (226457,7,'Ultrafiltrate output (not net patient UF)','mL',0,100000),
 (227547,8,'SV arterial','mL/beat',0,1000),(228374,8,'SV NICOM','mL/beat',0,1000),
 (227546,8,'SVV arterial','%',0,100),(228184,8,'SVV PiCCO','%',0,100),
 (228376,8,'SVV NICOM','%',0,100),(228179,8,'ELWI PiCCO','mL/kg',0,100),
 (228180,8,'GEDI PiCCO','mL/m2',0,10000),(228181,8,'ITBVI PiCCO','mL/m2',0,15000),
 (228182,8,'SVI PiCCO','mL/m2',0,1000),(228375,8,'SVI NICOM (dictionary unit conflict)','mL/m2',0,1000),
 (228380,8,'TFC NICOM','1/kOhm',0,1000),(228378,8,'TFCd NICOM','%',-10000,10000),
 (228379,8,'TFCd0 NICOM','%',-10000,10000),
 (220052,0,'MAP arterial','mmHg',0,300),(225312,0,'MAP ART','mmHg',0,300),
 (220181,0,'MAP noninvasive','mmHg',0,300),
 (220235,0,'PaCO2 charted arterial','mmHg',0,200),
]
SPEC=pd.DataFrame(SPECS,columns=['itemid','domain','source','canonical_unit','qa_lower','qa_upper'])
SPEC['itemid']=SPEC.itemid.astype(str)

def filtered_chunks(relative, ids, cols, chunksize=300000, key='itemid'):
    # Only allowlisted rows enter the CSV parser; other clinical variables are not analyzed.
    with gzip.open(M/relative,'rt') as f: header=next(csv.reader(f))
    i=header.index(key)
    pattern=r'^(?:[^,]*,){'+str(i)+r'}(?:'+'|'.join(sorted(ids))+r'),'
    dec=subprocess.Popen(['gzip','-cd',str(M/relative)],stdout=subprocess.PIPE)
    sel=subprocess.Popen(['rg','-a',pattern],stdin=dec.stdout,stdout=subprocess.PIPE)
    dec.stdout.close()
    try:
        yield from pd.read_csv(sel.stdout,names=header,header=None,usecols=cols,dtype=str,
                               keep_default_na=False,chunksize=chunksize)
    except pd.errors.EmptyDataError: pass
    finally:
        sel.stdout.close(); rc=sel.wait(); dc=dec.wait()
        if rc not in [0,1] or dc!=0: raise RuntimeError((relative,rc,dc))

if __name__=='__main__':
    d=read('icu/d_items.csv.gz')
    contract=SPEC.merge(d,on='itemid',how='left',validate='one_to_one')
    assert contract.label.notna().all()
    write('evidence/variable_contract.csv',contract)
    dump('audit/source_files.json',[{'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
        for p in [M/'icu/chartevents.csv.gz',M/'icu/outputevents.csv.gz',M/'hosp/labevents.csv.gz',
                  M/'icu/icustays.csv.gz',M/'hosp/patients.csv.gz',M/'hosp/diagnoses_icd.csv.gz',M/'hosp/procedures_icd.csv.gz']])
