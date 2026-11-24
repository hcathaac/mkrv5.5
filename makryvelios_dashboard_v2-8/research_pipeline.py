"""Native, approval-gated research runner. Local trusted workspace; no arbitrary code dispatch.
The LLM proposes configuration/prose; original workbench functions do all computations.
"""
from __future__ import annotations
import argparse, hashlib, io, json, math, os, re, shutil, subprocess, sys, tempfile, uuid, zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd
from analytics_core import (read_tabular_bytes, combine_frames, data_dictionary, descriptive_statistics, correlation_matrix, fit_detailed_model, monte_carlo_ols, normality_tests, group_tests, pca_table, arima_forecast, p_adjust)
from workbench_v6 import audit_all, clean, frame_hash, loan_forecast, recommend
from research_platform import validation, peer_review, publication_pack, funding_simulation, scenario_figures
from research_chair import extract_pdf_collection, docx_bytes
from output_guidance import build_output_guide, output_guide_markdown

ROOT=Path(__file__).resolve().parent
SCHEMA='mkrv-native-pipeline-1'
GATES=['Preflight & independent audit','Cleaning & integration','Literature & research design','Native execution & validation','Manuscript & evidence mapping','Editorial review','Release review & package']
MAX_FILE=50*1024*1024; MAX_TOTAL=200*1024*1024
DATA_EXT={'.xlsx','.xlsm','.xls','.csv','.tsv','.parquet','.pq','.feather','.arrow'}
DOC_EXT={'.pdf','.docx'}

def now(): return datetime.now(timezone.utc).isoformat()
def digest(b): return hashlib.sha256(b).hexdigest()
def jb(x): return json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=str).encode()
def load(p): return json.loads(p.read_bytes())
def code_fingerprint():
    files=sorted([*ROOT.glob('*.py'),*ROOT.glob('requirements*.txt'),ROOT/'VERSION.txt',*ROOT.glob('data/*.geojson'),*ROOT.glob('data/*.csv')]); return digest(jb({str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in files}))
def git_commit():
    try:
        r=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True,timeout=3,check=True).stdout.strip()
        return r if re.fullmatch(r'[0-9a-f]{40,64}',r) else None
    except (OSError,subprocess.SubprocessError): return None

def env():
    out={'python':sys.version.split()[0]}
    for p in ['numpy','pandas','scipy','statsmodels','scikit-learn','linearmodels','pyarrow','pypdf','python-docx','openpyxl','matplotlib','plotly','streamlit','mcp']:
        try: out[p]=version(p)
        except PackageNotFoundError: out[p]=None
    return out

def require(test,msg):
    if not test: raise ValueError(msg)
def obj(x,allowed,required=()):
    require(isinstance(x,dict),'Expected JSON object'); require(not(set(x)-set(allowed)),f'Unsupported keys: {set(x)-set(allowed)}'); require(set(required)<=set(x),f'Missing keys: {set(required)-set(x)}')
def nonempty(x): return isinstance(x,str) and bool(x.strip())
def safe_name(name):
    require(isinstance(name,str) and bool(re.fullmatch(r'[\w .()\-]+',name,flags=re.UNICODE)) and name not in ('.','..') and len(name)<=200,'Use a plain safe filename'); return name

def write_atomic(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp'); tmp.write_bytes(raw); os.replace(tmp,path)

@contextmanager
def lock(root):
    p=root/'.pipeline.lock'
    try: fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError: raise ValueError('Workspace is locked; another operation may be running. Recover a stale lock only after verifying no process is active.')
    os.close(fd)
    try: yield
    finally: p.unlink(missing_ok=True)

def parse_sources(root,state):
    frames={}
    for f in state['inputs']:
        if f['kind']!='dataset': continue
        raw=(root/f['path']).read_bytes(); require(digest(raw)==f['sha256'],'Input checksum mismatch: '+f['name'])
        for label,d in read_tabular_bytes(f['name'],raw,all_sheets=True).items():
            require(label not in frames,'Duplicate dataset label'); require(len(d)<=500000 and d.shape[1]<=1000 and d.size<=5_000_000,'Pipeline source exceeds row/column/cell cap')
            frames[label]=d
    require(0<len(frames)<=60 and sum(f.size for f in frames.values())<=10_000_000,'Supply 1–60 parsed tables within 10 million total cells'); return frames

def verified(root,state):
    require(state['schema']==SCHEMA,'Unsupported checkpoint schema')
    require(state['code_sha256']==code_fingerprint(),'Application code changed; use the pinned build or create a new run')
    require(state['environment']==env(),'Runtime dependency versions changed; resume with the recorded environment or start a new run')
    for f in state['inputs']+state['artifacts']:
        p=PurePosixPath(f['path']); require(not p.is_absolute() and '..' not in p.parts,'Unsafe manifest path')
        target=(root/p).resolve(); require(target.is_relative_to(root.resolve()) and target.is_file(),'Missing/unsafe artifact: '+f['path'])
        require(digest(target.read_bytes())==f['sha256'],'Checksum mismatch: '+f['path'])
    return state

def state_read(root): return verified(root,load(root/'run_state.json'))
def state_write(root,s): s['updated_utc']=now(); write_atomic(root/'run_state.json',jb(s))

def add_file(root,s,path,raw,kind='artifact'):
    require(len(raw)<=150*1024*1024 and sum(x['bytes'] for x in s['artifacts'])+len(raw)<=600*1024*1024,'Artifact size limit reached; split the analysis into smaller runs')
    write_atomic(root/path,raw); s['artifacts'].append({'path':path,'sha256':digest(raw),'bytes':len(raw),'kind':kind})

def new_run(root,files,question):
    root=Path(root); require(nonempty(question) and len(question)<=12000,'Supply a research objective (max 12000 characters)')
    require(not root.exists() or not any(root.iterdir()),'Use an empty workspace; never overwrite a run')
    require(1<=len(files)<=24 and sum(len(b) for _,b in files)<=MAX_TOTAL,'Input count/size exceeds limit')
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    with lock(root):
        s={'schema':SCHEMA,'run_id':uuid.uuid4().hex,'application_version':(ROOT/'VERSION.txt').read_text().strip(),'code_sha256':code_fingerprint(),'git_commit':git_commit(),'environment':env(),'created_utc':now(),'question':question,'next_gate':0,'inputs':[],'artifacts':[],'approvals':[],'pending':None,'history':[],'tables':{},'revision':0,'status':'initialising','human_review_required':True}
        seen=set()
        for n,b in files:
            n=safe_name(n); require(n.casefold() not in seen,'Duplicate filenames'); seen.add(n.casefold()); ext=Path(n).suffix.lower(); require(ext in DATA_EXT|DOC_EXT,'Unsupported input: '+n); require(0<len(b)<=MAX_FILE,'Invalid input size: '+n)
            path='inputs/'+n; write_atomic(root/path,b); s['inputs'].append({'name':n,'path':path,'sha256':digest(b),'bytes':len(b),'kind':'dataset' if ext in DATA_EXT else 'document'})
        frames=parse_sources(root,s); audits=audit_all(frames)
        add_file(root,s,'00_preflight/audits.json',jb(audits),'independent_audit')
        add_file(root,s,'00_preflight/environment.json',jb(s['environment']),'environment')
        s['sources']=list(frames); s['next_gate']=1; s['status']='awaiting_approval'; s['history'].append({'gate':0,'at':now(),'status':'completed','note':'Fresh workspace, no prior results/cache reused. Audit is parser-level; spreadsheet readers may normalise headers before audit.'}); state_write(root,s)
    return s

CLEAN_KEYS=['trim','normalise_headers','remove_duplicates','missing_tokens','numeric_columns','decimal_comma']
def prepare(root,s,c):
    obj(c,['cleaning','integration','missingness_policy','review_note'],['cleaning','integration','missingness_policy','review_note'])
    require(nonempty(c['missingness_policy']) and nonempty(c['review_note']),'Record missingness and audit decisions')
    raw=parse_sources(root,s); require(isinstance(c['cleaning'],dict) and set(c['cleaning'])==set(raw),'Provide cleaning settings for every parsed source ({} for defaults)')
    frames={}; changes={}
    for n,f in raw.items():
        cfg=c['cleaning'][n]; obj(cfg,CLEAN_KEYS)
        for k in ('trim','normalise_headers','remove_duplicates','decimal_comma'):
            if k in cfg: require(type(cfg[k]) is bool,k+' must be boolean')
        for k in ('missing_tokens','numeric_columns'):
            if k in cfg: require(isinstance(cfg[k],list) and all(isinstance(x,str) for x in cfg[k]),k+' must be a list of strings')
        frames[n],changes[n]=clean(f,**cfg)
    i=c['integration']; obj(i,['mode','sources','keys','how','reason','confirm_row_order'],['mode','sources','reason']); mode=i['mode']; ns=i['sources']; require(isinstance(ns,list) and ns and len(ns)==len(set(ns)) and set(ns)<=set(frames),'Select distinct known sources'); require(nonempty(i['reason']),'Document units, grain and compatibility')
    selected={n:frames[n] for n in ns}
    if mode=='separate': require(len(ns)==1,'Choose one active source or integrate explicitly'); active=selected[ns[0]].copy()
    elif mode=='append':
        require(len(ns)>1,'Append needs two sources'); require(all(set(f.columns)==set(next(iter(selected.values())).columns) for f in selected.values()),'Append requires matching columns; align definitions first'); require(all('__source_dataset__' not in f for f in selected.values()),'Reserved lineage column collision'); active=combine_frames(selected,'Append rows (union by column name)')
    elif mode=='join':
        keys=i.get('keys',[]); how=i.get('how'); require(len(ns)>1 and isinstance(keys,list) and keys and len(keys)==len(set(keys)) and how in ['inner','left','right','outer'],'Specify unique keys and join type')
        for f in selected.values():
            require('__source_dataset__' not in f and set(keys)<=set(f.columns),'Invalid join columns or reserved lineage name'); require(not f[keys].isna().any().any() and not f.duplicated(keys).any(),'Pipeline join is one-to-one only: resolve missing/duplicate keys first')
        active=combine_frames(selected,'Join datasets on key(s)',keys,how)
    elif mode=='side_by_side':
        require(len(ns)>1 and len({len(f) for f in selected.values()})==1 and i.get('confirm_row_order') is True,'Positional alignment requires equal lengths and explicit verified observation order'); active=combine_frames(selected,'Combine columns side-by-side (by row order)')
    else: raise ValueError('Unknown integration mode')
    audit=audit_all(frames); active_audit=audit_all({'active':active})['active']; require(active_audit['rows'] and active_audit['columns'] and not active_audit['duplicate_headers'] and not any(x['infinite'] for x in active_audit['fields']),'Active dataset failed mandatory audit')
    require(len(active)<=500000 and active.shape[1]<=1000 and active.size<=5_000_000,'Integrated dataset exceeds limits')
    return frames,active,changes,audit,active_audit

# Allowlist of native functions with explicit parameter domains. No arbitrary imports/eval.
# All listed parameters must be supplied (including defaults shown in schema).
METHODS={
 'descriptive':{'columns':[]}, 'correlation':{'columns':[],'method':'pearson'}, 'normality':{'columns':[]}, 'group_tests':{'outcomes':[],'group':''},
 'ols':{'y':'','x_vars':[],'covariance':'HC3'}, 'monte_carlo_ols':{'y':'','x_vars':[],'simulations':2000,'method':'Wild bootstrap','confidence':.95,'seed':42},
 'pca':{'columns':[],'n_components':2}, 'arima':{'variable':'','time_col':'','order':[1,0,0],'steps':5},
 'loan_forecast':{'date_col':'','value_col':'','frequency':'Y','steps':5,'aggregation':'sum'},
 'panel':{'entity':'','time':'','y':'','x_vars':[],'aggregation':'Mean','covariance':'Clustered by entity'},
 'prediction':{'y':'','x_vars':[],'folds':5,'seed':42},
 'funding_scenario':{'budget':0.0,'absorption':.75,'budget_shock':0.0,'absorption_sd':.1,'draws':5000,'seed':42},
 'greece_map':{'geography':'','metric':'','aggregation':'Sum','level':'NUTS 2 – Regions (13)'},
 'portfolio_mc':{'cost':'','benefit':'','budget':0.,'project_id':None,'simulations':1000,'cost_cv':.1,'benefit_cv':.2,'correlation':0.,'seed':42},
 'iv_2sls':{'y':'','endogenous':'','instruments':[],'exogenous':[]},
 'did':{'y':'','treatment':'','post':'','controls':[],'cluster':None},
 'clustering':{'columns':[],'method':'K-means','clusters':4,'scaling':'Standard (z-score)','automatic_k':True,'max_k':10,'linkage':'ward','eps':.7,'min_samples':5,'seed':42},
 'regularised':{'y':'','x_vars':[],'method':'Ridge','alpha':1.,'l1_ratio':.5}}
NOTES={'ols':'Continuous outcome; HC0–HC3 or classical covariance; confirm identification and dependence.','monte_carlo_ols':'100–20,000 draws, confidence .80–.999, seed 0–2^32−1. Wild/residual/parametric bootstrap; confirm independent observations and valid OLS.','prediction':'Random-fold CV only; do not use for temporal/grouped observations without a suitable design; justify no leakage.','arima':'Regular, ordered time series; confirm equally spaced and unique time keys.','pca':'Native median imputation and standardisation; explicitly approve.','funding_scenario':'Conditional simulation, not a fitted prediction or causal effect.','greece_map':'Use valid NUTS codes or recognised NUTS2 aliases; preserve boundary vintage.','panel':'Requires linearmodels and verified entity/time keys.','correlation':'Pearson/Spearman/Kendall, not causation.','portfolio_mc':'Greedy stochastic benefit/cost selection; not exact ITA/GAMS optimisation.','iv_2sls':'Instrument relevance and exclusion restrictions require independent identification review.','did':'Binary treatment/post; parallel trends, no anticipation and no differential shocks require separate review.','clustering':'Native median imputation and selected scaling; report stability and limits.','regularised':'Native deterministic random holdout; not temporal/group-aware validation.'}
def schema(): return {'gates':dict(enumerate(GATES)),'methods':{n:{'parameters':p,'note':NOTES.get(n,'Explicit mappings and suitable assumptions required.')} for n,p in METHODS.items()},'approval':'Local human UI/CLI must approve; MCP cannot approve.','source_selection':'Each plan may specify source: active or an exact audited, cleaned source name.','multiplicity':'Runner adjusts the joint OLS slope family only (when chosen); other method families require separate documented review.','unsupported':'Other native methods remain in original modules; use their native exports. No fallback estimation.'}

def cols(frame,names,numeric=True):
    require(isinstance(names,list) and names and len(names)==len(set(names)) and set(names)<=set(frame.columns),'Supply distinct existing columns')
    if numeric: require(all(pd.api.types.is_numeric_dtype(frame[x]) for x in names),'Numeric mapping required; clean explicitly first')

def integer(x,lo,hi): return type(x) is int and lo<=x<=hi
def number(x,lo,hi): return type(x) in (int,float) and math.isfinite(x) and lo<=x<=hi

def execute_method(f,method,p,figures):
    require(method in METHODS,'Method not supported by runner; use native module'); obj(p,METHODS[method],METHODS[method]); extra={}; comments=[]; tables={}
    if method in ('descriptive','correlation','normality','pca','clustering'):
        cols(f,p['columns']); require(len(p['columns'])<=100,'Max 100 columns per pipeline method')
    if method in ('ols','monte_carlo_ols','panel','prediction','regularised'):
        cols(f,[p['y'],*p['x_vars']]); require(isinstance(p['x_vars'],list) and 1<=len(p['x_vars'])<=30,'1–30 predictors per pipeline method')
    if 'seed' in p: require(integer(p['seed'],0,2**32-1),'Invalid seed')
    if method=='descriptive': tables={'Descriptive statistics':descriptive_statistics(f,p['columns'])}
    elif method=='correlation':
        require(p['method'] in ['pearson','spearman','kendall'] and len(p['columns'])>=2,'Invalid correlation specification'); a,b=correlation_matrix(f,p['columns'],p['method']); tables={'Correlation':a.reset_index(),'Correlation p-values':b.reset_index()}
    elif method=='normality': tables={'Normality':normality_tests(f,p['columns'])}
    elif method=='group_tests': cols(f,p['outcomes']); cols(f,[p['group']],False); tables={'Group tests':group_tests(f,p['outcomes'],p['group'])}
    elif method=='ols':
        require(p['covariance'] in ['nonrobust','HC0','HC1','HC2','HC3'],'Unsupported covariance for pipeline OLS')
        out=fit_detailed_model(f,p['y'],p['x_vars'],covariance=p['covariance']); tables={'Coefficients':out.coefficients,'Fit':out.fit,'Diagnostics':out.diagnostics,'Predictions':out.predictions}; comments=out.interpretation
        if figures:
            from visuals import ols_publication_bundle
            extra['figures.zip']=ols_publication_bundle(out.predictions,out.coefficients)
    elif method=='monte_carlo_ols':
        require(integer(p['simulations'],100,20000) and number(p['confidence'],.8,.999) and p['method'] in ['Wild bootstrap','Residual bootstrap','Parametric normal'],'Invalid Monte Carlo parameters; no clipping allowed'); a,b,c=monte_carlo_ols(f,**p); tables={'Monte Carlo summary':a,'Monte Carlo draws':b,'Monte Carlo fit':c}
        if figures:
            from visuals import monte_carlo_publication_bundle
            extra['figures.zip']=monte_carlo_publication_bundle(b,a,p['x_vars'][0])
    elif method=='pca':
        require(integer(p['n_components'],1,min(len(p['columns']),len(f))) and all(f[c].notna().any() for c in p['columns']),'Invalid PCA components or empty variables'); a,b=pca_table(f,**p); tables={'PCA loadings':a,'PCA variance':b}
    elif method=='arima':
        cols(f,[p['variable']]); cols(f,[p['time_col']],False); require(f[p['time_col']].is_unique and f[p['time_col']].notna().all() and f[p['variable']].notna().all(),'Verify unique complete time series'); t=f[p['time_col']].sort_values(); require((pd.api.types.is_numeric_dtype(t) and t.diff().dropna().nunique()==1) or (pd.api.types.is_datetime64_any_dtype(t) and len(t)>=3 and pd.infer_freq(t) is not None),'ARIMA requires numeric regular time keys or recognised regular datetime frequency'); require(isinstance(p['order'],list) and len(p['order'])==3 and all(integer(x,0,5) for x in p['order']) and integer(p['steps'],1,100),'Invalid ARIMA order/steps'); a,b=arima_forecast(f,p['variable'],p['time_col'],tuple(p['order']),p['steps']); tables={'ARIMA forecast':a,'ARIMA fit':b}
    elif method=='loan_forecast':
        cols(f,[p['date_col']],False); cols(f,[p['value_col']]); require(integer(p['steps'],1,24),'Invalid forecast steps'); o=loan_forecast(f,**p); tables={k:v for k,v in o.items() if isinstance(v,pd.DataFrame)}; comments=[o['caution']]
    elif method=='panel':
        from advanced_analytics import panel_model_suite
        cols(f,[p['entity'],p['time']],False); require(not f[[p['entity'],p['time']]].isna().any().any() and not f.duplicated([p['entity'],p['time']]).any() and not f[[p['y'],*p['x_vars']]].isna().any().any(),'Pipeline panel needs unique complete entity-time keys and complete model variables'); require(p['aggregation'] in ['Mean','Sum','Median'] and p['covariance'] in ['Clustered by entity','Robust','Unadjusted'],'Invalid panel parameters'); a,b,c,d,comments=panel_model_suite(f,**p); tables={'Panel fit':a,'Panel coefficients':b,'Hausman':c,'Prepared panel':d}
        if figures:
            from visuals import panel_publication_bundle
            extra['figures.zip']=panel_publication_bundle(b,a,c)
    elif method=='prediction':
        from advanced_analytics import predictive_model_comparison
        require(integer(p['folds'],2,10),'Invalid folds'); a,b,c,comments=predictive_model_comparison(f,**p); tables={'Predictive performance':a,'Predictions':b,'Importance':c}
        if figures:
            from visuals import predictive_publication_bundle
            extra['figures.zip']=predictive_publication_bundle(a,c,b)
    elif method=='funding_scenario':
        require(integer(p['draws'],100,50000),'Invalid scenario draws'); tables=funding_simulation(**p); comments=[NOTES[method]]
        if figures: extra['figures.zip']=scenario_figures(tables['Simulation draws'])
    elif method=='portfolio_mc':
        from analytics_core import monte_carlo_portfolio
        cols(f,[p['cost'],p['benefit']]); require(f[p['cost']].gt(0).all() and f[p['benefit']].ge(0).all(),'Resolve ineligible/missing cost and benefit rows first')
        if p['project_id'] is not None: cols(f,[p['project_id']],False); require(f[p['project_id']].notna().all() and f[p['project_id']].is_unique,'Project identifiers must be unique and complete')
        require(integer(p['simulations'],100,10000) and number(p['budget'],1e-12,1e18) and number(p['cost_cv'],0,2) and number(p['benefit_cv'],0,2) and number(p['correlation'],-.95,.95),'Invalid portfolio settings; no silent clipping'); a,b,c=monte_carlo_portfolio(f,**p); tables={'Portfolio summary':a,'Project selection':b,'Portfolio draws':c}; comments=[NOTES[method]]
    elif method=='iv_2sls':
        from analytics_core import instrumental_variables_2sls
        require(isinstance(p['instruments'],list) and p['instruments'] and isinstance(p['exogenous'],list),'Map excluded instruments and controls'); cols(f,[p['y'],p['endogenous'],*p['instruments'],*p['exogenous']]); a,b=instrumental_variables_2sls(f,**p); tables={'IV coefficients':a,'IV fit':b}; comments=[NOTES[method]]
    elif method=='did':
        from analytics_core import difference_in_differences
        cols(f,[p['y'],p['treatment'],p['post'],*p['controls']]); require(all(f[c].notna().all() and set(f[c].unique())=={0,1} for c in [p['treatment'],p['post']]),'DiD requires complete binary treatment/post variables'); require(len(f.groupby([p['treatment'],p['post']]))==4,'DiD requires all four cells'); require(not {'__treatment__','__post__','__did__'}.intersection(f.columns),'Reserved DiD column collision')
        if p['cluster'] is not None: cols(f,[p['cluster']],False); require(f[p['cluster']].notna().all() and f[p['cluster']].nunique()>1,'Cluster mapping needs complete multiple groups')
        o=difference_in_differences(f,**p); tables={'DiD coefficients':o.coefficients,'DiD fit':o.fit,'DiD diagnostics':o.diagnostics,'DiD predictions':o.predictions}; comments=o.interpretation
    elif method=='regularised':
        from analytics_core import regularised_regression
        require(p['method'] in ['Ridge','Lasso','Elastic Net'] and number(p['alpha'],1e-12,1e6) and number(p['l1_ratio'],0,1),'Invalid regularisation settings'); a,b=regularised_regression(f,**p); tables={'Regularised coefficients':a,'Regularised fit':b}; comments=[NOTES[method]]
    elif method=='clustering':
        from advanced_analytics import advanced_clustering
        require(p['method'] in ['K-means','Hierarchical agglomerative','Gaussian mixture','DBSCAN'] and p['scaling'] in ['Standard (z-score)','Robust (median/IQR)','Min-max [0,1]','None'] and type(p['automatic_k']) is bool and integer(p['clusters'],2,min(20,len(f)-1)) and integer(p['max_k'],2,min(20,len(f)-1)) and p['linkage'] in ['ward','complete','average','single'] and number(p['eps'],.0001,100) and integer(p['min_samples'],2,len(f)),'Invalid clustering settings; no silent clipping'); require(all(f[x].notna().sum()>=3 and f[x].nunique()>1 for x in p['columns']),'Resolve constant or sparse cluster variables first'); o=advanced_clustering(f,**p); tables={'Assignments':o.assignments,'Profiles':o.profiles,'Diagnostics':o.diagnostics,'Embedding':o.embedding}; comments=o.interpretation
        if figures:
            from visuals import clustering_publication_bundle
            extra['figures.zip']=clustering_publication_bundle(o.assignments,o.profiles,o.embedding,o.diagnostics)
    elif method=='greece_map':
        from mapping import aggregate_geography, fetch_geojson, match_nuts2, choropleth_figure, static_map_bytes
        cols(f,[p['geography']],False); cols(f,[p['metric']]); require(p['level'] in ['NUTS 2 – Regions (13)','NUTS 3 – Regional units'] and p['aggregation'] in ['Sum','Mean','Median','Count','Minimum','Maximum'],'Invalid geography settings')
        g=fetch_geojson(p['level']); valid={x['properties']['NUTS_ID'] for x in g['features']}; raw=f[p['geography']].astype('string').str.strip().str.upper(); codes=raw.where(raw.isin(valid),match_nuts2(f[p['geography']]) if p['level'].startswith('NUTS 2') else raw); require(codes.notna().all() and codes.isin(valid).all(),'Unmatched geography: correct mappings before plotting'); require(f[p['metric']].notna().all(),'Resolve missing map metric explicitly; no zero imputation')
        d=f.copy(); d[p['geography']]=codes; m=aggregate_geography(d, p['geography'],p['metric'],p['aggregation'],p['level']); tables={'Mapped values':m}; comments=['Aggregation is descriptive; no ecological or causal inference.']
        if figures:
            for mono in (False,True):
                stem='grayscale' if mono else 'colour'; extra[f'map_{stem}.html']=choropleth_figure(m,g,p['metric'],mono).to_html(full_html=True,include_plotlyjs=True).encode()
                for fmt in ('png','svg','pdf'): extra[f'map_{stem}.{fmt}']=static_map_bytes(m,g,p['metric'],mono,fmt,600)
    require(tables and any(not t.empty for t in tables.values()),'Native routine returned no usable results')
    if figures and not extra: comments.append('No native publication-figure route registered for this method; use its original module, not an invented plot.')
    return tables,extra,comments

def table_write(root,s,base,method,tables,rqs=()):
    for i,(name,t) in enumerate(tables.items()):
        require(isinstance(t,pd.DataFrame),'Native output is not a table'); key=f'{method}.{i+1:02d}'
        path=f'{base}/{key}.json'; raw=t.to_json(orient='split',date_format='iso',double_precision=15,default_handler=str).encode(); add_file(root,s,path,raw,'table')
        # CSV is an exact data export, not a formula-safe spreadsheet view.
        add_file(root,s,f'{base}/{key}.csv',t.to_csv(index=False).encode(),'table_csv')
        s['tables'][key]={'name':name,'path':path,'rows':len(t),'columns':[str(c) for c in t.columns],'research_questions':list(rqs)}

def table_read(root,s,key):
    require(key in s['tables'],'Unknown evidence table '+str(key)); return load(root/s['tables'][key]['path'])
def tables_all(root,s):
    return {k+' — '+v['name']:pd.DataFrame(t['data'],columns=t['columns'],index=t['index']) for k,v in s['tables'].items() for t in [table_read(root,s,k)]}
def config_at(root,n): return load(root/f'{n:02d}_approved.json')
def design(root,s,c):
    obj(c,['research_questions','hypotheses','source_notes','gap','literature_limitations'],['research_questions','hypotheses','source_notes','gap','literature_limitations']); require(nonempty(c['gap']) and nonempty(c['literature_limitations']),'Record tentative gap and literature limitations')
    rqs=c['research_questions']; hs=c['hypotheses']; require(isinstance(rqs,list) and 1<=len(rqs)<=40 and isinstance(hs,list) and len(hs)<=80,'Invalid RQ/hypothesis count'); ids=[]; hids=[]
    for r in rqs:
        obj(r,['id','group','question','type'],['id','group','question','type']); require(all(nonempty(r[k]) for k in r) and r['type'] in ['descriptive','associational','predictive','causal','scenario','qualitative','robustness'],'Invalid RQ'); ids.append(r['id'])
    require(len(ids)==len(set(ids)),'Duplicate RQ ids')
    for h in hs:
        obj(h,['id','rq_id','statement','decision_rule'],['id','rq_id','statement','decision_rule']); require(all(nonempty(h[k]) for k in h) and h['rq_id'] in ids,'Invalid hypothesis'); hids.append(h['id'])
    require(len(hids)==len(set(hids)),'Duplicate hypothesis ids')
    docs=[(x['name'],(root/x['path']).read_bytes()) for x in s['inputs'] if x['kind']=='document']; evidence=extract_pdf_collection(tuple(docs)); source_notes=c['source_notes']; require(isinstance(source_notes,list),'Source notes must be a list')
    for n in source_notes:
        obj(n,['document','page_or_block','excerpt','interpretation'],['document','page_or_block','excerpt','interpretation']); require(nonempty(n['excerpt']) and nonempty(n['interpretation']) and integer(n['page_or_block'],1,1000000),'Invalid source note'); match=evidence[(evidence.document==n['document'])&(evidence.page==n['page_or_block'])]; require(len(match)==1 and n['excerpt'] in match.iloc[0].text,'Source note excerpt not found at stated locator')
    return evidence

def plan_check(c,rqs):
    obj(c,['analyses','multiplicity','scope_note'],['analyses','multiplicity','scope_note']); require(nonempty(c['scope_note']),'Record analysis scope/limitations'); require(c['multiplicity'] in ['Benjamini–Hochberg','Bonferroni','not_applicable','separate_predeclared_families'],'Specify multiplicity policy'); aa=c['analyses']; require(isinstance(aa,list) and 1<=len(aa)<=20,'Plan 1–20 analyses'); ids=[]
    for a in aa:
        obj(a,['id','method','params','rq_ids','reason','assumptions','figures','source'],['id','method','params','rq_ids','reason','assumptions','figures']); require(isinstance(a['id'],str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}',a['id']),'Invalid analysis id'); ids.append(a['id']); require(a['id'].upper() not in ('AUDIT_VALIDATION','MULTIPLICITY') and not a['id'].lower().endswith('_validation'),'Reserved analysis id'); require(a['method'] in METHODS,'Unsupported analysis'); obj(a['params'],METHODS[a['method']],METHODS[a['method']]); require(isinstance(a['rq_ids'],list) and a['rq_ids'] and set(a['rq_ids'])<=rqs,'Map analysis to approved RQs'); require(nonempty(a['reason']) and nonempty(a['assumptions']) and type(a['figures']) is bool,'Explain method and assumptions; specify figures')
    require(len(ids)==len(set(ids)),'Duplicate analysis ids')

def manuscript_check(root,s,c):
    obj(c,['markdown','rq_answers','hypothesis_assessments','numeric_claims','bibliography','originality_note'],['markdown','rq_answers','hypothesis_assessments','numeric_claims','bibliography','originality_note']); require(nonempty(c['markdown']) and len(c['markdown'])<=500000 and nonempty(c['originality_note']),'Supply manuscript and qualified originality assessment'); d=config_at(root,2); rids={x['id'] for x in d['research_questions']}; hids={x['id'] for x in d['hypotheses']}; tbl=s['tables']
    def refs(x,rq):
        require(isinstance(x,list) and all(k in tbl and rq in tbl[k]['research_questions'] for k in x),'Evidence not mapped to this RQ')
    aa=c['rq_answers']; hh=c['hypothesis_assessments']; require(isinstance(aa,list) and isinstance(hh,list),'Provide registers'); require(len(aa)==len(rids) and {x['id'] for x in aa}==rids,'Answer every approved RQ exactly once'); require(len(hh)==len(hids) and {x['id'] for x in hh}==hids,'Assess every hypothesis exactly once'); hm={x['id']:x['rq_id'] for x in d['hypotheses']}
    for a in aa:
        obj(a,['id','status','answer','evidence_ids'],['id','status','answer','evidence_ids']); require(a['status'] in ['answered','partly_answered','not_answerable'] and nonempty(a['answer']),'Invalid answer'); refs(a['evidence_ids'],a['id']); require(a['status']=='not_answerable' or a['evidence_ids'],'Answered RQ requires computed evidence')
    for h in hh:
        obj(h,['id','verdict','reason','evidence_ids'],['id','verdict','reason','evidence_ids']); require(h['verdict'] in ['reject_null','fail_to_reject_null','consistent_with','inconsistent_with','inconclusive','not_tested'] and nonempty(h['reason']),'No proof/automatic acceptance of hypotheses'); refs(h['evidence_ids'],hm[h['id']]); require(h['verdict'] in ['inconclusive','not_tested'] or h['evidence_ids'],'Verdict requires evidence')
    claims=c['numeric_claims']; require(isinstance(claims,list) and len(claims)<=5000,'Invalid numeric claim register'); checked=[]; cache={}
    for q in claims:
        obj(q,['table_id','row','column','value','decimals','text'],['table_id','row','column','value','decimals','text']); key=q['table_id']; require(key in tbl and tbl[key]['research_questions'],'Numerical claim must reference an analysis table'); t=cache[key] if key in cache else table_read(root,s,key); cache[key]=t; require(integer(q['row'],0,len(t['data'])-1) and q['column'] in t['columns'] and integer(q['decimals'],0,12) and nonempty(q['text']),'Invalid numeric locator'); v=t['data'][q['row']][t['columns'].index(q['column'])]; require(type(v) in (int,float) and type(q['value']) in (int,float) and math.isfinite(v) and math.isfinite(q['value']) and abs(round(v,q['decimals'])-q['value'])<=max(1e-12,abs(v)*1e-14),'Claim does not match native value at stated rounding: '+key); checked.append(q|{'native_value':v})
    require(isinstance(c['bibliography'],list) and all(isinstance(x,dict) for x in c['bibliography']),'Bibliography must be CSL/Zotero JSON list; metadata is not automatically verified')
    return checked

def approve(root,gate,config,reviewer):
    root=Path(root); require(nonempty(reviewer) and len(reviewer)<=200,'Identify approving reviewer'); require(isinstance(config,dict) and len(jb(config))<=4*1024*1024,'Invalid or oversized config')
    with lock(root):
        s=state_read(root); require(gate==s['next_gate'] and 1<=gate<=6,'Gate out of order'); require(s['pending'] is None,'A decision is already pending; revoke before replacing')
        # Never accept a changed execution plan without a new recorded approval.
        s['pending']={'gate':gate,'config':config,'config_sha256':digest(jb(config)),'reviewer':reviewer,'approved_utc':now(),'approval_id':uuid.uuid4().hex}; s['status']='approved_not_executed'; state_write(root,s); return s

def revoke(root):
    root=Path(root)
    with lock(root):
        s=state_read(root); require(s['pending'] is not None,'No pending approval'); s['history'].append({'status':'revoked','approval_id':s['pending']['approval_id'],'at':now()}); s['pending']=None; s['status']='awaiting_approval'; state_write(root,s); return s

def advance(root):
    root=Path(root)
    with lock(root):
        s=state_read(root); p=s['pending']; gate=s['next_gate']; require(p is not None and p['gate']==gate and digest(jb(p['config']))==p['config_sha256'],'Explicit approval required before execution'); c=p['config']; before=len(s['artifacts']); base=f'{gate:02d}_'+['preflight','data','design','execution','manuscript','review','release'][gate]+(f'_r{s["revision"]}' if gate>=4 else '')
        try:
            if gate==1:
                frames,f,changes,audits,act=prepare(root,s,c); add_file(root,s,base+'/cleaned_audits.json',jb(audits)); add_file(root,s,base+'/active_audit.json',jb(act)); add_file(root,s,base+'/cleaning_log.json',jb({n:json.loads(t.to_json(orient='records')) for n,t in changes.items()})); add_file(root,s,base+'/active.csv',f.to_csv(index=False).encode(),'private_data');
                try: add_file(root,s,base+'/active.parquet',f.to_parquet(index=False),'private_data')
                except ImportError: add_file(root,s,base+'/PARQUET_NOT_AVAILABLE.txt',b'PyArrow not available. CSV plus original inputs and deterministic cleaning settings retained.')
                
                for idx,(label,cleaned) in enumerate(frames.items(),1):
                    stem=f'{base}/cleaned/source_{idx:02d}'; add_file(root,s,stem+'.csv',cleaned.to_csv(index=False).encode(),'private_data'); add_file(root,s,stem+'_dictionary.csv',data_dictionary(cleaned).to_csv(index=False).encode(),'private_data');
                    try: add_file(root,s,stem+'.parquet',cleaned.to_parquet(index=False),'private_data')
                    except ImportError: pass
                add_file(root,s,base+'/cleaned_source_index.json',jb([{'source':label,'path':f'{base}/cleaned/source_{idx:02d}.csv','sha256':frame_hash(cleaned)} for idx,(label,cleaned) in enumerate(frames.items(),1)])); s['active_sha256']=frame_hash(f); s['active_columns']=[str(x) for x in f.columns]; docs=[(x['name'],(root/x['path']).read_bytes()) for x in s['inputs'] if x['kind']=='document']; ev=extract_pdf_collection(tuple(docs)); add_file(root,s,base+'/literature_preview.json',ev.to_json(orient='records',force_ascii=False).encode(),'private_document_index')
            elif gate==2:
                evidence=design(root,s,c); add_file(root,s,base+'/literature_index.json',evidence.to_json(orient='records',force_ascii=False).encode(),'private_document_index'); add_file(root,s,base+'/research_design.json',jb(c)); add_file(root,s,base+'/recommended_routes.json',jb(recommend(s['question'])))
            elif gate==3:
                frames,f,_,_,_=prepare(root,s,config_at(root,1)); require(frame_hash(f)==s['active_sha256'],'Prepared frame changed'); d=config_at(root,2); rqs={r['id'] for r in d['research_questions']}; plan_check(c,rqs); all_tables={}; psets=[]
                for a in c['analyses']:
                    src=a.get('source','active'); require(src=='active' or src in frames,'Unknown analysis source'); analysis_frame=f if src=='active' else frames[src]; au=audit_all({'analysis':analysis_frame})['analysis']; require(au['rows'] and au['columns'] and not au['duplicate_headers'] and not any(x['infinite'] for x in au['fields']),'Selected analysis source failed mandatory audit'); t,extra,comments=execute_method(analysis_frame,a['method'],a['params'],a['figures']); table_write(root,s,base,a['id'],t,a['rq_ids']); all_tables.update({a['id']+' '+n:x for n,x in t.items()}); add_file(root,s,base+'/'+a['id']+'_interpretation.json',jb({'method':a['method'],'parameters':a['params'],'comments':comments,'source':src,'source_sha256':frame_hash(analysis_frame),'assumptions':a['assumptions'],'reason':a['reason']}));
                    for n,b in extra.items(): add_file(root,s,base+'/'+a['id']+'_'+n,b,'native_figure')
                    if a['method']=='ols':
                        q=t['Coefficients']; psets.extend({'analysis_id':a['id'],'term':r.term,'p_value':float(r.p_value)} for r in q.itertuples() if r.term!='const' and pd.notna(r.p_value)); v=validation({},analysis_frame,a['params']['y'],a['params']['x_vars']); require(not (v['Validation register'].status=='blocked').any(),'OLS validation blocked: inspect specification'); table_write(root,s,base,a['id']+'_validation',v,a['rq_ids']); all_tables.update({a['id']+' validation '+n:x for n,x in v.items()})
                if psets and c['multiplicity'] in ['Benjamini–Hochberg','Bonferroni']:
                    m=pd.DataFrame(psets); m['adjusted_p']=p_adjust(m.p_value,c['multiplicity']); table_write(root,s,base,'MULTIPLICITY',{'Joint OLS slope family':m},list(rqs)); all_tables['Joint OLS slope family']=m
                v=validation(frames); table_write(root,s,base,'AUDIT_VALIDATION',v); all_tables.update(v)
                require(sum(t.size for t in all_tables.values())<=2_000_000,'Publication table cell cap exceeded; split the plan into smaller runs'); add_file(root,s,base+'/output_guide.md',output_guide_markdown(build_output_guide(all_tables)).encode()); add_file(root,s,base+'/native_tables.zip',publication_pack(all_tables,question=s['question']),'native_publication')
                s['execution_sha256']=digest(jb({k:s['tables'][k] for k in sorted(s['tables'])}))
            elif gate==4:
                checked=manuscript_check(root,s,c); add_file(root,s,base+'/manuscript.md',c['markdown'].encode()); add_file(root,s,base+'/manuscript.docx',docx_bytes(c['markdown'])); add_file(root,s,base+'/claim_register.json',jb(checked)); add_file(root,s,base+'/rq_hypothesis_register.json',jb({'rq_answers':c['rq_answers'],'hypotheses':c['hypothesis_assessments'],'originality_note':c['originality_note']})); s['manuscript_path']=base+'/manuscript.md'; s['manuscript_config']=base+'/manuscript_config.json'; add_file(root,s,s['manuscript_config'],jb(c),'private_manuscript_config')
            elif gate==5:
                obj(c,['review_scope'],['review_scope']); require(nonempty(c['review_scope']),'Approve scope of structural editorial review'); review=peer_review((root/s['manuscript_path']).read_text(),tables_all(root,s)); add_file(root,s,base+'/editorial_review.csv',review.to_csv(index=False).encode()); add_file(root,s,base+'/editorial_review.json',review.to_json(orient='records').encode()); s['review_path']=base+'/editorial_review.json'; s['review_count']=len(review)
            elif gate==6:
                obj(c,['issues','human_review_note','reference_review','data_sharing_note','shareable_paths'],['issues','human_review_note','reference_review','data_sharing_note']); require(all(nonempty(c[k]) for k in ['human_review_note','reference_review','data_sharing_note']),'Human review, references and sharing decisions required'); issues=c['issues']; require(isinstance(issues,list) and len(issues)==s['review_count'] and {x['row'] for x in issues}==set(range(s['review_count'])),'Address every reviewer row');
                for x in issues:
                    obj(x,['row','resolution','status'],['row','resolution','status']); require(x['status'] in ['addressed','accepted_limitation'] and nonempty(x['resolution']),'Open or undocumented editorial issue; revise manuscript first')
                mc=load(root/s['manuscript_config']); all_tables=tables_all(root,s); add_file(root,s,base+'/publisher_evidence.zip',publication_pack(all_tables,question=s['question'],bibliography=mc['bibliography']),'private_review_required'); add_file(root,s,base+'/release_review.json',jb(c)); s['release_status']='candidate_package_human_review_required'; add_file(root,s,base+'/native_code.zip',native_code_zip(),'native_code'); paths=c.get('shareable_paths',[]); require(isinstance(paths,list) and len(paths)==len(set(paths)),'Select distinct shareable artifact paths');
                if paths: add_file(root,s,base+'/SELECTED_FOR_PUBLISHER.zip',selected_package(root,s,paths),'selected_candidate')
            # Approved settings are part of the auditable run. Revision-specific manuscript files retained.
            config_path=f'{gate:02d}_approved.json' if gate<4 else base+'/approved.json'; add_file(root,s,config_path,jb(c),'approved_configuration'); s['approvals'].append({k:v for k,v in p.items() if k!='config'}|{'config_path':config_path}); s['history'].append({'gate':gate,'at':now(),'status':'completed','approval_id':p['approval_id']}); s['next_gate']=gate+1; s['pending']=None; s['status']='candidate_complete' if gate==6 else 'awaiting_approval'; state_write(root,s); return s
        except Exception as exc:
            # Uncommitted files are not evidence. State remains at the approved gate; revoke/revise configuration after failure.
            for x in s['artifacts'][before:]: (root/x['path']).unlink(missing_ok=True)
            original=load(root/'run_state.json'); original['status']='blocked'; original['history'].append({'gate':gate,'at':now(),'status':'blocked','error':f'{type(exc).__name__}: {exc}'}); state_write(root,original); raise

def revise(root,reason,reviewer):
    root=Path(root); require(nonempty(reason) and nonempty(reviewer),'Document revision reason and reviewer')
    with lock(root):
        s=state_read(root); require(s['next_gate'] in [5,6,7] and s['pending'] is None,'Revision is available after manuscript gate; revoke pending approval first'); s['revision']+=1; s['next_gate']=4; s['status']='awaiting_approval'; s.pop('release_status',None); s['history'].append({'status':'revision','at':now(),'reason':reason,'reviewer':reviewer,'revision':s['revision']}); state_write(root,s); return s

def native_code_zip():
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted([*ROOT.glob('*.py'),*ROOT.glob('requirements*.txt'),ROOT/'VERSION.txt',*ROOT.glob('data/*.geojson'),*ROOT.glob('data/*.csv')]): z.write(p,str(p.relative_to(ROOT)))
        z.writestr('PIPELINE_CODE_SHA256.txt',code_fingerprint())
    return b.getvalue()

def selected_package(root,s,paths):
    # Selection is explicit, not an automated confidentiality guarantee.
    allowed={x['path']:x for x in s['artifacts'] if x['kind'] not in ('private_data','private_document_index','private_manuscript_config','approved_configuration','selected_candidate') and not x['path'].startswith(('00_preflight/','01_data/','02_design/')) and (not x['path'].startswith(('04_manuscript','05_review','06_release')) or f'_r{s["revision"]}/' in x['path'])}
    require(set(paths)<=set(allowed),'Selection contains private or unknown artifact; never share raw source files here')
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths: z.write(root/p,p)
        z.writestr('manifest.json',jb({'run_id':s['run_id'],'application_version':s['application_version'],'code_sha256':s['code_sha256'],'git_commit':s.get('git_commit'),'selected':[allowed[p] for p in paths],'human_review_required':True}))
        z.writestr('README.txt','Researcher-selected candidate. Verify confidentiality, publisher requirements, citations and every claim before sharing. No automatic anonymisation or acceptance guarantee.')
    return b.getvalue()

def inspect_stage(root,kind,offset=0,limit=50):
    root=Path(root); s=state_read(root); require(integer(offset,0,10**9) and integer(limit,1,200),'Invalid evidence page')
    paths={'audit':'00_preflight/audits.json','prepared':'01_data/active_audit.json','literature':'01_data/literature_preview.json','review':s.get('review_path')}
    require(kind in paths and paths[kind] and any(a['path']==paths[kind] for a in s['artifacts']),'Requested evidence is not yet available')
    data=load(root/paths[kind]);
    if isinstance(data,list):
        page=data[offset:offset+limit]
        if kind=='literature': page=[r|{'text':r['text'][:10000],'text_truncated':len(r['text'])>10000} for r in page]
        return {'total':len(data),'offset':offset,'rows':page}
    return data

def public_status(root):
    s=state_read(Path(root)); return {k:s[k] for k in ['run_id','application_version','code_sha256','git_commit','next_gate','status','sources','tables','human_review_required','revision'] }|{'pending_approval':None if not s['pending'] else {k:v for k,v in s['pending'].items() if k!='config'},'active_columns':s.get('active_columns',[]),'artifacts':s['artifacts'],'question':s['question']}


def checkpoint(root):
    root=Path(root)
    with lock(root):
        s=state_read(root); b=io.BytesIO()
        with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('run_state.json',jb(s));
            for x in s['inputs']+s['artifacts']: z.write(root/x['path'],x['path'])
            z.writestr('PRIVATE_README.txt','PRIVATE research checkpoint. Contains source data, documents and result tables. Do not publish without permissions and disclosure review. Human review remains required. No credentials should be supplied in configuration. Original source code is the pinned workbench build, not generated statistical code.\n')
        return b.getvalue()

def restore(root,raw):
    """Restore only your own trusted private checkpoint; checksums detect corruption, not forgery."""
    root=Path(root); require(not root.exists() or not any(root.iterdir()),'Restore into empty directory'); require(len(raw)<=MAX_TOTAL*3,'Archive too large')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        infos=z.infolist(); names=[x.filename for x in infos]; require(len(names)==len(set(names)) and len(names)<=5000 and sum(x.file_size for x in infos)<=MAX_TOTAL*5,'Unsafe archive limits'); s=json.loads(z.read('run_state.json')); require(s['schema']==SCHEMA,'Wrong checkpoint schema'); wanted={'run_state.json'}|{x['path'] for x in s['inputs']+s['artifacts']}; require(wanted<=set(names),'Missing checkpoint files'); require(set(names)<=wanted|{'PRIVATE_README.txt'},'Unexpected checkpoint entries')
        root.mkdir(parents=True,exist_ok=True,mode=0o700)
        for n in wanted:
            p=PurePosixPath(n); require(not p.is_absolute() and '..' not in p.parts and '\\' not in n and len(p.parts)<=4,'Unsafe checkpoint path'); write_atomic(root/p,z.read(n))
    return state_read(root)

def status(root): return state_read(Path(root))
def evidence(root,key,offset=0,limit=100):
    root=Path(root); s=state_read(root); require(integer(offset,0,10**9) and integer(limit,1,200),'Invalid evidence page'); t=table_read(root,s,key); return {'table_id':key,'total_rows':len(t['data']),'offset':offset,'columns':t['columns'],'rows':t['data'][offset:offset+limit]}

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('schema'); a=sub.add_parser('init'); a.add_argument('workspace'); a.add_argument('--question',required=True); a.add_argument('files',nargs='+');
    for n in ('status','advance','revoke','checkpoint','restore','approve','revise','evidence'):
        a=sub.add_parser(n); a.add_argument('workspace')
        if n in ('checkpoint','restore'): a.add_argument('archive')
        if n=='approve': a.add_argument('gate',type=int); a.add_argument('config'); a.add_argument('--reviewer',required=True)
        if n=='revise': a.add_argument('--reason',required=True); a.add_argument('--reviewer',required=True)
        if n=='evidence': a.add_argument('table_id'); a.add_argument('--offset',type=int,default=0); a.add_argument('--limit',type=int,default=100)
    a=p.parse_args(); cmd=a.cmd; r=Path(a.workspace) if cmd!='schema' else None
    if cmd=='schema': out=schema()
    elif cmd=='init': out=new_run(r,[(Path(f).name,Path(f).read_bytes()) for f in a.files],a.question)
    elif cmd=='approve': out=approve(r,a.gate,load(Path(a.config)),a.reviewer)
    elif cmd=='advance': out=advance(r)
    elif cmd=='revoke': out=revoke(r)
    elif cmd=='revise': out=revise(r,a.reason,a.reviewer)
    elif cmd=='checkpoint': Path(a.archive).write_bytes(checkpoint(r)); out={'archive':a.archive,'private':True}
    elif cmd=='restore': out=restore(r,Path(a.archive).read_bytes())
    elif cmd=='evidence': out=evidence(r,a.table_id,a.offset,a.limit)
    else: out=status(r)
    print(jb(out).decode())
if __name__=='__main__': main()
