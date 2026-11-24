"""Dataset-neutral research utilities. No network calls unless explicitly requested."""
from __future__ import annotations
import io, json, hashlib, re, zipfile, unicodedata
from datetime import datetime, timezone
import numpy as np
import pandas as pd

METHODS = [
 ('Data & research design','2. Data hub & audit','Audit missingness, identifiers, duplicates and types before choosing an estimator. Verify units and join keys.'),
 ('Data & research design','3. Research questions','Map hypotheses to measurable variables; distinguish description, prediction and identification.'),
 ('Data & research design','14. Data cleaning & validation','Preview transformations; preserve identifiers, original rows and a change log.'),
 ('Descriptive & inference','4. Descriptive statistics','Summarise distributions, missingness and associations. Correlation is not causation.'),
 ('Descriptive & inference','5. Hypothesis tests','Compare groups or categorical association; check independence, distribution and multiplicity.'),
 ('Econometrics & forecasting','6. OLS & econometric laboratory','Use OLS for continuous outcomes, GLMs for binary/count outcomes, IV or DiD only with identification assumptions.'),
 ('Econometrics & forecasting','6A. Monte Carlo & uncertainty','Assess sampling or portfolio uncertainty; record seeds and distribution assumptions.'),
 ('Econometrics & forecasting','7. 1,000 × 1,000 batch engine','Screen many relationships; apply multiplicity controls and independent confirmation.'),
 ('Econometrics & forecasting','8. Project-to-region-year panel','Aggregate compatible uploaded project records to regional year panels; use generic panel modelling for other schemas.'),
 ('Econometrics & forecasting','8A. Panel model laboratory','Repeated entities over time; verify unique entity-time keys, variation and covariance assumptions.'),
 ('Econometrics & forecasting','10. Time series & multivariate','Regular time series for ARIMA/Granger; numeric variables for PCA and scale reliability.'),
 ('Econometrics & forecasting','10B. Predictive model laboratory','Estimate predictive performance out of sample; avoid target leakage; use temporal validation for temporal data.'),
 ('Econometrics & forecasting','15. Public loan forecasting','Historical aggregate loan flows or stocks, regular frequency and homogeneous units; chronological backtesting and prediction intervals.'),
 ('Spatial & segmentation','9. Detailed Greece GIS','Use validated geographic identifiers and matching boundary vintage; avoid ecological inference.'),
 ('Spatial & segmentation','10A. Advanced clustering & segmentation','Scale variables; validate cluster stability and usefulness; clusters are not causal types.'),
 ('Policy & optimisation','12. Scenario & allocation engine','Explicit objective, constraints and scenario assumptions; distinguish sensitivity from forecast.'),
 ('Policy & optimisation','12A. Dedicated MCDA engine','Compare alternatives under defensible weights, criterion direction and sensitivity analysis.'),
 ('Policy & optimisation','12A.1 ITA / public-funding decision support','Eligibility, call ceilings, policy priorities and uncertainty; allocation is advisory.'),
 ('Policy & optimisation','12A.1B GAMS-compatible ITA Studio','Replicate and optimise explicitly mapped GAMS structures with solver diagnostics.'),
 ('Policy & optimisation','12A.2 Expert respondent analytics','Respondent-level preference data; assess agreement and heterogeneity before transferring weights.'),
 ('Policy & optimisation','12C. Frontier methods laboratory','Causal AIPW requires overlap and no unmeasured confounding; Bayesian methods require justified priors; Pareto/XAI have separate aims.'),
 ('Research & publication','11. Publication figures & HTML report','Export underlying data, captions, colour and monochrome figures; distinguish diagnostic and publication outputs.'),
 ('Research & publication','12B. Research Command Chair','Ask precise questions with roles, hypotheses, sources and scope; verify claims against computed evidence.'),
 ('Research & publication','12D. Agentic Research Mode','Approval-gated research plan, bounded execution and reproducible drafting.'),
 ('Research & publication','16. Conference & CFP discovery','Live web search with optional provider key; verify dates, publisher, organiser and official call.'),
 ('Research & publication','17. Zotero bibliography','Import CSL JSON or fetch Zotero items; inspect duplicates and missing metadata; verify citations.'),
 ('Research & publication','13. Methods & reproducibility','Retain exact data hash, transformations, code version, settings and diagnostics.'),
]

# Additive v6.1 routes; keep method catalogue available without Streamlit imports.
METHODS.extend([
 ('Data & research design','18. Scientific validation centre','Audit all sources and explicitly mapped OLS diagnostics; distinguish checks from untestable assumptions.'),
 ('Data & research design','19. Research project manager','Save and restore portable data and archived evidence without keys.'),
 ('Research & publication','20. Academic peer reviewer','Structural review and optional evidence-grounded simulated AI critique.'),
 ('Research & publication','21. Journal intelligence centre','Topic-based journal search; verify current fees, submission rules and indexing.'),
 ('Policy & optimisation','22. Public finance simulator','Simulate funding scenarios with explicit budget and absorption assumptions.'),
 ('Research & publication','23. Publication export centre','Consolidate actual tables, reports and attached publication assets.'),
 ('Research & publication','24. LLM connector setup','Configure isolated local or authenticated separately hosted MCP service.')])


def frame_hash(frame):
    schema=json.dumps([(str(c),str(d)) for c,d in zip(frame.columns,frame.dtypes)],ensure_ascii=False)
    return hashlib.sha256((schema+'\n'+frame.to_csv(index=False)).encode('utf-8')).hexdigest()

def audit(frame):
    if not isinstance(frame,pd.DataFrame): raise TypeError('Expected a table')
    duplicates = int(frame.duplicated().sum())
    columns = []
    for i,c in enumerate(frame.columns):
        s=frame.iloc[:,i]; numeric=pd.api.types.is_numeric_dtype(s)
        columns.append({'column':str(c),'dtype':str(s.dtype),'missing':int(s.isna().sum()),'unique':int(s.nunique(dropna=True)), 'infinite':int(np.isinf(pd.to_numeric(s,errors="coerce").to_numpy(dtype=float,na_value=np.nan)).sum()) if numeric and not pd.api.types.is_complex_dtype(s) else 0})
    return {'rows':len(frame),'columns':frame.shape[1],'duplicate_rows':duplicates,'duplicate_headers':int(frame.columns.duplicated().sum()),'sha256':frame_hash(frame),'fields':columns}

def clean(frame, *, trim=True, normalise_headers=True, remove_duplicates=False, missing_tokens=(), numeric_columns=(), decimal_comma=False):
    """Explicit reversible transformations; no imputation, winsorisation or implicit coercion."""
    result=frame.copy(deep=True); changes=[]
    if normalise_headers:
        used=set(); names=[]
        for c in result.columns:
            base=re.sub(r'\W+','_',unicodedata.normalize('NFKC',str(c)).strip(),flags=re.UNICODE).strip('_') or 'column'; name=base; k=2
            while name in used: name=f'{base}_{k}'; k+=1
            used.add(name); names.append(name)
            if name!=str(c): changes.append({'operation':'rename','column':str(c),'detail':name})
        result.columns=names
    if result.columns.duplicated().any(): raise ValueError('Resolve duplicate column names first')
    for c in result.columns:
        s=result[c]
        if trim and (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            t=s.map(lambda x:x.strip() if isinstance(x,str) else x)
            count=int((s.notna() & t.notna() & s.ne(t)).sum()); result[c]=t
            if count: changes.append({'operation':'trim','column':str(c),'detail':count})
        if missing_tokens:
            mask=result[c].map(lambda x:isinstance(x,str) and x in missing_tokens)
            if mask.any(): result.loc[mask,c]=None; changes.append({'operation':'missing token','column':str(c),'detail':int(mask.sum())})
    for c in numeric_columns:
        if c not in result: raise ValueError(f'Unknown numeric column: {c}')
        s=result[c]; t=s.astype('string').str.replace(',','.',regex=False) if decimal_comma else s
        converted=pd.to_numeric(t,errors='coerce'); bad=s.notna() & converted.isna()
        if bad.any(): raise ValueError(f'{c}: {int(bad.sum())} invalid values; no numeric conversion applied')
        result[c]=converted; changes.append({'operation':'numeric conversion','column':str(c),'detail':int(s.notna().sum())})
    if remove_duplicates:
        n=len(result); result=result.drop_duplicates(); changes.append({'operation':'remove exact duplicates','column':'*','detail':n-len(result)})
    return result.reset_index(drop=True),pd.DataFrame(changes,columns=['operation','column','detail'])

def audit_all(frames):
    return {str(name):audit(frame) for name,frame in frames.items()}

def recommend(question):
    q=question.casefold(); matches=[]
    rules=[(("validat","diagnostic","εγκυρό","εγκυρο"),"18."),(("project archive","restore project","αποθήκευση"),"19."),(("peer review","reviewer","κριτής"),"20."),(("journal","περιοδικό","περιοδικο"),"21."),(("scenario","simulation","σενάριο","σεναριο"),"22."),(("export","εξαγωγ"),"23."),(("connector","mcp"),"24."),(('loan','δάνει','δανει','repayment','disbursement'), '15.'),(('forecast','arima','πρόβλε','προβλε','time series'),'10.'),(('panel','fixed effect','διαχρον'),'8A.'),(('causal','difference-in','αιτι','instrumental'),'6.'),(('cluster','segmentation','ομαδοπ'),'10A.'),(('funding','ita','allocation','χρηματοδ','κατανομ'),'12A.1 '),(('mcda','ranking','κριτήρι','κριτηρι'),'12A.'),(('conference','cfp','συνέδ','συνεδ'),'16.'),(('zotero','bibliograph','βιβλιογρ'),'17.'),(('regression','ols','παλινδρ'),'6.'),(('predict','machine learning','προγνωσ'),'10B.'),(('hypothesis','test','υπόθε','υποθε'),'5.'),(('paper','manuscript','article','δημοσίευ','δημοσιευ'),'12B.')]
    for words,prefix in rules:
        if any(w in q for w in words): matches.extend(x for x in METHODS if x[1].startswith(prefix))
    return list(dict.fromkeys(matches)) or [x for x in METHODS if x[1].startswith(('2.','3.','12B.'))]

def loan_forecast(frame,date_col,value_col,frequency,steps,aggregation='sum'):
    """Aggregate public-loan time series, compare naive and linear trend by rolling origin.
    Intervals are empirical one-step residual bands, not calibrated multi-step guarantees.
    """
    if frequency not in ('M','Q','Y') or not 1<=steps<=24 or aggregation not in ('sum','last'): raise ValueError('Invalid forecast settings')
    dates=pd.to_datetime(frame[date_col].astype(str),errors='coerce'); values=pd.to_numeric(frame[value_col],errors='coerce')
    if dates.isna().any() or values.isna().any() or not np.isfinite(values).all(): raise ValueError('Invalid dates or values; clean and verify the data first')
    t=pd.DataFrame({'date':dates,'value':values}).sort_values('date'); t['period']=t.date.dt.to_period(frequency)
    if aggregation=='last' and t.date.duplicated().any(): raise ValueError('Stock data require unique dates or prior aggregation by entity')
    g=t.groupby('period').value; y=g.sum() if aggregation=='sum' else g.last()
    if len(y)<10: raise ValueError('At least 10 observed periods are needed for this baseline')
    if not y.index.equals(pd.period_range(y.index.min(),y.index.max(),freq=frequency)): raise ValueError('Missing periods: verify zero vs unreported observations before forecasting')
    v=y.to_numpy(dtype=float); errors={'naive':[],'linear_trend':[]}; rows=[]
    for i in range(max(6,len(v)//2),len(v)):
        pred={'naive':v[i-1],'linear_trend':np.polyval(np.polyfit(np.arange(i),v[:i],1),i)}
        for m,p in pred.items(): errors[m].append(float(v[i]-p)); rows.append({'period':str(y.index[i]),'model':m,'actual':v[i],'prediction':p,'error':v[i]-p})
    metrics=pd.DataFrame([{'model':m,'MAE':np.mean(np.abs(e)),'RMSE':np.sqrt(np.mean(np.square(e))),'n_backtests':len(e)} for m,e in errors.items()]).sort_values('MAE')
    best=metrics.iloc[0]['model']; h=np.arange(1,steps+1); pred=np.repeat(v[-1],steps) if best=='naive' else np.polyval(np.polyfit(np.arange(len(v)),v,1),len(v)-1+h)
    width=float(np.quantile(np.abs(errors[best]),.95)); periods=pd.period_range(y.index[-1]+1,periods=steps,freq=frequency)
    future=pd.DataFrame({'period':periods.astype(str),'forecast':pred,'lower_empirical_one_step':pred-width,'upper_empirical_one_step':pred+width})
    return {'history':pd.DataFrame({'period':y.index.astype(str),'actual':v}),'forecast':future,'backtest':pd.DataFrame(rows),'metrics':metrics,'model':best,'caution':'Aggregate historical extrapolation, not a causal estimate or credit decision. Bands use one-step errors and are not calibrated multi-step intervals. Stocks and flows must not be mixed.'}

def loan_schedule(principal,annual_rate,years,payments_per_year=12,grace_periods=0):
    """Fixed nominal rate with interest-only grace; no fees or variable rate assumptions."""
    n=int(years*payments_per_year); r=annual_rate/payments_per_year
    if not (principal>0 and 0<=annual_rate<=1 and 0<=grace_periods<n<=1200 and payments_per_year in (1,2,4,12)): raise ValueError('Invalid schedule assumptions')
    remaining=n-grace_periods; payment=principal/remaining if r==0 else principal*r/(-np.expm1(-remaining*np.log1p(r)))
    balance=float(principal); rows=[]
    for i in range(1,n+1):
        interest=balance*r; repaid=0 if i<=grace_periods else min(balance,payment-interest); balance=max(0,balance-repaid)
        rows.append({'period':i,'interest':interest,'principal_repaid':repaid,'debt_service':interest+repaid,'outstanding':balance})
    return pd.DataFrame(rows)

def bibliography(items):
    """CSL JSON or Zotero item envelope. No inferred citation metadata."""
    rows=[]; seen=set()
    for i,item in enumerate(items):
        d=item.get('data',item); title=str(d.get('title','')); doi=str(d.get('DOI','')).strip().lower().removeprefix('https://doi.org/'); year=str(d.get('date','') or d.get('issued',{}).get('date-parts',[['']])[0][0]); authors=d.get('author',d.get('creators',[])); names='; '.join(' '.join(filter(None,[str(a.get('given',a.get('firstName',''))),str(a.get('family',a.get('lastName',a.get('name',a.get('literal','')))))])) for a in authors)
        key=('doi',doi) if doi else ('title',re.sub(r'\W','',title.casefold()),year,names.casefold()) if title else ('row',i)
        duplicate=key in seen; seen.add(key)
        rows.append({'id':str(d.get('id',item.get('key',i))),'title':title,'authors':names,'year_or_date':year,'doi':doi,'url':str(d.get('URL',d.get('url',''))),'type':str(d.get('type',d.get('itemType',''))),'duplicate_candidate':duplicate,'missing_title':not bool(title),'missing_author':not bool(names),'missing_date':not bool(year)})
    return pd.DataFrame(rows,columns=["id","title","authors","year_or_date","doi","url","type","duplicate_candidate","missing_title","missing_author","missing_date"])

def zotero_fetch(kind,library_id,key='',limit=500):
    import requests
    if kind not in ('users','groups') or not str(library_id).isdigit(): raise ValueError('Invalid Zotero library')
    headers={'Zotero-API-Version':'3'}
    if key: headers['Zotero-API-Key']=key
    items=[]
    while len(items)<limit:
        r=requests.get(f'https://api.zotero.org/{kind}/{library_id}/items/top',headers=headers,params={'format':'json','limit':min(100,limit-len(items)),'start':len(items)},timeout=20,allow_redirects=False); r.raise_for_status(); batch=r.json()
        if not isinstance(batch,list): raise ValueError('Unexpected Zotero response')
        items.extend(batch)
        if not batch or len(items)>=int(r.headers.get('Total-Results',len(items))): break
    return items

def cfp_search(query,key):
    import requests
    if not key or not query.strip(): raise ValueError('Search query and Brave Search API key required')
    r=requests.get('https://api.search.brave.com/res/v1/web/search',headers={'X-Subscription-Token':key,'Accept':'application/json'},params={'q':query+' conference "call for papers"','count':20},timeout=20,allow_redirects=False); r.raise_for_status()
    return pd.DataFrame([{'title':x.get('title',''),'url':x.get('url',''),'snippet':x.get('description',''),'retrieved_utc':datetime.now(timezone.utc).isoformat(),'status':'Unverified search lead'} for x in r.json().get('web',{}).get('results',[])])

def bundle(tables, metadata):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for n,t in tables.items(): z.writestr(re.sub(r'[^\w.-]','_',n)+'.csv',t.to_csv(index=False))
        z.writestr('manifest.json',json.dumps(metadata,ensure_ascii=False,indent=2,default=str))
    return b.getvalue()


def evidence_digest(tables, max_tables=30):
    """Exact, traceable table inventory for a longer research draft; no causal inference."""
    lines=[]
    for name, table in list(tables.items())[:max_tables]:
        if not isinstance(table,pd.DataFrame): continue
        lines.append(f"### Evidence: {name}\n\nThe exported table contains {len(table)} rows and {table.shape[1]} variables. " + ("No observations were returned; no finding is supported." if table.empty else "The table should be interpreted at its own unit of observation. Exact selected values follow."))
        if not table.empty:
            cols=list(table.columns[:8]); rows=[]
            for row in table.iloc[:5,:8].itertuples(index=False,name=None):
                rows.append("; ".join(f"{c} = {str(v)[:160]}" for c,v in zip(cols,row)))
            lines.extend("- "+r for r in rows)
            if len(table)>5 or table.shape[1]>8: lines.append("The preview is bounded; use the complete exported table before drawing conclusions.")
    return "\n\n".join(lines) or "No executed numerical evidence is available."
