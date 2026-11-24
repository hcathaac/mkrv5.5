"""v6.1 additive research utilities; no bundled observations or credentials."""
from __future__ import annotations
import io, json, hashlib, zipfile, re, math
from datetime import datetime, timezone
from collections import Counter
from pathlib import PurePosixPath
import numpy as np
import pandas as pd
from workbench_v6 import audit_all, frame_hash

VERSION='6.1.0-rc1'
MAX_ARCHIVE=100*1024*1024


def validation(frames, active=None, outcome=None, predictors=(), time_col=None, entity_col=None, alpha=.05, effect=.15):
    """Audit every source; fit explicitly mapped exploratory OLS diagnostics only."""
    from statsmodels.api import OLS, add_constant
    from statsmodels.stats.outliers_influence import variance_inflation_factor, reset_ramsey
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson
    checks=[]; tables={}
    def add(check,status,detail,source='Active model'): checks.append(dict(source=source,check=check,status=status,detail=detail))
    for name,a in audit_all(frames).items():
        add('Non-empty source','pass' if a['rows'] and a['columns'] else 'blocked',f"{a['rows']} rows; {a['columns']} columns",name)
        add('Duplicate headers','blocked' if a['duplicate_headers'] else 'pass',str(a['duplicate_headers']),name)
        add('Exact duplicate records','review' if a['duplicate_rows'] else 'pass',str(a['duplicate_rows'])+'; duplicates may be legitimate',name)
        missing=sum(x['missing'] for x in a['fields']); inf=sum(x['infinite'] for x in a['fields'])
        add('Missing cells','review' if missing else 'pass',str(missing)+'; mechanism not inferred',name)
        add('Infinite numeric values','blocked' if inf else 'pass',str(inf),name)
        tables[f'Audit {name}']=pd.DataFrame(a['fields'])
    add('Endogeneity / identification','not assessed','Requires research design, credible instruments or identification assumptions; cannot be established by OLS residual tests.')
    add('Multiple testing','review','Declare the hypothesis family; use existing BH/Bonferroni engines where relevant. This diagnostic suite does not adjust other analyses.')
    add('Leakage and temporal validation','review','Verify feature availability at prediction time and use disjoint, ordered or grouped validation as required. No automatic leakage certificate.')
    add('Reproducibility','review','Retain source hashes, mappings, versions, exclusions and seeds; this report alone is not a full replication.')
    if active is not None and not active.empty and not active.columns.duplicated().any():
        if time_col:
            if time_col not in active: raise ValueError('Unknown time column')
            keys=[time_col]+([entity_col] if entity_col else [])
            if any(k not in active for k in keys): raise ValueError('Unknown entity column')
            duplicates=int(active.duplicated(keys).sum()); miss=int(active[keys].isna().any(axis=1).sum())
            add('Entity/time or time key','review' if duplicates or miss else 'pass',f'{duplicates} duplicate keys, {miss} missing keys; repeated transactions may require aggregation.')
        if outcome and predictors:
            cols=[outcome,*predictors]
            if len(set(cols))!=len(cols) or any(c not in active for c in cols): raise ValueError('Map distinct valid outcome and predictor columns')
            if any(not pd.api.types.is_numeric_dtype(active[c]) for c in cols): raise ValueError('Select numeric columns')
            if len(predictors)>30: raise ValueError('Maximum 30 diagnostic predictors')
            z=active[cols].replace([np.inf,-np.inf],np.nan).dropna(); n=len(z); k=len(predictors)
            add('Complete-case coverage','review' if n<len(active) else 'pass',f'{n}/{len(active)} retained; no imputation')
            x=add_constant(z[list(predictors)].astype(float),has_constant='add')
            if n<=k+2 or np.linalg.matrix_rank(x)<k+1 or z[outcome].nunique()<2:
                add('Model feasibility','blocked','Insufficient complete cases, rank deficiency or constant outcome')
            else:
                add('Model feasibility','pass',f'{n} cases, {k} predictors, full rank; not causal validation')
                fit=OLS(z[outcome].astype(float),x).fit(); robust=fit.get_robustcov_results(cov_type='HC3')
                vifs=pd.DataFrame([{'variable':c,'VIF':variance_inflation_factor(x.to_numpy(),i)} for i,c in enumerate(x.columns) if c!='const'])
                tables['VIF']=vifs; add('Multicollinearity','review' if (vifs.VIF>=5).any() else 'pass','VIF >=5 screening threshold, not a universal rule')
                def pcheck(label,func):
                    try:
                        p=float(func()); add(label,'not assessed' if not math.isfinite(p) else ('review' if p<alpha else 'pass'),f'p={p:.6g}; non-rejection is not proof')
                    except Exception as exc: add(label,'not assessed',type(exc).__name__)
                pcheck('Breusch–Pagan heteroskedasticity',lambda:het_breuschpagan(fit.resid,x)[1])
                pcheck('Ramsey RESET specification',lambda:reset_ramsey(fit,degree=3).pvalue)
                if time_col and not entity_col:
                    # Durbin-Watson needs a verified order; the current row order is never assumed chronological.
                    order=active.loc[z.index,time_col] if active.index.is_unique else None
                    if order is not None and order.notna().all() and order.is_unique:
                        add('Durbin–Watson','review',f'DW={durbin_watson(pd.Series(fit.resid,index=z.index).loc[order.sort_values().index]):.4g}; verify temporal frequency and ordering')
                tables['OLS covariance sensitivity']=pd.DataFrame({'term':x.columns,'estimate':fit.params.to_numpy(),'classical_SE':fit.bse.to_numpy(),'HC3_SE':robust.bse,'HC3_p':robust.pvalues})
                # Exact noncentral F power for a joint test of k regression slopes under stipulated f².
                from scipy.stats import f,ncf
                if not 0<alpha<1 or not effect>0: raise ValueError('Invalid alpha or assumed f²')
                df2=n-k-1; power=float(ncf.sf(f.ppf(1-alpha,k,df2),k,df2,effect*n))
                add('Conditional joint-test power','review',f'{power:.3f} assuming f²={effect:g}, alpha={alpha:g}, iid errors; not observed/post-hoc power or design adequacy certificate')
                add('Covariance robustness','review','Classical vs HC3 comparison computed; panel/cluster/time dependence requires appropriate covariance and additional sensitivity analysis.')
    else: add('Active model','not assessed','No active dataset / model mapping')
    tables['Validation register']=pd.DataFrame(checks)
    return tables


def _safe_zip(data):
    if len(data)>MAX_ARCHIVE: raise ValueError('Archive exceeds 100 MB')
    z=zipfile.ZipFile(io.BytesIO(data)); entries=z.infolist()
    if len(entries)>300 or sum(x.file_size for x in entries)>MAX_ARCHIVE: z.close(); raise ValueError('Archive expanded size or member limit exceeded')
    names=[x.filename for x in entries]
    if len(names)!=len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names): z.close(); raise ValueError('Unsafe archive paths')
    return z


def _json(x): return json.dumps(x,ensure_ascii=False,indent=2,default=str,allow_nan=False).encode('utf-8')


def project_pack(title,frames,question='',tables=None,documents=None,settings=None):
    """Portable, data-only ZIP; no pickle, session secrets or filesystem paths."""
    if not title.strip(): raise ValueError('Project name required')
    b=io.BytesIO(); m={'format':'mkrv-project-1','version':VERSION,'title':title[:200],'question':question,'created_utc':datetime.now(timezone.utc).isoformat(),'datasets':[],'tables':[],'documents':[],'settings':{k:v for k,v in (settings or {}).items() if k in ('active_source','method','research_question','dataset_relationship','join_keys','join_type')}}
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for group,items in [('datasets',frames),('tables',tables or {})]:
            for i,(name,df) in enumerate(items.items()):
                if not isinstance(df,pd.DataFrame): continue
                # Parquet preserves column types and missing values; no pickle fallback.
                if df.columns.duplicated().any(): raise ValueError(f'Duplicate column labels: {name}')
                payload=df.to_parquet(index=True); path=f'{group}/{i:03d}.parquet'
                z.writestr(path,payload); m[group].append({'name':str(name),'path':path,'sha256':hashlib.sha256(payload).hexdigest(),'rows':len(df)})
        for i,(name,payload) in enumerate((documents or {}).items()):
            suffix=PurePosixPath(name).suffix.lower()
            if suffix not in ('.pdf','.docx','.json','.bib','.ris','.md','.txt'): raise ValueError('Unsupported project document')
            path=f'documents/{i:03d}{suffix}'; z.writestr(path,payload); m['documents'].append({'name':PurePosixPath(name).name,'path':path,'sha256':hashlib.sha256(payload).hexdigest()})
        z.writestr('manifest.json',_json(m))
    with _safe_zip(b.getvalue()): pass
    return b.getvalue()


def project_read(data):
    with _safe_zip(data) as z:
        m=json.loads(z.read('manifest.json'))
        if m.get('format')!='mkrv-project-1': raise ValueError('Unsupported project format')
        out={}
        for group in ('datasets','tables','documents'):
            items={}
            for item in m.get(group,[]):
                path=item['path']; name=item['name']
                if not path.startswith(group+'/') or name in items: raise ValueError('Invalid project manifest')
                raw=z.read(path)
                if hashlib.sha256(raw).hexdigest()!=item['sha256']: raise ValueError('Project checksum mismatch')
                if group!='documents':
                    if not path.endswith('.parquet'): raise ValueError('Unexpected data format')
                    frame=pd.read_parquet(io.BytesIO(raw))
                    if len(frame)!=item['rows']: raise ValueError('Project row count mismatch')
                    items[name]=frame
                else: items[name]=raw
            out[group]=items
    return m,out


def collect_tables(state):
    """Allowlisted computed results only; no provider configuration, raw chat or keys."""
    results={}
    keys=['group_tests','chi_tests','normal_tests','model_output','iv_result','did_result','regularised','mc_ols','mc_portfolio','batch_coef','batch_fit','rd_panel','panel_suite','ts_tests','arima','granger','pca','clusters','alpha','advanced_clusters','predictive_suite','scenario','allocation','mcda_output','chair_result','autopilot_result','agentic_run_result','_v6_forecast','_v6_schedule','_v61_validation','_v61_simulation','_v61_debt_stress','_v61_review','_v61_journals']
    def visit(name,obj,depth=0):
        if depth>3 or len(results)>=160: return
        if isinstance(obj,pd.DataFrame): results[name]=obj
        elif isinstance(obj,dict):
            for k,v in obj.items(): visit(f'{name} / {k}',v,depth+1)
        elif isinstance(obj,(tuple,list)):
            for i,v in enumerate(obj): visit(f'{name} / {i+1}',v,depth+1)
        else:
            for attr in ('tables','coefficients','predictions','diagnostics','fit','summary','rankings','assignments','profiles','embedding'):
                v=getattr(obj,attr,None)
                if isinstance(v,(pd.DataFrame,dict)): visit(f'{name} / {attr}',v,depth+1)
    for k in keys:
        if k in state: visit(k,state[k])
    if '_v6_bib' in state: visit('Bibliography audit',state['_v6_bib'][1])
    return results


def peer_review(text, tables=None):
    """Structural evidence checklist, not acceptance prediction or independent expert judgement."""
    sections={'Research question':r'research question|objective|aim|ερευνητικ.{0,4} ερ[ώω]τη|σκοπ[όο]','Methods':r'method|μεθοδολογ','Results':r'results|findings|αποτελεσμ|ευρ[ήη]μα','Limitations':r'limitation|περιορισμ','References':r'references|bibliography|βιβλιογραφ','Uncertainty':r'confidence interval|credible interval|standard error|διάστημα εμπιστοσύνης','Reproducibility':r'reproducib|replicat|data availability|αναπαραγωγ','Ethics / disclosures':r'ethic|conflict of interest|δεοντολογ|σύγκρουση συμφερόντων'}
    if not text.strip(): raise ValueError('No extractable manuscript text')
    rows=[]
    for label,pat in sections.items():
        m=re.search(pat,text,re.I); rows.append({'reviewer':'Methods / reporting','criterion':label,'status':'keyword found — verify' if m else 'not located — review','evidence':text[max(0,m.start()-35):m.end()+65].replace('\n',' ') if m else '', 'action':'Verify substantive adequacy and evidence; keyword presence is not validation.'})
    nums=re.findall(r'(?<!\w)p\s*[=<≤]\s*(\d*\.\d+)',text,re.I)
    rows.append({'reviewer':'Statistical reviewer','criterion':'Reported p-values','status':'review','evidence':f'{len(nums)} textual p-value patterns; {len(tables or {})} available result tables','action':'Match each claim to the exact model, sample, coefficient, CI and adjusted p-value where applicable. No automatic numerical matching claimed.'})
    rows.append({'reviewer':'Editorial reviewer','criterion':'Originality and scope','status':'not assessed','evidence':'No verified external novelty search','action':'Compare contribution against verified literature and target journal; check citations and ethical requirements.'})
    return pd.DataFrame(rows)


def funding_simulation(budget, absorption, budget_shock, absorption_sd, draws=5000, seed=42, regions=None):
    """Scenario assumptions, not fitted forecasts. Budget cap enforced in every draw."""
    if not all(math.isfinite(float(v)) for v in (budget,absorption,budget_shock,absorption_sd)) or budget<0 or not 0<=absorption<=1 or budget_shock< -1 or not 0<=absorption_sd<=1 or not 100<=draws<=50000: raise ValueError('Invalid assumptions')
    rng=np.random.default_rng(seed); cap=budget*(1+budget_shock); rates=np.clip(rng.normal(absorption,absorption_sd,draws),0,1); spent=cap*rates
    samples=pd.DataFrame({'draw':np.arange(1,draws+1),'budget':cap,'absorption_rate':rates,'absorbed':spent,'unspent':cap-spent})
    summary=pd.DataFrame([{'metric':c,'mean':samples[c].mean(),'p05':samples[c].quantile(.05),'median':samples[c].median(),'p95':samples[c].quantile(.95)} for c in ('absorption_rate','absorbed','unspent')])
    out={'Scenario summary':summary,'Simulation draws':samples}
    if regions is not None and len(regions):
        if regions.region.duplicated().any() or regions.weight.isna().any() or (regions.weight<0).any() or not np.isfinite(regions.weight).all() or regions.weight.sum()<=0: raise ValueError('Use unique regions and nonnegative finite weights with positive sum')
        alloc=regions.copy(); alloc['share']=alloc.weight/alloc.weight.sum(); alloc['budget']=cap*alloc.share; alloc['mean_absorbed']=spent.mean()*alloc.share; out['Regional scenario allocation']=alloc
    return out


def journal_search(query,openalex_key='',limit=25,session=None,check_doaj=False):
    """Crossref sample of topic-matched works; optional OpenAlex ISSN enrichment, DOAJ flag.
    A count is sample relevance, not impact factor or publication probability."""
    import requests
    s=session or requests.Session(); q=query.strip()
    if not q or not 1<=limit<=50: raise ValueError('Enter a topic; limit 1–50')
    r=s.get('https://api.crossref.org/works',params={'query.bibliographic':q,'filter':'type:journal-article','rows':min(100,limit*4),'select':'container-title,ISSN,publisher,DOI'},timeout=20); r.raise_for_status()
    grouped={}
    for w in r.json()['message']['items']:
        title=next(iter(w.get('container-title') or []),''); issn=next(iter(w.get('ISSN') or []),'')
        if not title: continue
        k=issn or title.casefold(); item=grouped.setdefault(k,{'journal':title,'ISSN':issn,'publisher':w.get('publisher',''),'sample_matches':0,'example_DOI':w.get('DOI',''),'example_article_url':'https://doi.org/'+w['DOI'] if w.get('DOI') else '','DOAJ_direct':'not checked','OpenAlex_DOAJ':'not checked','OpenAlex_OA':'not checked','APC':'verify official site','submission_requirements':'verify official site','website':''})
        item['sample_matches']+=1
    rows=sorted(grouped.values(),key=lambda x:-x['sample_matches'])[:limit]
    if openalex_key:
        for item in rows[:10]:
            if not item['ISSN']: continue
            try:
                res=s.get('https://api.openalex.org/sources',params={'filter':'issn:'+item['ISSN'],'per_page':1,'api_key':openalex_key},timeout=10); res.raise_for_status(); src=next(iter(res.json().get('results',[])),{})
                if src:
                    item['OpenAlex_DOAJ']=src.get('is_in_doaj','unknown'); item['OpenAlex_OA']=src.get('is_oa','unknown'); item['website']=src.get('homepage_url') or ''; item['APC']=str(src['apc_usd'])+' USD (OpenAlex; verify)' if src.get('apc_usd') is not None else 'verify official site'
            except requests.RequestException: item['OpenAlex_DOAJ']='enrichment unavailable'
    if check_doaj:
        from urllib.parse import quote
        for item in rows[:10]:
            issn=item['ISSN']
            if not re.fullmatch(r'\d{4}-[\dXx]{4}',issn): continue
            try:
                resp=s.get('https://doaj.org/api/v4/search/journals/'+quote('issn:'+issn,safe=':'),params={'pageSize':1},timeout=6); resp.raise_for_status()
                records=resp.json().get('results',[])
                item['DOAJ_direct']='record found' if records else 'not found for queried ISSN'
                if records: item['DOAJ_record']='https://doaj.org/toc/'+issn
            except requests.RequestException: item['DOAJ_direct']='lookup unavailable'
    return pd.DataFrame(rows)


def publication_pack(tables,frames=None,question='',bibliography=None,extras=None):
    """Actual output tables and portable report; no fake model reproduction scripts."""
    from docx import Document
    b=io.BytesIO(); doc=Document(); doc.add_heading('Research evidence export',0); doc.add_paragraph(question or 'Research question not supplied'); manifest={'version':VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':{n:frame_hash(f) for n,f in (frames or {}).items()},'tables':[],'notes':'Tables are actual captured results. No journal acceptance or replication guarantee. Data files optional.'}
    def csv_safe(frame):
        out=frame.copy()
        for c in out.columns:
            if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]): out[c]=out[c].map(lambda v:"'"+v if isinstance(v,str) and v.startswith(('=','+','-','@','\t','\r')) else v)
        return out.to_csv(index=False).encode('utf-8-sig')
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        excel=io.BytesIO()
        with pd.ExcelWriter(excel,engine='openpyxl') as writer:
            pd.DataFrame({'question':["'"+question if question.startswith(('=','+','-','@')) else question],'version':[VERSION]}).to_excel(writer,sheet_name='Readme',index=False)
            for i,(name,df) in enumerate(tables.items()):
                path=f'tables/{i:03d}.csv'; z.writestr(path,csv_safe(df)); manifest['tables'].append({'name':name,'path':path,'rows':len(df),'columns':df.shape[1]})
                if len(df)<1048576 and df.shape[1]<=16384:
                    # Preserve numeric types and neutralise text formulas separately.
                    safe=df.copy();
                    for c in safe.columns:
                        if pd.api.types.is_object_dtype(safe[c]) or pd.api.types.is_string_dtype(safe[c]): safe[c]=safe[c].map(lambda v:"'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v)
                    safe.columns=["'"+str(c) if str(c).startswith(('=','+','-','@')) else str(c) for c in safe.columns]; safe.to_excel(writer,sheet_name=f'Table_{i:03d}',index=False)
                doc.add_heading(str(name),1); doc.add_paragraph(f'{len(df)} rows × {df.shape[1]} variables; see {path} for complete evidence.');
                if df.shape[1]:
                    sample=df.iloc[:8,:8]; t=doc.add_table(rows=1,cols=sample.shape[1]);
                    for j,c in enumerate(sample.columns): t.rows[0].cells[j].text=str(c)
                    for row in sample.itertuples(index=False,name=None):
                        cells=t.add_row().cells
                        for j,v in enumerate(row): cells[j].text=str(v)[:300]
                if len(df)<=10000 and 0<df.shape[1]<=100:
                    z.writestr(f'tables/{i:03d}.tex',df.to_latex(index=False,escape=True))
        z.writestr('tables.xlsx',excel.getvalue()); d=io.BytesIO(); doc.save(d); z.writestr('research_evidence.docx',d.getvalue()); z.writestr('manifest.json',_json(manifest))
        if bibliography is not None:
            from workbench_v6 import bibliography as audit_bibliography
            z.writestr('bibliography.json',_json(bibliography)); refs=audit_bibliography(bibliography); z.writestr('bibliography_audit.csv',csv_safe(refs))
            # Minimal portable bibliographic records, without inferred missing metadata.
            ris=[]; bib=[]
            def tex(v): return str(v).replace('\\',r'\textbackslash{}').replace('{',r'\{').replace('}',r'\}').replace('%',r'\%').replace('&',r'\&').replace('_',r'\_').replace('#',r'\#').replace('\n',' ').replace('\r',' ')
            def line(v): return str(v).replace('\n',' ').replace('\r',' ')
            for i,r in refs.iterrows():
                typ='JOUR' if str(r['type']) in ('article-journal','journalArticle') else 'GEN'
                fields={'title':r['title'],'author':str(r['authors']).replace('; ',' and '),'year':str(r['year_or_date'])[:4] if re.match(r'^\d{4}',str(r['year_or_date'])) else '', 'doi':r['doi'],'url':r['url']}
                bib.append('@'+('article' if typ=='JOUR' else 'misc')+'{ref'+str(i+1)+',\n'+',\n'.join('  '+k+' = {'+tex(v)+'}' for k,v in fields.items() if v)+'\n}')
                ris.extend(['TY  - '+typ,'TI  - '+line(r['title'])]+['AU  - '+line(n) for n in str(r['authors']).split('; ') if n]+['PY  - '+line(fields['year']),'DO  - '+line(r['doi']),'UR  - '+line(r['url']),'ER  - ',''])
            z.writestr('references.bib','\n\n'.join(bib)); z.writestr('references.ris','\n'.join(ris))
        for i,(name,df) in enumerate((frames or {}).items()): z.writestr(f'data/{i:03d}.parquet',df.to_parquet(index=True))
        for i,(name,raw) in enumerate((extras or {}).items()):
            if PurePosixPath(name).suffix.lower() in ('.svg','.pdf','.png','.tiff','.zip','.docx','.bib','.ris','.tex','.r','.do','.py','.html'): z.writestr(f'artifacts/{i:03d}_{PurePosixPath(name).name}',raw)
    return b.getvalue()


def scenario_figures(draws):
    """Publication graphics derived from actual simulation draws."""
    import matplotlib.pyplot as plt
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for mono in (False,True):
            fig,ax=plt.subplots(figsize=(7.5,4.5)); ax.hist(draws['absorbed'],bins=35,color='0.35' if mono else '#187A98',edgecolor='white'); ax.axvline(draws['absorbed'].mean(),color='black' if mono else '#B44923',linestyle='--',label='Mean'); ax.set(xlabel='Absorbed funding (€)',ylabel='Simulation draws',title='Conditional funding scenario'); ax.legend(); fig.tight_layout()
            for fmt in ('png','svg','pdf','tiff'):
                out=io.BytesIO(); fig.savefig(out,format=fmt,dpi=600 if fmt in ('png','tiff') else None,**({'pil_kwargs':{'compression':'tiff_lzw'}} if fmt=='tiff' else {})); z.writestr(f'scenario_{"grayscale" if mono else "colour"}.{fmt}',out.getvalue())
            plt.close(fig)
    return b.getvalue()


def debt_rate_stress(principal,base_rate,shock,years,frequency=12):
    from workbench_v6 import loan_schedule
    if not 0<=base_rate+shock<=1: raise ValueError('Stressed nominal rate must be between 0% and 100%')
    rows=[]; schedules={}
    for label,rate in [('Base',base_rate),('Stress',base_rate+shock)]:
        schedule=loan_schedule(principal,rate,years,frequency); schedules[label+' debt schedule']=schedule
        rows.append({'scenario':label,'annual_nominal_rate':rate,'total_interest':schedule.interest.sum(),'total_debt_service':schedule.debt_service.sum(),'peak_periodic_payment':schedule.debt_service.max()})
    return {'Interest-rate stress summary':pd.DataFrame(rows),**schedules}
