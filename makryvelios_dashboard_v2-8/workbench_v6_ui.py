from __future__ import annotations
import json, io, zipfile
from urllib.parse import quote_plus
import pandas as pd
import streamlit as st
from workbench_v6 import METHODS, audit_all, clean, frame_hash, recommend, loan_forecast, loan_schedule, bibliography, zotero_fetch, cfp_search, bundle

NEW_PAGES=['14. Data cleaning & validation','15. Public loan forecasting','16. Conference & CFP discovery','17. Zotero bibliography']

def clear_session(hard=False):
    # Streamlit caches are process-wide: clear only the two user-data functions in app.py.
    # Never clear unrelated global resources in a multi-user app.
    if hard:
        epoch=st.session_state.get('_v6_epoch',0)+1
        st.session_state.clear(); st.session_state['_v6_epoch']=epoch
    else:
        for k in list(st.session_state):
            if not (k.startswith('global_llm_') or k in ('_v6_epoch','_v6_signature','_v6_upload_signature','_v6_module','_v6_upload','_v6_cleaned') or k.startswith(('_v6_upload_', '_v6_intake_'))):
                del st.session_state[k]
    st.rerun()

def invalidate_on_change(signature):
    previous=st.session_state.get('_v6_signature')
    if previous is not None and previous!=signature:
        # Called before analysis-page widgets are created. Keep global provider settings and intake widgets.
        for k in list(st.session_state):
            if k.startswith(('global_llm_','_v6_')) or k=='llm_config': continue
            del st.session_state[k]
        for k in ('_v6_forecast','_v6_schedule','_v6_clean_preview'):
            st.session_state.pop(k,None)
    st.session_state['_v6_signature']=signature

def overview(frames,pages):
    st.subheader('Research workflow navigator')
    q=st.text_area('Describe your research question (Greek or English)',key='_v6_question')
    if q.strip():
        routes=recommend(q); st.markdown('**Recommended workflow (rule-based, not an LLM inference)**')
        st.write('1. Upload source datasets and verify observation units, variable definitions and dates. 2. Audit each source independently. 3. Resolve duplicate keys, units and missingness before joining. 4. Choose the relevant estimator and map variables explicitly. 5. Review diagnostics and export the evidence.')
        st.dataframe(pd.DataFrame(routes,columns=['Family','Module','Use and prerequisites']),hide_index=True)
        options=[x[1] for x in routes if x[1] in pages]
        if options:
            chosen=st.selectbox('Open recommended module',options,key='_v6_target')
            def go():
                st.session_state['_v6_family']='All modules'
                st.session_state['_v6_module']=st.session_state['_v6_target']
            st.button('Go to module',on_click=go,key='_v6_go')
        if frames:
            audit=audit_all(frames); st.dataframe(pd.DataFrame([{k:v for k,v in a.items() if k!='fields'}|{'dataset':n} for n,a in audit.items()]),hide_index=True)
            st.download_button('Download source audit JSON',json.dumps(audit,ensure_ascii=False,indent=2),'source_audit.json','application/json')
        else: st.info('No dataset loaded. Upload files for quantitative analysis; literature and conference tools can work independently.')
    with st.expander('Complete grouped method catalogue'):
        st.dataframe(pd.DataFrame(METHODS,columns=['Family','Module','Use and prerequisites']),hide_index=True)
        st.download_button('Download grouped methods CSV',pd.DataFrame(METHODS,columns=['Family','Module','Use and prerequisites']).to_csv(index=False),'method_catalogue.csv','text/csv')

def cleaner(frames,df,selected_label):
    st.header('Data cleaning & validation'); st.caption('No automatic imputation, outlier deletion, or numeric conversion. Review all changes before applying.')
    if not frames: st.info('Upload a dataset first.'); return
    audits=audit_all(frames)
    st.download_button('Download audit of every source (JSON)',json.dumps(audits,ensure_ascii=False,indent=2),'all_sources_audit.json','application/json')
    label=st.selectbox('Source to clean',list(frames),key='_v6_clean_source'); original=frames[label]
    st.dataframe(pd.DataFrame(audits[label]['fields']),hide_index=True)
    trim=st.checkbox('Trim text whitespace',True); headers=st.checkbox('Standardise column names',True); duplicates=st.checkbox('Remove exact duplicate observations (only if justified)',False)
    tokens=st.text_input('Explicit missing-value codes, comma separated (optional)'); numbers=st.multiselect('Explicit numeric columns (after header normalisation)',list(clean(original,normalise_headers=headers,trim=False)[0].columns)); decimal=st.checkbox('Decimal comma (not thousands separator)',False)
    if st.button('Preview cleaning'):
        try:
            data,log=clean(original,trim=trim,normalise_headers=headers,remove_duplicates=duplicates,missing_tokens=tuple(t.strip() for t in tokens.split(',') if t.strip()),numeric_columns=numbers,decimal_comma=decimal)
            st.session_state['_v6_clean_preview']=(label,frame_hash(original),data,log)
        except Exception as exc: st.error(str(exc))
    item=st.session_state.get('_v6_clean_preview')
    if item and item[0]==label and item[1]==frame_hash(original):
        _,_,data,log=item; st.dataframe(log,hide_index=True); st.dataframe(data.head(200)); st.download_button('Download cleaned data + change log',bundle({'cleaned':data,'changes':log},{'before':audits[label],'after':audit_all({label:data})[label]}),'cleaned_dataset.zip','application/zip')
        if st.button('Apply cleaned source to all analytical modules'):
            st.session_state.setdefault('_v6_cleaned',{})[label]=(frame_hash(original),data); st.rerun()
    if label in st.session_state.get('_v6_cleaned',{}):
        if st.button('Restore original uploaded source'):
            st.session_state['_v6_cleaned'].pop(label,None); st.rerun()

def _forecast_figures(run):
    import matplotlib.pyplot as plt
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for mono in (False,True):
            fig,ax=plt.subplots(figsize=(8,4.5)); h=run['history']; f=run['forecast']; x=range(len(h)); y=range(len(h),len(h)+len(f)); ax.plot(x,h.actual,color='black' if mono else '#126782',label='Observed'); ax.plot(y,f.forecast,'--',color='0.3' if mono else '#b04a14',label='Forecast'); ax.fill_between(list(y),f.lower_empirical_one_step.to_numpy(),f.upper_empirical_one_step.to_numpy(),color='0.75' if mono else '#e4a76e',alpha=.4,label='One-step error band'); ax.set_ylabel('Loan amount (source units)'); ax.set_xlabel('Sequential period'); ax.legend(); fig.tight_layout()
            for fmt in ('png','svg','pdf'):
                out=io.BytesIO(); fig.savefig(out,format=fmt,dpi=600 if fmt=='png' else None); z.writestr(f'forecast_{"grayscale" if mono else "colour"}.{fmt}',out.getvalue())
            plt.close(fig)
    return b.getvalue()

def loans(df):
    st.header('Public loan forecasting and debt-service scenarios'); st.warning('Research and budget planning only. Not a credit decision or causal estimate. Do not mix commitments, disbursements, repayments and outstanding balances.')
    tab,sch=st.tabs(['Historical forecasting','Debt-service scenario'])
    with tab:
        if df.empty: st.info('Upload historical records containing period and amount.');
        else:
            dates=st.selectbox('Date / year',list(df.columns)); val=st.selectbox('Amount in one currency and unit',list(df.columns)); freq=st.selectbox('Frequency',['Y','Q','M']); agg=st.selectbox('Aggregation',['sum','last'],help='Sum flows; last observation for stocks only if dates are unique.'); horizon=st.slider('Forecast periods',1,24,3)
            settings=(frame_hash(df),dates,val,freq,agg,horizon)
            if st.session_state.get('_v6_loan_settings') != settings:
                st.session_state.pop('_v6_forecast',None); st.session_state['_v6_loan_settings']=settings
            if st.button('Backtest and forecast'):
                try: st.session_state['_v6_forecast']=loan_forecast(df,dates,val,freq,horizon,agg)
                except Exception as exc: st.error(str(exc))
            run=st.session_state.get('_v6_forecast')
            if run:
                st.write(run['caution']); st.dataframe(run['metrics']); st.dataframe(run['forecast']); st.line_chart(pd.concat([run['history'],run['forecast']],ignore_index=True).set_index('period')[['actual','forecast']]); st.download_button('Download forecast evidence',bundle({k:v for k,v in run.items() if isinstance(v,pd.DataFrame)},{'model':run['model'],'caution':run['caution'],'dataset_sha256':frame_hash(df)}),'loan_forecast.zip','application/zip'); st.download_button('Download publication figures (600 dpi, SVG, PDF; colour + grayscale)',_forecast_figures(run),'loan_figures.zip','application/zip')
    with sch:
        p=st.number_input('Principal',min_value=1.0,value=100000.0); r=st.number_input('Annual nominal rate (%)',min_value=0.0,max_value=100.0,value=3.0); years=st.number_input('Maturity (years)',1,50,10); per=st.selectbox('Payments per year',[12,4,2,1]); grace=st.number_input('Interest-only grace periods',0,int(years*per)-1,0)
        settings=(p,r,years,per,grace)
        if st.session_state.get('_v6_schedule_settings') != settings:
            st.session_state.pop('_v6_schedule',None); st.session_state['_v6_schedule_settings']=settings
        if st.button('Calculate illustrative schedule'): st.session_state['_v6_schedule']=loan_schedule(p,r/100,years,per,grace)
        s=st.session_state.get('_v6_schedule')
        if s is not None: st.dataframe(s); st.download_button('Download amortisation CSV',s.to_csv(index=False),'loan_schedule.csv','text/csv')

def conferences():
    st.header('Conference & calls for papers discovery'); st.caption('Live search uses Brave Search API; search results are unverified leads, not confirmed open calls. Only the query is sent, not datasets.')
    query=st.text_input('Research field, topic, region or year'); key=st.text_input('Brave Search API key (session only)',type='password')
    if query: st.markdown('[Open browser search](https://www.google.com/search?q='+quote_plus(query+' conference call for papers')+')')
    if st.button('Search live'):
        try: st.session_state['_v6_cfp']=cfp_search(query,key)
        except Exception as exc: st.error(str(exc))
    result=st.session_state.get('_v6_cfp')
    if result is not None: st.dataframe(result,hide_index=True,column_config={'url':st.column_config.LinkColumn('Source')}); st.download_button('Download CFP leads',result.to_csv(index=False),'cfp_leads.csv','text/csv'); st.info('Check the official organiser page, deadline, venue, fees and submission requirements before relying on any lead.')

def zotero():
    st.header('Zotero bibliography'); st.caption('Read-only import; no remote modifications. Export preserves raw source metadata and flags candidate duplicates.')
    f=st.file_uploader('Import Zotero / CSL JSON',type=['json'],key='_v6_zotero_file'); kind=st.selectbox('Library type',['users','groups']); ident=st.text_input('Zotero numeric library ID'); key=st.text_input('Read-only Zotero API key (optional for public libraries)',type='password')
    if st.button('Load bibliography'):
        try:
            items=json.loads(f.getvalue()) if f else zotero_fetch(kind,ident,key)
            if not isinstance(items,list): raise ValueError('Expected a JSON array of items')
            st.session_state['_v6_bib']=(items,bibliography(items))
        except Exception as exc: st.error(str(exc))
    item=st.session_state.get('_v6_bib')
    if item:
        items,table=item; st.dataframe(table,hide_index=True); st.download_button('Download bibliography audit CSV',table.to_csv(index=False),'bibliography_audit.csv','text/csv'); st.download_button('Download original bibliography JSON',json.dumps(items,ensure_ascii=False,indent=2),'bibliography.json','application/json')
        report=table.groupby('type',dropna=False).size().rename('count').reset_index(); st.dataframe(report); st.download_button('Download bibliography report',bundle({'references':table,'types':report},{'items':len(table),'candidate_duplicates':int(table.duplicate_candidate.sum()),'warning':'Metadata not independently verified'}),'zotero_report.zip','application/zip')
