"""Streamlit front end for the native pipeline; separate from the sidebar analysis cache."""
from __future__ import annotations
import json,tempfile
from pathlib import Path
import streamlit as st
import pandas as pd
import research_pipeline as rp
PAGE='25. Guided research pipeline'

def make_template(s,root):
    gate=s['next_gate']; cols=s.get('active_columns',[])
    if gate==1: return {'cleaning':{n:{} for n in s['sources']},'integration':{'mode':'separate','sources':s['sources'][:1],'reason':''},'missingness_policy':'','review_note':''}
    if gate==2: return {'research_questions':[{'id':'RQ1','group':'Descriptive','question':s['question'],'type':'descriptive'}],'hypotheses':[],'source_notes':[],'gap':'','literature_limitations':''}
    if gate==3: return {'analyses':[{'id':'A1','method':'descriptive','params':{'columns':[]},'rq_ids':[r['id'] for r in rp.config_at(root,2)['research_questions']][:1],'reason':'','assumptions':'','figures':False}],'multiplicity':'not_applicable','scope_note':''}
    if gate==4:
        d=rp.config_at(root,2); return {'markdown':'','rq_answers':[{'id':r['id'],'status':'not_answerable','answer':'','evidence_ids':[]} for r in d['research_questions']],'hypothesis_assessments':[{'id':h['id'],'verdict':'not_tested','reason':'','evidence_ids':[]} for h in d['hypotheses']],'numeric_claims':[],'bibliography':[],'originality_note':''}
    if gate==5: return {'review_scope':'Structural and reporting checklist; verify every claim and source independently.'}
    if gate==6: return {'issues':[{'row':i,'resolution':'','status':'addressed'} for i in range(s['review_count'])],'human_review_note':'','reference_review':'','data_sharing_note':'','shareable_paths':[]}
    return {}

def pipeline_ui():
    st.header('Guided research pipeline'); st.caption('One gate at a time · original computation only · private checkpoints · no external LLM or network call by the runner itself.')
    holder=st.session_state.get('_v6_pipeline_temp')
    if holder is None:
        st.info('Start a fresh independent research session here. Sidebar datasets are not implicitly reused; this runner has its own inputs and no previous analysis cache.')
        q=st.text_area('Principal research objective',key='_v6_pipeline_objective'); files=st.file_uploader('Upload datasets and supporting PDF / DOCX',type=[x[1:] for x in sorted(rp.DATA_EXT|rp.DOC_EXT)],accept_multiple_files=True,key='_v6_pipeline_inputs')
        archive=st.file_uploader('Or restore your own trusted private checkpoint (same code and environment)',type=['zip'],key='_v6_pipeline_restore')
        if st.button('Start / restore private research run',type='primary',disabled=not archive and (not files or not q.strip())):
            tmp=tempfile.TemporaryDirectory(prefix='mkrv_pipeline_'); root=Path(tmp.name)
            try:
                if archive: rp.restore(root,archive.getvalue())
                else: rp.new_run(root,[(f.name,f.getvalue()) for f in files],q)
                st.session_state['_v6_pipeline_temp']=tmp; st.rerun()
            except Exception as e: tmp.cleanup(); st.error(str(e))
        return
    root=Path(holder.name)
    try: s=rp.status(root)
    except Exception as e: st.error(str(e)); return
    gate=s['next_gate']; st.progress(min(gate,7)/7,text=f'{min(gate,7)} / 7 stages completed'); st.caption(f"Run {s['run_id'][:12]} · {s['application_version']} · {s['status']}")
    st.dataframe(pd.DataFrame([{'Stage':i,'Purpose':n,'Status':'Completed' if i<gate else ('Current' if i==gate else 'Waiting')} for i,n in enumerate(rp.GATES)]),hide_index=True)
    st.warning('Checkpoint contains private inputs and evidence. Download only to authorised storage; source text and row-level results may be confidential.')
    if st.button('Prepare downloadable checkpoint',key='_v6_pipeline_checkpoint'):
        try: st.session_state['_v6_pipeline_archive']=rp.checkpoint(root)
        except Exception as e: st.error(str(e))
    if st.session_state.get('_v6_pipeline_archive'):
        st.download_button('Download PRIVATE checkpoint',st.session_state['_v6_pipeline_archive'],'private_research_checkpoint.zip','application/zip')
    with st.expander('Audits, evidence and results',expanded=gate==1):
        for k in ['audit','prepared','literature','review']:
            try:
                data=rp.inspect_stage(root,k,0,30)
                with st.expander(k): st.json(data,expanded=False)
            except ValueError: pass
        if s['tables']:
            t=st.selectbox('Native result table',list(s['tables']),format_func=lambda x:x+' — '+s['tables'][x]['name'],key='_v6_pipeline_table'); v=rp.evidence(root,t); st.dataframe(pd.DataFrame(v['rows'],columns=v['columns']),hide_index=True); st.caption(f"First {len(v['rows'])} of {v['total_rows']} rows. Complete JSON/CSV in checkpoint.")
        paths=[x['path'] for x in s['artifacts'] if x['path'].endswith(('.zip','.docx','.md','.csv','.json','.pdf','.png','.svg','.html','.parquet'))]
        if paths:
            selected=st.selectbox('Choose native file to download',paths,key='_v6_pipeline_artifact'); st.download_button('Download selected file',(root/selected).read_bytes(),Path(selected).name)
    if gate<=6:
        st.subheader(f'{gate}. {rp.GATES[gate]}')
        if s['status']=='blocked': st.error(s['history'][-1].get('error','Gate blocked')); st.caption('Revoke pending approval and correct the configuration; no gate was advanced.')
        if s['pending']:
            st.info(f"Approved configuration · {s['pending']['reviewer']} · {s['pending']['approval_id'][:12]}"); st.json(s['pending']['config'],expanded=False)
            a,b=st.columns(2)
            if a.button('Execute approved stage',type='primary'):
                try: rp.advance(root); st.session_state.pop('_v6_pipeline_archive',None); st.rerun()
                except Exception as e: st.error(str(e))
            if b.button('Revoke approval / edit'):
                rp.revoke(root); st.session_state.pop('_v6_pipeline_archive',None); st.rerun()
        else:
            template=make_template(s,root); text=json.dumps(template,ensure_ascii=False,indent=2)
            if gate==1: st.write('Review the independent audit. Specify every source’s cleaning options and explicit integration mode. Empty cleaning object means trim and normalise headers only; duplicates remain. No silent imputation.')
            if gate==2: st.write('Use the extracted PDF page / DOCX block evidence. Supply grouped research questions and hypotheses with decision rules; record limitations of literature coverage.')
            if gate==3: st.write('Choose native methods and exact parameters. Copy the schema into your LLM or edit the JSON. Explicitly approve seeds, sample rules, multiplicity and methods. Unsupported analyses remain in their original modules.')
            if gate==4: st.write('Supply your LLM-assisted or human manuscript and mapped RQ answers, hypotheses and numerical claim register. The runner compares listed numeric claims with actual native table cells; prose and citations still need review.')
            if gate==6: st.write('Resolve each review row. Optional shareable_paths are selected artifacts only, after disclosure review; no automated anonymisation. Empty list produces only a private evidence package.')
            with st.expander('Native method schema / parameter guidance',expanded=gate==3): st.json(rp.schema(),expanded=False)
            st.download_button('Download editable stage template (JSON)',text,f'gate_{gate}_template.json','application/json'); st.download_button('Download native method schema (JSON)',rp.jb(rp.schema()),'pipeline_schema.json','application/json')
            uploaded=st.file_uploader('Optional: upload completed configuration from your LLM (JSON)',type=['json'],key=f'_v6_pipeline_config_file_{s["run_id"]}_{gate}_{s["revision"]}')
            txt=st.text_area('Review / edit proposed configuration',value=uploaded.getvalue().decode('utf-8') if uploaded else text,height=300,key=f'_v6_pipeline_config_{s["run_id"]}_{gate}_{s["revision"]}_{uploaded.name if uploaded else "manual"}')
            reviewer=st.text_input('Approving researcher / reviewer',key='_v6_pipeline_reviewer'); confirmation=st.checkbox('I have reviewed the exact configuration and authorise this stage',key=f'_v6_pipeline_confirm_{gate}_{s["revision"]}')
            if st.button('Record approval',type='primary',disabled=not confirmation or not reviewer.strip()):
                try: rp.approve(root,gate,json.loads(txt),reviewer); st.session_state.pop('_v6_pipeline_archive',None); st.rerun()
                except Exception as e: st.error(str(e))
    else: st.success('Candidate package prepared. Final scientific, citation, confidentiality and editorial review remains mandatory; no claim of journal acceptance.')
    if gate in [5,6,7] and not s['pending']:
        with st.expander('Revise manuscript while preserving computed evidence'):
            reason=st.text_input('Revision rationale',key='_v6_pipeline_revision_reason'); who=st.text_input('Reviewer',key='_v6_pipeline_revision_who')
            if st.button('Open new manuscript revision',disabled=not reason.strip() or not who.strip()):
                rp.revise(root,reason,who); st.session_state.pop('_v6_pipeline_archive',None); st.rerun()
    st.divider(); confirm=st.checkbox('I have downloaded my checkpoint or no longer need this session',key='_v6_pipeline_delete')
    if st.button('Close / delete temporary research session',disabled=not confirm):
        holder.cleanup();
        for k in list(st.session_state):
            if k.startswith('_v6_pipeline_'): del st.session_state[k]
        st.rerun()
