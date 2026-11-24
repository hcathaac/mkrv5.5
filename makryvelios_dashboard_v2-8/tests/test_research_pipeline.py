import io,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,pytest
import research_pipeline as rp

def inputs():
    rng=np.random.default_rng(24); x=rng.normal(size=70); f=pd.DataFrame({'id':range(70),'x':x,'y':2*x+rng.normal(size=70),'region':['EL30','EL51']*35}); b=io.BytesIO(); from docx import Document
    d=Document(); d.add_paragraph('A methodological note about observational association.'); d.save(b)
    return [('A.csv',f.to_csv(index=False).encode()),('B.csv',f[['id','region']].to_csv(index=False).encode()),('C.csv',f[['id','x']].to_csv(index=False).encode()),('notes.docx',b.getvalue())]

def g1(s): return {'cleaning':{n:{} for n in s['sources']},'integration':{'mode':'separate','sources':['A.csv'],'reason':'Separate observation units; no join needed.'},'missingness_policy':'Native complete-case analysis','review_note':'Checked each dataset and retained original data.'}
def g2(): return {'research_questions':[{'id':'RQ1','group':'Associational','question':'Association between x and y?','type':'associational'}],'hypotheses':[{'id':'H1','rq_id':'RQ1','statement':'Slope differs from zero','decision_rule':'Assess HC3 two-sided p-value and multiplicity.'}],'source_notes':[{'document':'notes.docx [DOCX block, not page]','page_or_block':1,'excerpt':'observational association','interpretation':'Do not infer causality.'}],'gap':'Tentative association question','literature_limitations':'Single supplied note; novelty not established.'}
def g3(): return {'analyses':[{'id':'A1','method':'ols','params':{'y':'y','x_vars':['x'],'covariance':'HC3'},'rq_ids':['RQ1'],'reason':'Continuous outcome','assumptions':'Independent observations and linear association; no causal claim.','figures':False},{'id':'A2','method':'monte_carlo_ols','params':{'y':'y','x_vars':['x'],'simulations':150,'method':'Wild bootstrap','confidence':.95,'seed':42},'rq_ids':['RQ1'],'reason':'Uncertainty','assumptions':'Independent observations; heteroskedasticity sensitivity','figures':False}],'multiplicity':'Benjamini–Hochberg','scope_note':'One active dataset; other audited files remain separate.'}
def step(root,g,c): rp.approve(root,g,c,'Researcher'); return rp.advance(root)

def test_end_to_end_and_revision(tmp_path):
    root=tmp_path/'run'; s=rp.new_run(root,inputs(),'Estimate x and y association'); assert s['next_gate']==1 and len(s['sources'])==3
    with pytest.raises(ValueError,match='approval'): rp.advance(root)
    with pytest.raises(ValueError,match='out of order'): rp.approve(root,3,g3(),'R')
    step(root,1,g1(s)); assert rp.inspect_stage(root,'literature')['total']==1
    step(root,2,g2()); s=step(root,3,g3()); assert s['next_gate']==4 and 'A1.01' in s['tables']; assert rp.evidence(root,'A2.02')['total_rows']==150
    t=rp.evidence(root,'A1.01'); row=1; val=t['rows'][row][t['columns'].index('coefficient')];
    c={'markdown':'# Research question\n\nMethods: OLS. Results and confidence interval. Limitations: association. References and reproducibility. Ethics.', 'rq_answers':[{'id':'RQ1','status':'answered','answer':'Association estimated; see coefficient table.','evidence_ids':['A1.01']}],'hypothesis_assessments':[{'id':'H1','verdict':'consistent_with','reason':'Review estimate and diagnostics','evidence_ids':['A1.01']}],'numeric_claims':[{'table_id':'A1.01','row':row,'column':'coefficient','value':round(val,5),'decimals':5,'text':'Slope estimate'}],'bibliography':[],'originality_note':'Not verified independently.'}
    bad=json.loads(json.dumps(c)); bad['numeric_claims'][0]['value']=999
    rp.approve(root,4,bad,'R');
    with pytest.raises(ValueError,match='does not match'): rp.advance(root)
    assert rp.status(root)['next_gate']==4; rp.revoke(root); step(root,4,c); s=step(root,5,{'review_scope':'Structural review'}); assert s['review_count']>5
    rp.revise(root,'Improve reporting','R'); assert rp.status(root)['next_gate']==4; step(root,4,c); s=step(root,5,{'review_scope':'Recheck revised article'});
    s=step(root,6,{'issues':[{'row':i,'status':'accepted_limitation','resolution':'Human review remains required'} for i in range(s['review_count'])],'human_review_note':'Candidate only','reference_review':'Verify original references','data_sharing_note':'No raw data for publisher','shareable_paths':['04_manuscript_r1/manuscript.docx']}); assert s['next_gate']==7 and s['human_review_required']
    raw=rp.checkpoint(root); r=tmp_path/'restored'; s2=rp.restore(r,raw); assert s2['execution_sha256']==s['execution_sha256']
    with zipfile.ZipFile(r/'06_release_r1/SELECTED_FOR_PUBLISHER.zip') as z: assert '04_manuscript_r1/manuscript.docx' in z.namelist() and not any(n.startswith('inputs/') for n in z.namelist())
    (r/'inputs/A.csv').write_bytes(b'altered')
    with pytest.raises(ValueError,match='Checksum'): rp.status(r)

def test_join_and_parameter_guards(tmp_path):
    root=tmp_path/'run'; s=rp.new_run(root,inputs(),'test'); c=g1(s); c['integration']={'mode':'join','sources':['A.csv','B.csv'],'keys':['region'],'how':'left','reason':'Test duplicate keys'}
    rp.approve(root,1,c,'R');
    with pytest.raises(ValueError,match='one-to-one'): rp.advance(root)
    rp.revoke(root); step(root,1,g1(s)); step(root,2,g2()); p=g3(); p['analyses'][1]['params']['simulations']=20001; rp.approve(root,3,p,'R')
    with pytest.raises(ValueError,match='Monte Carlo parameters'): rp.advance(root)
    s=rp.status(root); assert s['next_gate']==3 and not s['tables']; rp.revoke(root)
    p=g3(); p['analyses'][0]['method']='eval'; rp.approve(root,3,p,'R')
    with pytest.raises(ValueError,match='Unsupported analysis'): rp.advance(root)

def test_map_and_seed():
    f=pd.DataFrame({'region':['EL30','EL51'],'amount':[2.,3.]}); t,_,_=rp.execute_method(f,'greece_map',{'geography':'region','metric':'amount','aggregation':'Sum','level':'NUTS 2 – Regions (13)'},False); assert t['Mapped values'].amount.sum()==5
    f.loc[0,'region']='unmatched';
    with pytest.raises(ValueError,match='Unmatched'): rp.execute_method(f,'greece_map',{'geography':'region','metric':'amount','aggregation':'Sum','level':'NUTS 2 – Regions (13)'},False)
    p={'budget':1000.,'absorption':.8,'budget_shock':0.,'absorption_sd':.1,'draws':100,'seed':21}; a,_,_=rp.execute_method(f,'funding_scenario',p,False); b,_,_=rp.execute_method(f,'funding_scenario',p,False); pd.testing.assert_frame_equal(a['Simulation draws'],b['Simulation draws'])

def test_more_native_routes():
    rng=np.random.default_rng(56); n=80; z=rng.normal(size=n); x=z+rng.normal(size=n); y=x+rng.normal(size=n); f=pd.DataFrame({'x':x,'y':y,'z':z,'cost':np.ones(n)*10,'benefit':np.abs(y),'treatment':np.tile([0,0,1,1],20),'post':np.tile([0,1,0,1],20),'year':np.arange(n),'id':np.arange(n)})
    plans={'iv_2sls':{'y':'y','endogenous':'x','instruments':['z'],'exogenous':[]},'did':{'y':'y','treatment':'treatment','post':'post','controls':['x'],'cluster':None},'regularised':{'y':'y','x_vars':['x','z'],'method':'Ridge','alpha':1.,'l1_ratio':.5},'portfolio_mc':{'cost':'cost','benefit':'benefit','budget':100.,'project_id':'id','simulations':100,'cost_cv':.1,'benefit_cv':.2,'correlation':0.,'seed':42},'clustering':{'columns':['x','y'],'method':'K-means','clusters':3,'scaling':'Standard (z-score)','automatic_k':False,'max_k':4,'linkage':'ward','eps':.7,'min_samples':5,'seed':42},'arima':{'variable':'y','time_col':'year','order':[1,0,0],'steps':3},'pca':{'columns':['x','y'],'n_components':2}}
    for method,p in plans.items():
        tables,_,_=rp.execute_method(f,method,p,False); assert any(not t.empty for t in tables.values()),method
    f.loc[0,'year']=999
    with pytest.raises(ValueError,match='regular time'): rp.execute_method(f,'arima',plans['arima'],False)
