import io, sys, zipfile, asyncio, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from research_platform import validation, funding_simulation, project_pack, project_read, peer_review, publication_pack, journal_search, collect_tables, _safe_zip


def test_validation_and_rank():
    rng=np.random.default_rng(22); x=rng.normal(size=80); d=pd.DataFrame({'x':x,'y':2*x+rng.normal(size=80),'time':range(80)})
    out=validation({'A':d,'B':d.copy(),'C':d.copy()},d,'y',['x'],'time')
    reg=out['Validation register']; assert set(reg.source)>={'A','B','C'}
    assert 'OLS covariance sensitivity' in out and 'VIF' in out
    assert reg.loc[reg.check=='Ramsey RESET specification','status'].iloc[0] in ('pass','review')
    assert reg.loc[reg.check=='Endogeneity / identification','status'].iloc[0]=='not assessed'
    d['x2']=2*x; bad=validation({'A':d},d,'y',['x','x2'])['Validation register']; assert 'blocked' in bad.status.values


def test_simulation_conservation_and_seed():
    r=pd.DataFrame({'region':['a','b'],'weight':[1.,3.]}); a=funding_simulation(1000,.7,.1,.2,1000,11,r); b=funding_simulation(1000,.7,.1,.2,1000,11,r)
    pd.testing.assert_frame_equal(a['Simulation draws'],b['Simulation draws']); t=a['Simulation draws']; assert (t.absorbed.between(0,1100.000001)).all(); assert np.allclose(t.absorbed+t.unspent,t.budget)
    assert a['Regional scenario allocation'].budget.sum()==pytest.approx(1100)
    with pytest.raises(ValueError): funding_simulation(100,.5,0,.2,100,1,pd.DataFrame({'region':['a'],'weight':[-1.]}))


def test_project_roundtrip():
    pytest.importorskip('pyarrow'); d=pd.DataFrame({'id':['001','002'],'amount':[3.5,np.nan],'time':pd.to_datetime(['2020-01-01','2021-01-01'])})
    raw=project_pack('test',{'source':d},'question',{'result':d}, {'note.txt':b'hello'}, {'active_source':'source','api_key':'NEVER_INCLUDE'})
    m,content=project_read(raw); pd.testing.assert_frame_equal(content['datasets']['source'],d); assert content['documents']['note.txt']==b'hello'; assert 'api_key' not in m['settings']
    b=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as old,zipfile.ZipFile(b,'w') as z:
        for n in old.namelist(): z.writestr(n, b'bad' if n.startswith('datasets/') else old.read(n))
    with pytest.raises(ValueError,match='checksum'): project_read(b.getvalue())


def test_unsafe_archive():
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:z.writestr('../x','bad')
    with pytest.raises(ValueError,match='Unsafe'): _safe_zip(b.getvalue())


def test_review_and_export():
    text='Research question: test. Methods: regression. Results p=0.03. Limitations: sampling.'; table=pd.DataFrame({'y':[1,2],'label':['=unsafe','ok']}); review=peer_review(text,{'test':table}); assert len(review)>=9 and not any(review.status.str.contains('accepted'))
    assert collect_tables({'model_output':{'tables':table},'llm_config':{'api_key':'x'}}).keys()=={'model_output / tables'}
    raw=publication_pack({'test':table},question='=formula');
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert {'tables.xlsx','research_evidence.docx','tables/000.csv','tables/000.tex','manifest.json'}<=set(z.namelist()); assert b"'=unsafe" in z.read('tables/000.csv')
        from openpyxl import load_workbook
        w=load_workbook(io.BytesIO(z.read('tables.xlsx'))); assert w['Table_000']['B2'].data_type!='f'; assert w['Readme']['A2'].data_type!='f'


def test_journal_provider_parsing():
    class R:
        def __init__(self,j): self.j=j
        def json(self): return self.j
        def raise_for_status(self): pass
    class S:
        def get(self,url,**kw):
            if 'crossref' in url: return R({'message':{'items':[{'container-title':['Journal X'],'ISSN':['1234-5678'],'DOI':'10.1/x'}]*2}})
            if 'openalex' in url: return R({'results':[{'is_in_doaj':True,'is_oa':True,'apc_usd':500}]})
            return R({'results':[{'id':'example'}]})
    t=journal_search('topic','user-key',session=S(),check_doaj=True); assert t.iloc[0].sample_matches==2 and t.iloc[0].DOAJ_direct=='record found' and t.iloc[0].OpenAlex_DOAJ


def test_bearer_guard():
    from workbench_connector import BearerGuard
    async def inner(scope,receive,send): await send({'type':'http.response.start','status':200,'headers':[]})
    async def check(auth):
        got=[]
        async def send(x): got.append(x)
        await BearerGuard(inner,'a'*32)({'type':'http','headers':[(b'authorization',auth)]},None,send); return got[0]['status']
    assert asyncio.run(check(b'Bearer bad'))==401; assert asyncio.run(check(b'Bearer '+b'a'*32))==200
    with pytest.raises(ValueError): BearerGuard(inner,'short')
