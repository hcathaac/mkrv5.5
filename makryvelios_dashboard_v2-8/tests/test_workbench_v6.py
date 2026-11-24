import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
import pytest
from workbench_v6 import audit,clean,loan_forecast,loan_schedule,bibliography,recommend,bundle

def test_clean_and_audit():
    a=pd.DataFrame({' Amount ':[' 2,5 ',' 2,5 ','NA'],'ID':['001','001','002']})
    b,log=clean(a,remove_duplicates=True,missing_tokens=('NA',),numeric_columns=('Amount',),decimal_comma=True)
    assert len(b)==2 and b.Amount.iloc[0]==2.5 and b.ID.iloc[0]=='001' and a.shape==(3,2)
    assert audit(b)['fields'][0]['missing']==1 and len(log)>=3

def test_no_silent_coercion():
    with pytest.raises(ValueError): clean(pd.DataFrame({'x':['oops']}),numeric_columns=('x',))

def test_loan():
    f=pd.DataFrame({'year':range(2000,2020),'amount':np.arange(20)*3+100})
    r=loan_forecast(f,'year','amount','Y',3)
    assert r['model']=='linear_trend' and r['forecast'].forecast.iloc[0]==pytest.approx(160)
    with pytest.raises(ValueError): loan_forecast(f.drop(4),'year','amount','Y',3)

def test_schedule():
    for rate in (0,.03):
        r=loan_schedule(100000,rate,10,12,6)
        assert r.principal_repaid.sum()==pytest.approx(100000) and r.outstanding.iloc[-1]==pytest.approx(0,abs=1e-6)

def test_bib():
    b=bibliography([{'title':'X','DOI':'10.1/A'},{'title':'Y','DOI':'https://doi.org/10.1/a'}]); assert b.duplicate_candidate.tolist()==[False,True]
    assert recommend('loan forecasting')[0][1].startswith('15.')
    assert bundle({'b':b},{'x':1}).startswith(b'PK')


def test_docx_block_provenance():
    from docx import Document
    from io import BytesIO
    from research_chair import extract_pdf_collection
    d=Document(); d.add_paragraph('Public funding methodology'); t=d.add_table(rows=1,cols=2); t.cell(0,0).text='Amount'; t.cell(0,1).text='100'
    b=BytesIO(); d.save(b)
    pages=extract_pdf_collection((('methods.docx',b.getvalue()),))
    assert len(pages)==2 and 'DOCX block, not page' in pages.document.iloc[0]
    assert pages.text.iloc[1]=='Amount | 100'


def test_connector_audits_three_sources(tmp_path,monkeypatch):
    from workbench_connector import run_research
    for i in range(1,4):
        pd.DataFrame({'x':range(1,21),'y':[2*x+i for x in range(1,21)]}).to_csv(tmp_path/f'dataset{i}.csv',index=False)
    monkeypatch.setenv('MKRV_DATA_DIR',str(tmp_path))
    files=[f'dataset{i}.csv' for i in range(1,4)]
    check=run_research('OLS regression',files,'',{})
    assert check['status']=='mapping_required' and len(check['audits'])==3
    result=run_research('OLS regression',files,check['available_sources'][0],{'outcome':'y','predictors':['x']})
    assert len(result['audits'])==3 and result['tables']
    with pytest.raises(ValueError): run_research('OLS',['../outside.csv'],'',{})
