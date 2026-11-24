"""Optional local MCP server. Set MKRV_DATA_DIR to an explicit authorised directory.
Install requirements and `mcp`, then run `python workbench_connector.py` (stdio).
Never exposes shell, arbitrary paths, credentials, or arbitrary Python execution.
"""
from pathlib import Path
import os, json
import pandas as pd
from workbench_v6 import audit_all, recommend, loan_forecast, frame_hash
from analytics_core import read_tabular_bytes, tidy_frame
from research_chair import ProtocolResult, execute_natural_language_command, extract_pdf_collection

def run_research(question, datasets, active, roles, documents=None):
    if not os.environ.get('MKRV_DATA_DIR'): raise ValueError('Set MKRV_DATA_DIR to an authorised folder')
    root=Path(os.environ['MKRV_DATA_DIR']).resolve(strict=True)
    def checked(name):
        if Path(name).name!=name: raise ValueError('Only plain filenames are allowed')
        p=(root/name).resolve(strict=True)
        if p.parent!=root or not p.is_file() or p.stat().st_size>50*1024*1024: raise ValueError('Invalid file or exceeds 50 MB limit')
        return p
    if not 1<=len(datasets)<=12: raise ValueError('Supply 1–12 dataset filenames')
    frames={}
    for name in datasets:
        p=checked(name)
        if p.suffix.lower() not in ('.xlsx','.xlsm','.xls','.csv','.tsv','.parquet','.pq','.feather','.arrow'): raise ValueError('Unsupported dataset type')
        for sheet,frame in read_tabular_bytes(p.name,p.read_bytes(),all_sheets=True).items(): frames[f'{name}::{sheet}']=tidy_frame(frame,normalise_columns=True)
    checks=audit_all(frames)
    if active not in frames: return {'status':'mapping_required','audits':checks,'available_sources':list(frames),'recommendations':recommend(question)}
    frame=frames[active]; check=checks[active]
    if not len(frame) or check['duplicate_headers'] or any(x['infinite'] for x in check['fields']): return {'status':'blocked_by_audit','audits':checks,'recommendations':recommend(question)}
    allowed={'outcome','predictors','year_column','group_column','aggregation','equation'}
    if set(roles)-allowed: raise ValueError('Unsupported role key')
    for key in ('outcome','year_column','group_column'):
        if roles.get(key) and roles[key] not in frame: raise ValueError(f'Unknown {key}')
    if any(x not in frame for x in roles.get('predictors',[])): raise ValueError('Unknown predictor')
    docs=[]
    for name in documents or []:
        p=checked(name)
        if p.suffix.lower() not in ('.pdf','.docx'): raise ValueError('Unsupported supporting document')
        docs.append((p.name,p.read_bytes()))
    evidence=extract_pdf_collection(tuple(docs)); result,reply=execute_natural_language_command(frame,question,roles,ProtocolResult(algorithm='Connector request'),evidence)
    tables={name:{'total_rows':len(table),'returned_rows':min(len(table),200),'records':json.loads(table.head(200).to_json(orient='records',date_format='iso'))} for name,table in result.tables.items()}
    return {'status':'executed_router','active_source':active,'dataset_sha256':frame_hash(frame),'audits':checks,'algorithm':result.algorithm,'comments':result.comments,'reply':reply,'tables':tables,'recommendations':recommend(question),'warning':'Review feasibility and diagnostics. Returned tables capped at 200 rows. No unsupported analysis or causal claim should be inferred.'}

def main():
    from mcp.server.fastmcp import FastMCP
    mcp=FastMCP('Makryvelios research workbench')
    @mcp.tool()
    def research(question: str, datasets: list[str], active: str='', roles: dict|None=None, documents: list[str]|None=None) -> dict:
        """Audit every supplied dataset, then execute existing bounded Research Chair routes on a selected source. Call once with empty active to discover mapped source names."""
        return run_research(question,datasets,active,roles or {},documents)
    mcp.run(transport='stdio')
if __name__=='__main__': main()
