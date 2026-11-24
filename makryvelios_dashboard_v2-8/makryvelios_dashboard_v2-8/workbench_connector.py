"""Optional local MCP server. Set MKRV_DATA_DIR to an explicit authorised directory.
Install requirements and `mcp`, then run `python workbench_connector.py` (stdio).
Never exposes shell, arbitrary paths, credentials, or arbitrary Python execution.
"""
from pathlib import Path
import os, json, hashlib, hmac, logging, time, secrets
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

def build_server(remote=False):
    from mcp.server.fastmcp import FastMCP
    mcp=FastMCP('Makryvelios research workbench',stateless_http=True,json_response=True) if remote else FastMCP('Makryvelios research workbench')
    @mcp.tool()
    def research(question: str, datasets: list[str], active: str='', roles: dict|None=None, documents: list[str]|None=None) -> dict:
        """Audit every supplied dataset, then execute existing bounded Research Chair routes on a selected source. Call once with empty active to discover mapped source names."""
        if len(question)>12000: raise ValueError('Question too long')
        if len(documents or [])>12: raise ValueError('Maximum 12 supporting documents')
        result=run_research(question,datasets,active,roles or {},documents)
        result['request_id']=secrets.token_hex(12)
        logging.getLogger('mkrv.audit').info('request=%s status=%s sources=%s dataset_sha256=%s',result['request_id'],result['status'],len(datasets),result.get('dataset_sha256','not selected'))
        return result
    # Separate opt-in private single-run directory. No tool can record human approval.
    def pipeline_root():
        raw=os.environ.get('MKRV_PIPELINE_DIR')
        if not raw: raise ValueError('Administrator must set MKRV_PIPELINE_DIR to the authorised private run directory')
        return Path(raw).resolve(strict=True)
    @mcp.tool()
    def pipeline_schema() -> dict:
        """Read supported native methods and parameters; no computation."""
        from research_pipeline import schema
        return schema()
    @mcp.tool()
    def pipeline_status() -> dict:
        """Read the current authorised pipeline state, artifact index and approval status."""
        from research_pipeline import public_status
        return public_status(pipeline_root())
    @mcp.tool()
    def pipeline_execute_approved() -> dict:
        """Execute only the next already approved stage. Approval is recorded separately by the researcher via local UI/CLI; no tool may approve."""
        from research_pipeline import advance,public_status
        root=pipeline_root(); advance(root); return public_status(root)
    @mcp.tool()
    def pipeline_evidence(table_id: str, offset: int=0, limit: int=100) -> dict:
        """Read bounded pages of the actual native result table. May contain private evidence."""
        from research_pipeline import evidence
        return evidence(pipeline_root(),table_id,offset,limit)
    @mcp.tool()
    def pipeline_inspect(kind: str, offset: int=0, limit: int=30) -> dict:
        """Read authorised audit, prepared schema, literature locator or editorial review evidence; no arbitrary path."""
        from research_pipeline import inspect_stage
        return inspect_stage(pipeline_root(),kind,offset,limit)
    return mcp

class BearerGuard:
    """Single-tenant ASGI guard. Reverse proxy must terminate HTTPS and limit rates."""
    def __init__(self,app,token):
        if len(token)<32: raise ValueError('MKRV_MCP_TOKEN must be at least 32 characters')
        self.app=app; self.token=('Bearer '+token).encode()
    async def __call__(self,scope,receive,send):
        if scope['type']=='http':
            headers=dict(scope.get('headers',[])); auth=headers.get(b'authorization',b'')
            if not hmac.compare_digest(auth,self.token):
                await send({'type':'http.response.start','status':401,'headers':[(b'www-authenticate',b'Bearer'),(b'content-type',b'text/plain')]})
                await send({'type':'http.response.body','body':b'Unauthorized'}); return
        await self.app(scope,receive,send)

def remote_app():
    if not os.environ.get('MKRV_DATA_DIR'): raise ValueError('Set isolated MKRV_DATA_DIR')
    Path(os.environ['MKRV_DATA_DIR']).resolve(strict=True)
    token=os.environ.get('MKRV_MCP_TOKEN','')
    if len(token)<32: raise ValueError('Set a strong MKRV_MCP_TOKEN (at least 32 characters)')
    logging.basicConfig(level=logging.INFO)
    return BearerGuard(build_server(remote=True).streamable_http_app(),token)

def main():
    build_server().run(transport='stdio')
if __name__=='__main__': main()
