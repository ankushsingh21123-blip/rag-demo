"""AI-assisted, synthetic, read-only semantic retrieval proof.
Local browser embeddings. LangChain retrieval chain. Real PostgreSQL/pgvector.
No hosted LLM, paid API, user accounts, private data or independent-work claim.
"""
import math
import os
from pathlib import Path
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title='Semantic Support Atlas')
MODEL = 'Xenova/all-MiniLM-L6-v2'
# Public publishable configuration, not a database password or admin key.
PUBLIC_URL = 'https://vvsyhsdzbvmnzxmauxjq.supabase.co'
PUBLIC_PUBLISHABLE_KEY = 'sb_publishable_pXCYOuFAsdQEu8x6xV2g8g_KnnSKy5Q'

class Query(BaseModel):
    question: str = Field(min_length=3, max_length=300)
    embedding: list[float] = Field(min_length=384, max_length=384)
    @field_validator('embedding')
    @classmethod
    def valid_vector(cls, vector):
        if any(not math.isfinite(v) or abs(v) > 1.001 for v in vector):
            raise ValueError('A finite normalised 384-dimensional vector is required.')
        if not 0.98 < sum(v*v for v in vector) < 1.02:
            raise ValueError('Vector must have unit length.')
        return vector

def rpc(name, payload):
    base = os.environ.get('SUPABASE_URL', PUBLIC_URL)
    key = os.environ.get('SUPABASE_PUBLISHABLE_KEY', PUBLIC_PUBLISHABLE_KEY)
    if not base.startswith('https://') or not base.endswith('.supabase.co') or not key:
        raise RuntimeError('Storage is not configured.')
    with httpx.Client(timeout=12) as client:
        response = client.post(base+'/rest/v1/rpc/'+name, json=payload,
                               headers={'apikey':key, 'Content-Type':'application/json'})
    response.raise_for_status()
    return response.json()

def retrieve(query):
    rows = rpc('semantic_atlas_search', {'query_embedding':query.embedding, 'result_limit':3})
    if not isinstance(rows, list):
        raise RuntimeError('Invalid storage response.')
    return {'question':query.question, 'rows':rows}

def ground(result):
    # Retrieval similarity is not answer confidence. A threshold is only a
    # conservative demo heuristic, not a calibrated factuality guarantee.
    relevant=[r for r in result['rows'] if float(r['score']) >= 0.30]
    return {
        'answer': ' '.join(f'[{i}] {r["content"]}' for i,r in enumerate(relevant,1)) if relevant else 'No sufficiently close passage was found. Try a question about the fictional help documents.',
        'citations': relevant,
        'answer_provider':'extractive quotations, not an LLM',
        'embedding_provider':MODEL+' (q8, browser-local)',
        'store':'PostgreSQL + pgvector, read-only synthetic corpus',
        'chain':'LangChain RunnableLambda retrieval and citation pipeline',
    }

chain = RunnableLambda(retrieve) | RunnableLambda(ground)

@app.middleware('http')
async def boundaries(request: Request, call_next):
    size=request.headers.get('content-length','0')
    if not size.isdigit() or int(size)>20000:
        return HTMLResponse('Request too large.',413)
    if request.method == 'POST' and len(await request.body()) > 20000:
        return HTMLResponse('Request too large.',413)
    response=await call_next(request)
    response.headers.update({'X-Content-Type-Options':'nosniff','X-Frame-Options':'DENY','Referrer-Policy':'no-referrer','Cache-Control':'no-store','Permissions-Policy':'camera=(), microphone=(), geolocation=()'})
    return response

@app.get('/', include_in_schema=False)
def root():
    return RedirectResponse('/semantic',307)

@app.get('/semantic', response_class=HTMLResponse, include_in_schema=False)
def page():
    return Path(__file__).with_name('semantic-ui.html').read_text()

@app.get('/health')
def health():
    try:
        status=rpc('semantic_atlas_health', {})
        return {'status':'ok','database':status,'model':MODEL,'chain':'LangChain','answer_provider':'extractive, no LLM'}
    except Exception:
        raise HTTPException(503, 'Synthetic vector store unavailable.')

@app.post('/query')
def query(query:Query):
    try:
        return chain.invoke(query)
    except Exception:
        raise HTTPException(503, 'Retrieval unavailable. No generated answer was substituted.')
