import json
import math
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from app import semantic

client=TestClient(semantic.app)
seed=json.loads(Path(__file__).with_name('seed-data.json').read_text())
vec=seed[0]['embedding']

@pytest.mark.parametrize('path,status',[('/',307),('/semantic',200),('/missing',404)])
def test_pages(path,status):
    assert client.get(path,follow_redirects=False).status_code==status

def test_html_honest():
    html=client.get('/semantic').text
    for text in ['No generative LLM is connected','AI-assisted portfolio demo','Synthetic data only','read-only RPC']:
        assert text in html

def test_live_health():
    r=client.get('/health');assert r.status_code==200
    assert r.json()['database']['documents']==8
    assert r.json()['database']['dimensions']==384

def test_live_exact_vector():
    r=client.post('/query',json={'question':'Where can I recover access?','embedding':vec})
    assert r.status_code==200
    data=r.json();assert data['citations'][0]['id']==1
    assert data['citations'][0]['score']>.999
    assert 'not an LLM' in data['answer_provider']
    for row in data['citations']:
        assert row['content']==next(d['text'] for d in seed if d['id']==row['id'])

@pytest.mark.parametrize('vector',[[0]*384,[1]*384,[1],[0]*385,[float('inf')]*384])
def test_reject_bad_vector(vector):
    with pytest.raises(ValidationError):semantic.Query(question='Valid question',embedding=vector)

@pytest.mark.parametrize('question',['','ab','a'*301])
def test_reject_bad_question(question):
    assert client.post('/query',json={'question':question,'embedding':vec}).status_code==422

def test_body_size_limit():
    assert client.post('/query',content='x'*20001,headers={'Content-Type':'application/json'}).status_code==413

def test_security_headers():
    r=client.get('/semantic')
    assert r.headers['x-frame-options']=='DENY'
    assert r.headers['x-content-type-options']=='nosniff'

def test_no_write_route():
    assert client.post('/documents',json={'text':'sample'}).status_code==404

def test_backend_never_substitutes_memory(monkeypatch):
    def broken(*a,**kw):raise RuntimeError('database unavailable')
    monkeypatch.setattr(semantic,'rpc',broken)
    r=client.post('/query',json={'question':'Valid question','embedding':vec})
    assert r.status_code==503
    assert client.get('/health').status_code==503

def test_threshold_honest():
    result=semantic.ground({'rows':[{'score':0.1,'content':'not related'}]})
    assert result['citations']==[]
    assert 'No sufficiently close' in result['answer']

def test_direct_table_not_exposed():
    import httpx
    r=httpx.get(semantic.PUBLIC_URL+'/rest/v1/passages?select=id',headers={'apikey':semantic.PUBLIC_PUBLISHABLE_KEY})
    assert r.status_code in {401,403,404}
