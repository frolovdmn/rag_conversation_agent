from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

app = FastAPI()

@app.get('/')
async def redirect_root_to_docs():
    return RedirectResponse('/docs')

from rag_conversation.chain import chain

add_routes(app = app, 
           runnable = chain,
           path = '/rag-conversation',
           playground_type = 'default')

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, 
                host = '0.0.0.0', 
                port = 8000)
