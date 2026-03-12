from fastapi import FastAPI

app = FastAPI(title="Multivector RAG")

@app.get("/health")
def health():
    return {"status": "ok"}