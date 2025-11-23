from fastapi import FastAPI
from jtl_reporter.api.v1.endpoints import router as api_router

app = FastAPI(
    title="JTL AI Reporter",
    version="1.0.0",
    description="API backend for JTL file summarization and reporting"
)

# Register API routes
app.include_router(api_router, prefix="/api/v1")