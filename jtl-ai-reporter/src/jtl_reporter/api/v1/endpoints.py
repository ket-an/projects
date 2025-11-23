from fastapi import APIRouter, UploadFile, File, HTTPException
from jtl_reporter.jtl_reporter import parse_jtl

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/upload-jtl")
async def upload_jtl(file: UploadFile = File(...)):
    # Validate file type
    if not (file.filename.endswith(".jtl") or file.content_type in ["text/xml", "text/csv"]):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload a .jtl file.")

    # Read file bytes
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty JTL file provided.")

    # Call parser
    try:
        result = parse_jtl(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse JTL: {str(e)}")

    return {
        "file_name": file.filename,
        "parsed_result": result
    }



