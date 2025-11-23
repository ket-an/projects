from fastapi import APIRouter, UploadFile, File, HTTPException
from jtl_reporter.jtl_reporter import parse_and_report_from_bytes, parse_jtl
import tempfile


router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/upload-jtl")
async def upload_jtl(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    report = parse_and_report_from_bytes(content, filename=file.filename)
    return {"file_name": file.filename, "report": report}

@router.post("/parse")
async def parse_jtl_api(file: UploadFile = File(...)):
    try:
        suffix = ".xml" if "xml" in file.content_type else ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        result = parse_jtl(temp_path)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


