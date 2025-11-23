from fastapi import APIRouter, UploadFile, File, HTTPException
from jtl_reporter.jtl_reporter import parse_and_report_from_bytes, parse_jtl
from jtl_reporter.summarizer import build_ai_summary
import tempfile

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


# ------------------------------
# 1) Upload JTL + Full Report + AI Summary
# ------------------------------
@router.post("/upload-jtl")
async def upload_jtl(file: UploadFile = File(...)):
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        # Parse & build raw report (existing logic)
        parsed_report = parse_and_report_from_bytes(content, filename=file.filename)

        # AI summary
        ai_summary = build_ai_summary(parsed_report)

        return {
            "file_name": file.filename,
            "parsed_report": parsed_report,
            "ai_summary": ai_summary
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


# ------------------------------
# 2) Parse API (XML/CSV) + AI Summary
# ------------------------------
@router.post("/parse")
async def parse_jtl_api(file: UploadFile = File(...)):
    try:
        # Decide suffix for temp file based on content type
        suffix = ".xml" if "xml" in file.content_type else ".csv"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        parsed = parse_jtl(temp_path)
        ai_summary = build_ai_summary(parsed)

        return {
            "parsed": parsed,
            "ai_summary": ai_summary
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
