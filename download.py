import requests
from pathlib import Path
from enum import IntEnum

class EdinetDocType(IntEnum):
    MAIN_AND_AUDIT = 1
    PDF = 2
    ATTACHMENT = 3
    ENGLISH = 4
    CSV = 5

API_KEY = "b053b9de6579485d94409e03743f16c4"
docID = "S1006MW7"

docType = EdinetDocType.CSV

url = f"https://api.edinet-fsa.go.jp/api/v2/documents/{docID}"
params = {
    "type": docType,
    "Subscription-Key": API_KEY
}

res = requests.get(url, params=params, stream=True)
res.raise_for_status()

# Content-Type から拡張子を決定
content_type = res.headers.get("Content-Type", "")
if "pdf" in content_type:
    suffix = ".pdf"
elif "zip" in content_type or "octet-stream" in content_type:
    suffix = ".zip"
elif "json" in content_type:
    suffix = ".json"
else:
    suffix = ".bin"

out = Path(f"{docID}_{docType}_{suffix}")

with out.open("wb") as f:
    for chunk in res.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print(f"saved: {out.resolve()}")
