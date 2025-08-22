import hashlib, re, requests, csv, io
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text

def _make_id(source, text):
    return f"{source}:{hashlib.md5(text.encode('utf-8')).hexdigest()}"

def fetch_html_list(url, source_name, li_selectors=("li","p")):
    html = requests.get(url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    lines = []
    for sel in li_selectors:
        for el in soup.select(sel):
            t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
            if t and len(t) > 6:
                lines.append({
                    "id": _make_id(source_name, t),
                    "text": t,
                    "source": source_name
                })
    return lines

def fetch_pdf_lines(url, source_name):
    pdf_bytes = requests.get(url, timeout=60).content
    text = extract_text(io.BytesIO(pdf_bytes))
    # split on bullet/line boundaries
    chunks = [re.sub(r"\s+", " ", s).strip() for s in re.split(r"[\n•-–]+", text)]
    items = []
    for t in chunks:
        if t and len(t) > 6:
            items.append({"id": _make_id(source_name, t), "text": t, "source": source_name})
    return items

def load_csv(path, text_col="text", source_name="kaggle_csv"):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = (r.get(text_col) or "").strip()
            if t and len(t) > 6:
                rows.append({"id": _make_id(source_name, t), "text": t, "source": source_name})
    return rows
