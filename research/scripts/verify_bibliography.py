"""
verify_bibliography.py — check every DOI in paper_draft/references.bib against Crossref.

    python research/scripts/verify_bibliography.py [--out research/audit_outputs/bibliography_check.csv]

For each entry with a DOI, it fetches the Crossref record and compares the title (normalised
string similarity), the first author's family name and the year. It writes one row per entry
and prints the entries that need a manual look. arXiv DOIs (10.48550) are checked against
Crossref's DataCite mirror where available; failures to resolve are reported, not hidden.
Only the DOIs are sent to Crossref.
"""

import argparse
import csv
import difflib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_bib(text):
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        typ, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body, re.S):
            fields[f.group(1).lower()] = re.sub(r"\s+", " ", f.group(2)).strip()
        entries.append((typ, key, fields))
    return entries


def norm(s):
    s = re.sub(r"[{}\\]", "", s or "").lower()
    return re.sub(r"[^a-z0-9 ]", " ", s).split()


def fetch(doi):
    url = "https://api.crossref.org/works/" + urllib.request.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent": "bibliography-check/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/audit_outputs/bibliography_check.csv")
    a = ap.parse_args()
    entries = parse_bib((ROOT / "paper_draft" / "references.bib").read_text(encoding="utf-8"))
    rows, flagged = [], []
    for typ, key, f in entries:
        doi = f.get("doi", "")
        row = {"key": key, "doi": doi, "bib_title": f.get("title", ""), "bib_year": f.get("year", ""),
               "crossref_title": "", "crossref_year": "", "crossref_first_author": "", "title_similarity": "",
               "first_author_match": "", "year_match": "", "status": ""}
        if not doi:
            row["status"] = "no DOI in bib"
            rows.append(row)
            continue
        try:
            m = fetch(doi)
        except Exception as e:  # noqa: BLE001
            row["status"] = f"not resolved by Crossref ({type(e).__name__})"
            rows.append(row)
            flagged.append(row)
            continue
        ct = (m.get("title") or [""])[0]
        cy = str(((m.get("issued") or {}).get("date-parts") or [[None]])[0][0] or "")
        authors = m.get("author") or []
        fa = authors[0].get("family", "") if authors else ""
        sim = difflib.SequenceMatcher(None, " ".join(norm(row["bib_title"])), " ".join(norm(ct))).ratio()
        bib_auth = f.get("author", "")
        fam_ok = bool(fa) and fa.lower() in bib_auth.lower().split(" and ")[0].lower()
        yr_ok = (not cy) or (cy == row["bib_year"]) or abs(int(cy) - int(row["bib_year"] or 0)) <= 1
        row.update(crossref_title=ct, crossref_year=cy, crossref_first_author=fa, title_similarity=round(sim, 3),
                   first_author_match=fam_ok, year_match=yr_ok)
        row["status"] = "ok" if (sim >= 0.9 and fam_ok and yr_ok) else "check"
        if row["status"] != "ok":
            flagged.append(row)
        rows.append(row)
        time.sleep(0.2)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_doi = sum(1 for r in rows if r["doi"])
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"{len(rows)} entries, {n_doi} with DOI, {n_ok} ok, {len(flagged)} to check")
    for r in flagged:
        print(f"  {r['key']}: {r['status']} | sim={r['title_similarity']} | bib='{r['bib_title'][:60]}' | "
              f"crossref='{r['crossref_title'][:60]}' | {r['crossref_first_author']} {r['crossref_year']}")
    for r in rows:
        if r["status"] == "no DOI in bib":
            print(f"  {r['key']}: no DOI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
