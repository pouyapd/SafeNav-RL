# Submission package (prepared 2026-11-17; nothing has been submitted)

**Primary target:** *Robotics and Autonomous Systems* (Elsevier).
**Alternatives:** *Engineering Applications of Artificial Intelligence*; *Journal of Intelligent & Robotic Systems*.
See [`../q1_readiness_review.md`](../q1_readiness_review.md) §6 for the reasoning and for what was and was not verified.

| File | Content |
|---|---|
| [`../manuscript.tex`](../manuscript.tex), [`../manuscript.pdf`](../manuscript.pdf) | Manuscript: LaTeX source and PDF (33 pages, no LaTeX warnings). Blue "Author to confirm" markers show what only the author can supply |
| [`../supplementary.tex`](../supplementary.tex), [`../supplementary.pdf`](../supplementary.pdf) | Supplementary material S1–S8: configuration, recovery parameters, pre-registration record, additional round-4 tables and figures, timeout analysis, statistics, moved analyses, reproduction |
| [`../references.bib`](../references.bib), [`../figures/`](../figures/) | Bibliography (all DOIs checked against Crossref) and vector figures |
| [`abstract_and_keywords.txt`](abstract_and_keywords.txt) | Title, 250-word abstract, 6 keywords |
| [`highlights.txt`](highlights.txt) | 5 highlights, each ≤ 85 characters (RAS requires a separate editable file) |
| [`cover_letter.md`](cover_letter.md) | Draft cover letter to the RAS editor; bracketed items to complete |
| [`declarations.md`](declarations.md) | Data availability, competing interests, funding, CRediT, ethics, generative-AI use: drafts to confirm |
| [`suggested_reviewer_expertise.md`](suggested_reviewer_expertise.md) | Reviewer expertise areas (no names) |
| [`../reproducibility_checklist.md`](../reproducibility_checklist.md) | Reproduction commands and status |

## Before submitting (author actions)

1. Fill in the title-page affiliation, e-mail and ORCID. Confirm the declarations: funding, competing interests, CRediT and generative-AI use.
2. Create an archived release of the repository with a DOI (for example GitHub → Zenodo), and insert the DOI in the data availability statement and the cover letter.
3. Check the journal's JCR quartile in Clarivate if your institution requires JCR Q1. The JCR values in the review come from a third-party aggregator.
4. Read the guide for authors once more for format details. RAS's "Your Paper Your Way" policy was not verified in this pass. Export `highlights.txt` as an editable file.
5. Read the whole manuscript once yourself, then compile with `latexmk -pdf manuscript.tex` and `latexmk -pdf supplementary.tex`.
