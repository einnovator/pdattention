# Paper 8.5 builds

The publication manuscript and the auditable technical report share one source
of scientific claims.

```powershell
cd docs/papers/paper8_5
latexmk -pdf -interaction=nonstopmode -halt-on-error paper_8_5_draft.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error paper_8_5_report.tex
```

- `paper_8_5_draft.pdf` is the concise publication edition.
- `paper_8_5_report.pdf` defines `\PaperEightFiveTechnicalReport` and includes
  the detailed chronology, rejected policies, diagnostics, and evidence ledger.

Headline numerical claims must be updated in `paper_8_5_draft.tex`; do not edit
the report wrapper independently.  Rebuild and visually inspect both PDFs after
every material evidence update.
