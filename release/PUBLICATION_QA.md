# Publication QA

This note records the additional checks performed before promoting the public
release candidate. It complements, rather than replaces, the automated release
and no-download smoke checks.

## Dedicated secret scan

- Date: 2026-08-24
- Scanner: Gitleaks v8.30.1, built from its official Go module
- Configuration: default rules, no baseline, no allowlist
- Git scope: all seven public commits through `7b2a10a`
- Directory scope: the complete public working tree
- Approximate content scanned per mode: 1.29 MB
- Result: zero findings in both full-history and directory scans

The release checker also retains its smaller built-in credential-signature
screen. Neither check proves that arbitrary sensitive information is absent,
but the independent scanner materially strengthens the publication boundary.

## Manuscript PDF compile and render check

- Source: `paper/manuscript.md` with the curated Table 1, Figure 1, and Table 2
  assets under `results/v1/`
- Builder: isolated ReportLab 4.4.9 venue-neutral preview
- Renderer: Poppler
- Output: 7 US-letter pages, 444,220 bytes
- PDF SHA-256:
  `f1734f3408720f0f25c9f43f18ba1006fdc259c724af8d54db899970204a04e8`
- Structural result: all seven pages reopen and yield nonempty text; expected
  title, abstract, tables, figure, conclusion, and references are present
- Visual result: all seven rendered pages were inspected; no clipping,
  overlap, unreadable glyphs, or broken table/figure placement was observed

The generated PDF is a QA artifact and is not tracked in this repository. The
Markdown manuscript and curated assets remain the source of record. A future
venue-specific submission template will require its own compile and visual
review.
