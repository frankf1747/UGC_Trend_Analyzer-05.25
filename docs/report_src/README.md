# Report source

`report_v3.html` is the whole three-page report, self-contained: the cover is the
first card from the project's own thumbnail design file, reused verbatim, and the
Anton / Archivo / IBM Plex Mono faces are embedded as base64 woff2 so it renders
identically with no network access.

Rebuild:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer --virtual-time-budget=15000 \
  --print-to-pdf=../UGC_Trend_Analyzer_Report.pdf report_v3.html
```

Page size is 1200×627 — the cover card's own dimensions — set via `@page`.
