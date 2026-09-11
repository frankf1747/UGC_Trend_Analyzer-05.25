# The pipeline runs once; artifacts/ is committed. `make app` needs none of it.
PY ?= python3

app:            ## open the dashboard against committed artifacts
	streamlit run app/streamlit_app.py

pipeline: 01 02 03 04 05 07 08 09   ## recompute everything (~2h, mostly stage 05)

01: ; cd pipeline && $(PY) 01_prepare.py
02: ; cd pipeline && $(PY) 02_embed.py
03: ; cd pipeline && $(PY) 03_polarity.py
04: ; cd pipeline && $(PY) 04_cluster.py
05: ; cd pipeline && $(PY) 05_assign_themes.py
07: ; cd pipeline && $(PY) 07_drivers.py
08: ; cd pipeline && $(PY) 08_language.py
09: ; cd pipeline && $(PY) 09_evidence.py

audit: ; cd pipeline && $(PY) 06_validate_labels.py --emit   ## draw a fresh label-audit sample

.PHONY: app pipeline audit 01 02 03 04 05 07 08 09
