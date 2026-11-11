"""
webreq -- request-level AI for web-attack detection, classification and prioritisation.

Where the CIC-IDS2017 experiments work on network-flow statistics, this package works on
the HTTP request itself: method, URL path, query parameters, headers and body.

  httpreq      parse raw HTTP requests (CSIC 2010 format, proxy logs, pasted text)
  features     interpretable features from URLs, parameters, headers and content
  signatures   conventional signature-based detector (WAF-style regular-expression rules)
  datasets     loaders for HTTP CSIC 2010 and HttpParamsDataset
  models       the AI/ML models (feature-based and character n-gram NLP models)
  experiments  R1-R4 experiments, tables and figures   (python -m webreq.experiments)
  analyzer     the assessment prototype: score, classify, explain and prioritise requests
"""
