# Byte-sized — Table Extraction System

Use this deck outline to build a final presentation (10–15 minutes).

## 1. Title & Team (1 slide)
- Project: Byte-sized — AI OCR Table Extraction + Chat
- Team members, roles
- Demo URL (localhost) and repo link

## 2. Problem & Motivation (1–2 slides)
- Unstructured tables in PDFs/images hinder analytics
- Manual extraction is error-prone, costly, slow
- Goals: robust extraction, structure recovery, and interactive Q&A

## 3. Solution Overview (1 slide)
- End-to-end pipeline: detect → structure → OCR → export → chat
- Local-first, open-source, multilingual (ko/en)

## 4. Architecture (1–2 slides)
- FastAPI backend, enterprise UI, HF Table Transformer, EasyOCR
- Caching & preloading for low latency
- Diagram of modules and data flow

## 5. Models & Techniques (2 slides)
- Microsoft Table Transformer (detection + structure)
- OCR: EasyOCR (ko/en); notes on parameters
- Post-processing and cell assignment

## 6. Demo (2–3 slides)
- UI walkthrough: upload, pipeline stages, visualization
- Export all tables (CSV/HTML/Excel)
- Chatbot: ask table questions (sum/compare/filter)

## 7. Evaluation (1–2 slides)
- Metrics: Cell-F1, TEDS, IoU, CER/WER
- Qualitative examples; error cases and fixes

## 8. Engineering & Ops (1 slide)
- Preloading, caching, optional GPU
- Clean REST API; tests and logging

## 9. Roadmap (1 slide)
- FAISS retriever, auth & multi-user, fine-tuning dataset
- Borderless/complex table robustness

## 10. Takeaways (1 slide)
- Reliable extraction, fast UX, practical chat interface
- Ready for domain adaptation and scaling

## 11. Appendix (backup)
- Endpoint list, config flags, troubleshooting, references
