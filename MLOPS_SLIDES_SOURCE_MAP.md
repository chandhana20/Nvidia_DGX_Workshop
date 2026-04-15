# MLOps Slides Source Map

## How to Assemble the MLOps Section (Slides 25-36)

Copy slides from these 3 source decks and customize for GPU anomaly detection use case.

### Source Decks
| Label | Title | URL |
|-------|-------|-----|
| **Deck A** | MLOps end2end | [1W2Cxv...](https://docs.google.com/presentation/d/1W2Cxv_osYGGkm7xxYCxerk6EMhrbvKt00w7aP_iCcDI/edit) |
| **Deck B** | [Field-demos]-MLOps end2end - Churn | [1qY8d1...](https://docs.google.com/presentation/d/1qY8d1seaLsZbZgqTbVcc6knNGsBeDt5klt1z5Ok6ydI/edit) |
| **Deck C** | Databricks Overview (ML) | [1IZMXFoJ...](https://docs.google.com/presentation/d/1IZMXFoJ_Hw1Uwc6QBJVkO27xovev-JWb0neBztvLxiY/edit) |

---

### Slide-by-Slide Assembly

| Workshop Slide | Copy From | Customize |
|----------------|-----------|-----------|
| **25 - Why Companies Struggle** | Deck A slide 2 OR Deck B slide 4 | Add: "87% of ML projects never reach production" stat. Tie to NVIDIA: "You build the GPUs — now let's operationalize ML on them." |
| **26 - Full ML Lifecycle** | Deck A slide 5 OR Deck C slide 7 | Highlight the 3 layers (DataOps/ModelOps/DevOps). Add callout: "We've been building in the serving layer all morning — now we go upstream." |
| **27 - Databricks ML Platform** | Deck A slide 4 OR Deck C slide 4 | No major changes needed. This is a product overview slide. |
| **28 - Unity Catalog for AI** | Deck C slide 8 | Add example lineage: `gpu_telemetry` -> `gpu_health_features` -> `gpu_anomaly_detector` model. Highlight "one governance layer for data AND models." |
| **29 - Feature Store** | Deck C slide 11 | Customize: show PRIMARY KEY + TIMESERIES constraint from our `gpu_health_features` table. Mention online/offline feature serving. |
| **30 - MLflow Tracking & Registry** | Deck C slide 12 | Customize: show screenshot of our `gpu-anomaly-detection-workshop` experiment. Mention Champion/Challenger (replaces old Staging/Production). |
| **31 - MLOps CI/CD/CT/CM** | Deck A slide 10 | This is the conceptual framework slide. Map each loop to a Databricks product. Emphasize: "Most teams only do CI. The value is in CT + CM." |
| **32 - E2E Workflow (Money Slide)** | Deck B slide 39 (final v3) | This is the most complete workflow diagram. Walk through slowly. Map each step to our workshop notebook. |
| **33 - Model Serving** | Deck C slide 14 (adapted) | Customize for GPU use case: "Deploy anomaly detector, scale-to-zero, A/B test Champion vs Challenger." |
| **34 - Monitoring & Drift** | Deck A slide 9 + Deck B slide 39 | Combine data drift + prediction drift concepts. Show Jensen-Shannon threshold. Mention custom metrics. |
| **35 - Automated Retraining** | Deck B slide 29 | Show the webhook/automation flow. Tie to Databricks Workflows for trigger-based retraining. |
| **36 - Production Architecture** | **Create new** | Composite diagram: GPU Telemetry -> Features -> Training -> Registry -> Serving -> Monitoring -> Retrain. Overlay GenAI layer (Genie + KA + Supervisor). This is your capstone visual. |

---

### Timing Guide (15 min total)

| Phase | Slides | Time | Style |
|-------|--------|------|-------|
| **The Problem** | 25 | 2 min | Story — why MLOps matters |
| **The Framework** | 26-28 | 3 min | Architecture — 3 layers, UC governance |
| **The Building Blocks** | 29-31 | 4 min | Product deep-dive — Feature Store, MLflow, CI/CD loops |
| **The Full Picture** | 32 | 2 min | Money slide — pause and walk through the complete workflow |
| **What We'll Build** | 33-36 | 4 min | Preview — serving, monitoring, retraining, production arch |

---

### What to Change on Copied Slides

1. **Replace "Customer Churn" with "GPU Anomaly Detection"** everywhere
2. **Replace table names:** `churn_features` -> `gpu_health_features`, `churn_model` -> `gpu_anomaly_detector`
3. **Replace "Staging/Production"** with **"Champion/Challenger"** (v3 pattern)
4. **Add NVIDIA DGX context:** GPU telemetry, thermal thresholds, ECC errors
5. **Remove old webhook references** (slides 20-28 in Deck B) — use the v3 Deployment Job pattern instead
6. **Add the GenAI overlay** on the production architecture slide (slide 36)
