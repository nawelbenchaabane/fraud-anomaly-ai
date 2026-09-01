# Fraud Anomaly AI

An end-to-end fraud investigation project combining **unsupervised anomaly detection**, **temporal validation**, **explainability**, **risk routing**, and **LLM-assisted analyst reporting**.

The core design principle is simple:

> **Machine-learning models detect and score anomalies.  
> The LLM does not detect fraud; it summarizes supplied evidence for a human analyst.**

---

## Project overview

This project uses the synthetic **BankSim** transaction dataset to explore several anomaly-detection approaches for fraud screening:

- Isolation Forest
- Local Outlier Factor
- One-Class SVM
- Autoencoder

The project then adds:

- temporal train / validation / test evaluation
- operational threshold selection
- score-drift analysis
- model comparison
- transaction-level explainability
- LOW / MEDIUM / HIGH investigation routing
- structured investigation packets
- deterministic analyst reports
- optional LLM-generated analyst reports
- CLI inference
- automated reporting tests

The final system is designed as an **investigation-support pipeline**, not an autonomous fraud-decision engine.

---

## Why this project?

Fraud detection is not only a ranking problem.

A useful system must answer several operational questions:

- Which transactions should be reviewed?
- How many alerts will analysts receive?
- How much fraud coverage is achieved?
- How many legitimate transactions become false alerts?
- Why was a transaction considered unusual?
- Can a human-readable investigation summary be produced without allowing an LLM to invent evidence?

This project focuses on those questions rather than optimizing a single benchmark metric.

---

## System architecture

```text
Raw BankSim transaction
        ↓
Saved preprocessing pipeline
        ↓
┌─────────────────────────────┐
│ Isolation Forest            │
│ One-Class SVM               │
│ Autoencoder                 │
└─────────────────────────────┘
        ↓
Continuous anomaly scores
        ↓
Validation-referenced percentiles
+ operational thresholds
        ↓
LOW / MEDIUM / HIGH
investigation routing
        ↓
Historical/contextual explanation
+ Autoencoder reconstruction evidence
        ↓
Structured investigation packet
        ↓
┌─────────────────────────────┐
│ Deterministic local report  │
│ Optional LLM report         │
└─────────────────────────────┘
        ↓
Human analyst
```

---

## Responsibility boundaries

The system deliberately separates detection, explanation, and language generation.

### Anomaly detectors

Responsible for:

- learning transaction patterns
- producing anomaly scores
- triggering saved operational thresholds

### Explainability layer

Responsible for:

- validation-referenced anomaly percentiles
- model agreement
- amount rarity
- merchant/category rarity
- Autoencoder reconstruction evidence

### LLM layer

Responsible only for:

- summarizing supplied evidence
- identifying model agreement/disagreement
- recommending analyst verification steps
- communicating limitations
- producing a consistent analyst-facing report

The LLM does **not**:

- calculate anomaly scores
- modify model thresholds
- modify the upstream risk tier
- receive the hidden fraud label
- interpret anomaly scores as fraud probabilities
- declare confirmed fraud
- replace human review

---

# Dataset

## BankSim

The project uses the main BankSim transaction file:

```text
bs140513_032310.csv
```

Dataset size:

```text
594,643 transactions
```

Original columns:

| Column | Meaning |
|---|---|
| `step` | Simulated time |
| `customer` | Anonymized customer |
| `age` | Age group |
| `gender` | Gender code |
| `zipcodeOri` | Origin ZIP-like field |
| `merchant` | Anonymized merchant |
| `zipMerchant` | Merchant ZIP-like field |
| `category` | Transaction category |
| `amount` | Transaction amount |
| `fraud` | Ground-truth fraud label |

Fraud prevalence:

```text
7,200 frauds / 594,643 transactions
≈ 1.21%
```

---

## Important BankSim limitations

BankSim is synthetic and contains strong structural shortcuts.

Examples observed during EDA:

- `es_transportation` represents roughly 85% of transactions and contains no fraud in this dataset
- several categories have very high fraud rates
- several merchants are strongly associated with fraud
- fraudulent transactions have much larger amounts on average

Example amount statistics:

```text
Legitimate mean amount ≈ 31.85
Fraud mean amount      ≈ 530.93
```

Therefore:

> **The reported metrics should not be interpreted as representative of production banking fraud detection.**

This repository is primarily an engineering and anomaly-detection case study.

---

# Temporal evaluation

A temporal split is used instead of a random split.

```text
Train       step 0–119
Validation  step 120–149
Test        step 150–179
```

Resulting sizes:

```text
Train       374,914 transactions
Validation  108,423 transactions
Test        111,306 transactions
```

This better approximates the real operational question:

> Can a detector trained on historical transactions generalize to future transactions?

---

# Preprocessing

Final detector inputs begin with:

### Numerical

- `step`
- `log1p(amount)`

### Categorical

- `age`
- `gender`
- `merchant`
- `category`

The preprocessing pipeline uses:

- `StandardScaler` for numerical variables
- `OneHotEncoder(handle_unknown="ignore")` for categorical variables

The preprocessor is fitted **only on the training set**.

Final transformed dimensionality:

```text
79 features
```

The high-cardinality `customer` identifier is deliberately excluded as a direct one-hot feature.

It is better suited to future behavioral features such as:

- customer average amount
- transaction count
- customer amount deviation
- first-time merchant usage
- first-time category usage

---

# Unsupervised training

The anomaly detectors are trained without the fraud target.

Conceptually:

```python
model.fit(X_train)
```

not:

```python
model.fit(X_train, y_train)
```

The `fraud` label is used only for:

- evaluation
- validation threshold selection
- benchmark comparison

This distinction is central to the methodology.

---

# Models

## 1. Isolation Forest

Configuration:

```python
IsolationForest(
    n_estimators=200,
    max_samples=4096,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)
```

Scoring convention:

```text
anomaly_score = -decision_function(X)
```

Therefore:

```text
higher score = more anomalous
```

---

## 2. Local Outlier Factor

LOF was evaluated as a local-density baseline with:

```python
LocalOutlierFactor(
    n_neighbors=35,
    contamination="auto",
    novelty=True,
    n_jobs=-1
)
```

It performed poorly in the full encoded feature space.

Selected experiments:

| Feature space | Validation ROC-AUC | Validation PR-AUC |
|---|---:|---:|
| All encoded features | 0.1886 | 0.0113 |
| Numerical only | 0.1332 | 0.0060 |
| `amount_log` only | 0.6292 | 0.0467 |

This is kept as an informative **negative benchmark**.

The result illustrates how strongly local-density methods depend on feature-space geometry.

---

## 3. One-Class SVM

Final model:

```python
OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05
)
```

Training uses a reproducible random sample of:

```text
30,000 training transactions
```

### Important ablation: removing `step`

The original One-Class SVM used absolute `step` as a feature.

This caused severe score drift:

```text
Train period      step 0–119
Validation period step 120–149
Test period       step 150–179
```

With an RBF kernel, later observations became increasingly distant from the training support even when they were legitimate.

Removing `step` reduced the feature space from:

```text
79 → 78 features
```

and stabilized validation/test score distributions.

This is one of the main methodological findings of the project:

> A feature can be useful for **temporal splitting** while still being harmful as a direct model input.

---

## 4. Autoencoder

CPU-first PyTorch Autoencoder:

```text
78 → 32 → 12 → 32 → 78
```

The absolute `step` feature is excluded.

Training configuration:

- MSE reconstruction loss
- Adam optimizer
- batch size 1024
- early stopping
- CPU execution

Anomaly score:

```text
mean squared reconstruction error
```

Therefore:

```text
higher reconstruction error = more anomalous
```

---

# Final ranking metrics

Final reproducible runs:

| Model | Validation ROC-AUC | Validation PR-AUC | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|---:|---:|
| Isolation Forest | 0.9654 | 0.2112 | **0.9672** | 0.2102 |
| One-Class SVM | 0.9249 | **0.4473** | 0.9255 | **0.4370** |
| Autoencoder | 0.9658 | 0.1801 | 0.9661 | 0.1752 |

### Interpretation

- Isolation Forest and Autoencoder provide the strongest global ROC ranking.
- One-Class SVM dominates PR-AUC.
- PR-AUC is particularly important because fraud represents only about 1.2% of the dataset.

No single model wins every objective.

---

# Operational thresholds

Two thresholding strategies are evaluated.

## Max F1

Select the validation threshold maximizing:

```text
F1 = harmonic mean of Precision and Recall
```

## High Recall

Among validation thresholds achieving at least 80% Recall, choose the threshold with maximum Precision.

Threshold selection is done **only on validation**.

The test set is never used to tune thresholds.

---

# Final operational results

## Max-F1 strategy

| Model | Precision | Recall | F1 | False Positive Rate | Alert Rate |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | 21.15% | 46.00% | 28.98% | 1.87% | 2.34% |
| **One-Class SVM** | **57.25%** | 38.17% | **45.80%** | **0.31%** | **0.72%** |
| Autoencoder | 29.94% | 28.92% | 29.42% | 0.74% | 1.04% |

### Best high-confidence model

**One-Class SVM**

Test result:

```text
458 frauds detected
742 frauds missed
342 false alerts
109,764 true negatives
```

Only about:

```text
0.72%
```

of test transactions are sent to the alert queue.

---

## High-recall strategy

| Model | Precision | Recall | F1 | False Positive Rate | Alert Rate |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | **13.40%** | 81.58% | 23.02% | **5.75%** | **6.56%** |
| One-Class SVM | 10.69% | 79.50% | 18.84% | 7.24% | 8.02% |
| Autoencoder | 13.18% | **93.92%** | **23.12%** | 6.74% | 7.68% |

### Maximum fraud coverage

**Autoencoder**

```text
1,127 frauds detected
73 frauds missed
7,421 false alerts
102,685 true negatives
```

### Balanced high-recall screening

**Isolation Forest**

It detects fewer frauds than the Autoencoder but maintains a smaller alert queue and lower false-positive rate.

---

# Model selection by operational objective

| Operational goal | Recommended model | Why |
|---|---|---|
| High-confidence investigations | **One-Class SVM** | Highest precision and F1, lowest alert rate |
| Balanced high-recall screening | **Isolation Forest** | Strong recall with controlled alert volume |
| Maximum fraud coverage | **Autoencoder** | Highest Recall |

This is a key project conclusion:

> **Model selection should depend on investigation capacity and business cost, not only on ROC-AUC.**

---

# Risk routing

The production-style pipeline converts model output into investigation tiers.

Scores from the three models use different numerical scales.

They are therefore converted into empirical percentiles relative to each model's **validation score distribution**.

Example:

```text
0.50 → around the middle of validation anomaly scores
0.95 → more anomalous than about 95% of validation transactions
0.99 → extremely unusual according to that detector
```

---

## LOW

Typical characteristics:

- low anomaly percentiles
- limited model agreement
- no high-confidence One-Class SVM trigger

Operational action:

```text
routine monitoring
```

---

## MEDIUM

Meaningful anomaly evidence exists but does not reach the priority-investigation rule.

Operational action:

```text
routine analyst review
```

---

## HIGH

A transaction is routed HIGH when:

```text
3 / 3 models are in their top 5% anomaly region
```

or:

```text
One-Class SVM triggers its high-confidence Max-F1 threshold
```

Operational action:

```text
priority analyst review
```

A HIGH risk tier does **not** mean confirmed fraud.

---

# Explainability

The explainability layer uses training-period statistics only.

It can provide:

- global amount percentile
- amount percentile within transaction category
- amount percentile for the same merchant
- category rarity
- merchant rarity
- model agreement
- operational threshold status
- Autoencoder reconstruction evidence

---

## Autoencoder explanation caveat

For every transformed feature:

```text
feature reconstruction error
=
(input - reconstruction)²
```

The largest errors show which transformed dimensions were hardest to reconstruct.

They are **not causal fraud explanations**.

A transaction can always have a "largest reconstruction feature", even when its overall Autoencoder anomaly score is low.

Therefore, detailed reconstruction evidence is down-weighted when the Autoencoder itself does not consider the transaction materially anomalous.

---

# Structured investigation packet

The scoring and explainability layers produce an LLM-safe JSON packet.

Conceptually:

```json
{
  "transaction_id": 483406,
  "risk_tier": "HIGH",
  "models_top_5pct": 3,
  "transaction": {
    "step": 150,
    "customer": "...",
    "merchant": "...",
    "category": "...",
    "amount": 100.0
  },
  "model_scores": {
    "isolation_forest": {},
    "one_class_svm": {},
    "autoencoder": {}
  },
  "context": {},
  "autoencoder_top_reconstruction_features": []
}
```

The hidden BankSim `fraud` label is deliberately excluded.

---

# Analyst reporting

Two report renderers are available.

## Deterministic local renderer

Works without an external API.

Useful for:

- testing
- reproducibility
- offline operation
- semantic baseline

## Optional LLM renderer

The LLM receives only the structured investigation packet.

It is instructed to:

- preserve risk tier
- preserve transaction ID
- distinguish detector evidence from contextual evidence
- acknowledge model disagreement
- avoid probability claims
- avoid unsupported facts
- recommend analyst actions
- communicate limitations

---

# LOW / MEDIUM / HIGH report validation

Representative real pipeline cases were tested manually and automatically.

Validated examples:

```text
LOW     transaction 483337
MEDIUM  transaction 483339
HIGH    transaction 483406
```

Final report validation:

| Renderer | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|
| Deterministic local | ✅ | ✅ | ✅ |
| LLM | ✅ | ✅ | ✅ |

Semantic checks include:

- transaction ID preserved
- risk tier preserved
- LOW does not escalate
- MEDIUM uses routine analyst review
- HIGH uses priority analyst review
- no affirmative confirmed-fraud claim
- anomaly scores are not interpreted as fraud probabilities

---

# Example LOW report behavior

Example detector output:

```text
Isolation Forest percentile : 0.108
One-Class SVM percentile    : 0.289
Autoencoder percentile      : 0.124

Models in top 5%            : 0 / 3
```

Expected conclusion:

```text
Current detector evidence does not justify priority investigation.
The transaction can remain under routine monitoring.
```

---

# Example MEDIUM report behavior

Example:

```text
Isolation Forest percentile : 0.979
One-Class SVM percentile    : 0.971
Autoencoder percentile      : 0.920

Models in top 5%            : 2 / 3
```

Expected routing:

```text
routine analyst review
```

---

# Example HIGH report behavior

Example:

```text
Isolation Forest percentile : 0.995
One-Class SVM percentile    : 0.976
Autoencoder percentile      : 0.979

Models in top 5%            : 3 / 3
```

Expected routing:

```text
priority analyst review
```

without claiming that fraud has been confirmed.

---

# Repository structure

```text
fraud-anomaly-ai/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── preprocessor.joblib
│   ├── feature_names.json
│   ├── isolation_forest.joblib
│   ├── isolation_forest_thresholds.json
│   ├── one_class_svm_no_step.joblib
│   ├── one_class_svm_thresholds.json
│   ├── one_class_svm_train_indices.npy
│   ├── autoencoder_state_dict.pt
│   ├── autoencoder_config.json
│   └── autoencoder_thresholds.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_isolation_forest.ipynb
│   ├── 04_local_outlier_factor.ipynb
│   ├── 05_one_class_svm.ipynb
│   ├── 06_autoencoder.ipynb
│   ├── 07_model_comparison.ipynb
│   ├── 08_anomaly_explainability.ipynb
│   ├── 09_llm_investigation_reports.ipynb
│   └── 10_llm_report_validation.ipynb
│
├── results/
│   ├── figures/
│   ├── metrics/
│   ├── scores/
│   ├── investigations/
│   ├── cli/
│   └── llm_validation/
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── scoring.py
│   ├── explainability.py
│   ├── investigation.py
│   ├── pipeline.py
│   └── reporting.py
│
├── tests/
│   ├── test_helpers.py
│   └── test_reporting.py
│
├── run_investigation.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Notebook workflow

```text
01 EDA
   ↓
02 Preprocessing
   ↓
03 Isolation Forest
04 Local Outlier Factor
05 One-Class SVM
06 Autoencoder
   ↓
07 Model comparison
   ↓
08 Anomaly explainability
   ↓
09 LLM investigation reports
   ↓
10 LLM report validation
   ↓
Reusable src/ pipeline + CLI
```

---

# Installation

Python 3.12 is recommended.

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For tests:

```powershell
python -m pip install pytest
```

For optional OpenAI report generation:

```powershell
python -m pip install -U openai
```

---

# CLI demo

The reusable pipeline can analyze a transaction directly from the command line.

Example using a transaction from the temporal test set:

```powershell
python run_investigation.py --test-index 0
```

The command:

1. loads the preprocessor
2. loads the three saved models
3. scores the transaction
4. computes validation-referenced percentiles
5. assigns a risk tier
6. generates contextual explanations
7. computes Autoencoder feature-level reconstruction errors
8. creates an investigation packet
9. creates an analyst report
10. saves the result

Outputs:

```text
results/cli/
├── transaction_<id>_packet.json
├── transaction_<id>_report.json
└── transaction_<id>_report.md
```

---

# Custom transaction

Create:

```text
transaction.json
```

Example:

```json
{
  "step": 170,
  "customer": "C123",
  "age": "3",
  "gender": "F",
  "merchant": "M1823072687",
  "category": "es_health",
  "amount": 750.0
}
```

Then:

```powershell
python run_investigation.py --json transaction.json
```

---

# Optional LLM report

Configure an API key through an environment variable.

Do **not** commit secrets to Git.

Example PowerShell session:

```powershell
$env:OPENAI_API_KEY="..."
```

Then:

```powershell
python run_investigation.py --test-index 0 --llm
```

The model can be configured through:

```text
--model
```

or:

```text
OPENAI_MODEL
```

The local deterministic renderer remains available without an API key.

---

# Tests

Run:

```powershell
python -m pytest -q
```

The current test suite covers:

- validation-referenced percentile calculation
- ground-truth leakage prevention
- LOW report semantics
- MEDIUM report semantics
- HIGH report semantics
- high-confidence One-Class SVM routing
- preservation of transaction ID
- preservation of risk tier
- avoidance of probability claims

---

# Saved artifacts

Each final model notebook persists reusable artifacts.

Examples:

```text
models/
results/scores/
results/metrics/
```

This avoids requiring retraining just to reproduce comparisons or inference.

---

# Key engineering lessons

## 1. ROC-AUC alone is not enough

The One-Class SVM has lower ROC-AUC than Isolation Forest but dramatically better PR-AUC and precision in selective mode.

---

## 2. Threshold choice is a business decision

The same anomaly score ranking can produce very different systems depending on the threshold.

```text
high precision
vs
high recall
```

is an operational trade-off.

---

## 3. Temporal leakage matters

The preprocessor is fitted only on training data.

Thresholds are selected only on validation data.

The test period remains untouched until final evaluation.

---

## 4. Features can create artificial drift

Absolute `step` produced severe score drift for the RBF One-Class SVM.

Removing it stabilized future scoring.

---

## 5. Local-density methods can fail in high-dimensional encoded spaces

LOF performed poorly on the full one-hot feature space.

That negative result is useful and documented.

---

## 6. Explainability must match model confidence

A feature can have the largest Autoencoder reconstruction error even when the overall transaction is not anomalous.

Feature-level explanations must therefore be interpreted together with the global anomaly score.

---

## 7. LLMs should not own the detection decision

The LLM is intentionally downstream of the ML and risk-routing layers.

This reduces hallucination risk and makes the system easier to audit.

---

# Limitations

This project is not a production-ready banking fraud platform.

Important limitations include:

- synthetic BankSim data
- strong category/merchant shortcuts
- no real identity context
- limited customer behavioral history
- no graph-based transaction relationships
- no concept-drift monitoring in deployment
- no production latency benchmark yet
- no cost-sensitive financial loss function
- no analyst feedback loop
- no real-world regulatory integration

---

# Future work

Potential extensions:

- historical customer behavioral features
- rolling transaction aggregates
- new merchant/category indicators
- graph-based fraud features
- score-distribution drift monitoring
- cost-sensitive threshold optimization
- model ensemble experiments
- lightweight API
- web dashboard
- analyst feedback loop
- richer LLM evaluation benchmark
- deployment packaging

---

# Tech stack

```text
Python 3.12
pandas
NumPy
SciPy
scikit-learn
PyTorch
joblib
Jupyter
pytest
OpenAI API (optional)
```

The entire anomaly-detection pipeline is designed to run **CPU-first**.

---

# Project status

```text
EDA                         ✅
Temporal preprocessing      ✅
Isolation Forest            ✅
LOF benchmark               ✅
One-Class SVM               ✅
Autoencoder                 ✅
Model comparison            ✅
Explainability              ✅
Risk routing                ✅
CLI inference               ✅
Deterministic reporting     ✅
Automated tests             ✅
LLM reporting               ✅
LLM semantic validation     ✅
Portfolio documentation     ✅
```

---

## Final takeaway

This project demonstrates more than anomaly-model training.

It builds a complete investigation workflow:

> **detect → score → calibrate → route → explain → report → review**

The most important result is not that one algorithm “wins”.

It is that different anomaly detectors support different operational objectives, and that a useful fraud-investigation system must explicitly manage those trade-offs while keeping detection logic separate from language-generation logic.
