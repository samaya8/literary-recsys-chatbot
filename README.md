# Enterprise Hybrid Book Recommendation & Conversational Intelligence Platform

[![CI](https://github.com/samaya8/literary-recsys-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/samaya8/literary-recsys-chatbot/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-42%20Passed-success?style=flat&logo=pytest&logoColor=white)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **GitHub Repository**: `literary-recsys-chatbot`  
> **Repository Description**: Enterprise hybrid recommendation platform uniting SVD matrix factorization, multi-field LSI semantic discovery with MMR diversification, and a multi-turn conversational bookstore agent with dialogue memory.

A dual-capability intelligence platform combining collaborative filtering matrix factorization, latent semantic content discovery, adaptive dynamic hybrid fusion, and a stateful conversational bookstore assistant equipped with sliding-window dialogue memory and entity/constraint extraction.

---

## Architecture & Workflows

### 1. Hybrid Recommendation Fusion Architecture
```text
  ┌──────────────────────────────────────────────┐     ┌──────────────────────────────────────────────┐
  │     Collaborative Filtering Pipeline         │     │         Semantic Content Pipeline            │
  │  • Explicit User-Item Ratings (Sparse)       │     │  • Multi-Field Weighted TF-IDF               │
  │  • Global Mean (μ) + User (bu) + Item (bi)   │     │    (Title 3x, Genre 2x, Author 2x, Desc 1x)  │
  │  • Latent Factors: P_u, Q_i (Truncated SVD)  │     │  • Latent Semantic Indexing (LSI SVD)        │
  │  • Rating Pred: r̂_ui = μ + bu + bi + P_u·Q_i │     │  • Maximal Marginal Relevance (MMR, λ=0.7)   │
  └──────────────────────┬───────────────────────┘     └──────────────────────┬───────────────────────┘
                         │                                                    │
                         ▼                                                    ▼
             [Collaborative Score S_CF]                           [Semantic Score S_Sem]
                         │                                                    │
                         └──────────────────────┬─────────────────────────────┘
                                                ▼
                                   [Adaptive Dynamic Fusion]
                                    α = f(User Rating Count)
                              • Cold-Start (<1): α = 0.00 (Pure Semantic)
                              • Emerging (2-4):  α = 0.40 (Balanced)
                              • Mature (≥5):     α = 0.65 (Collaborative Prioritized)
                                                │
                                                ▼
                             S_Final = α · S_CF + (1 - α) · S_Sem
                                                │
                                                ▼
                                  [Explainability Generator]
                           (Attribution rationale + genre alignment)
                                                │
                                                ▼
                                    [Top-K Ranked Books]
```

### 2. Conversational Bookstore Agent Architecture
```text
  [User Message: "I loved Dune, recommend another sci-fi book without romance"]
                                  │
                                  ▼
                    [Dialogue Memory Sliding Window]
                    (Recent turns, preferred & disliked genres)
                                  │
                                  ▼
                    [Intent Classification (TF-IDF)]
         ├── Recommendation (0.94)  ├── Book Information
         ├── Catalog Query          ├── Greeting / Farewell
                                  │
                                  ▼
               [Entity & Constraint Extraction (NER)]
         ├── Target Titles: ["Dune"]
         ├── Preferred Genres: ["science fiction", "sci-fi"]
         ├── Negative Constraints: ["romance"]
                                  │
                                  ▼
                    [Hybrid Recommender Engine]
         (Filtered by category & constraints, boosted by seed similarity)
                                  │
                                  ▼
                   [Structured Natural Language Reply]
         ("Here are top-rated science fiction titles matching your taste...")
```

---

## Problem Statement

Modern digital reading platforms and e-commerce bookstores encounter four critical discovery bottlenecks:
1. **Cold-Start Vulnerability**: Pure collaborative filtering fails when new readers lack interaction history.
2. **Catalog Redundancy & Homogeneity**: Naive content recommenders suggest repetitive books by the same author or sub-subgenre without novelty.
3. **Conversational Disconnect**: Traditional chatbots lack memory across dialogue turns and fail to parse negative constraints (*"no romance"*, *"something different"*).
4. **Black-Box Alienation**: Readers reject recommendations when algorithms fail to provide intuitive reasoning or justification.

This platform resolves these challenges through adaptive dynamic fusion, MMR catalog diversification, sliding-window dialogue memory tracking, and transparent explainability.

---

## Key Features

- **Matrix Factorization (SVD) with Bias Modeling**: Learns dense latent user and item representations while accounting for individual user rating propensities and item popularity biases.
- **Multi-Field Semantic Discovery (LSI)**: Weights titles ($3\times$), categories ($2\times$), authors ($2\times$), and synopses ($1\times$), projecting into compact latent semantic dimensions.
- **Maximal Marginal Relevance (MMR)**: Balances query relevance against intra-list redundancy ($\lambda = 0.70$) to ensure diverse suggestions.
- **Adaptive Dynamic Weighting**: Seamlessly pivots between pure semantic retrieval ($\alpha=0.0$) for cold-start users and mature collaborative filtering ($\alpha=0.65$) for established profiles.
- **Stateful Conversational Assistant**: Tracks multi-turn conversation context, resolves pronouns, updates active preference slots, and respects negative exclusion constraints.
- **Dual Streamlit Applications**:
  - `app.py`: Customer storefront (*"Bookworm Studio"*) with interactive chat, persona picker, and recommendation cards.
  - `app2.py`: ML diagnostic workbench (*"Recommender Intelligence Lab"*) featuring model ablation arena, 2D latent factor scatter projection, and live MMR tuning.
- **Enterprise CLI**: Production CLI supporting `recommend`, `chat` (interactive and single-turn), and `benchmark` commands.
- **Offline CPU Reproducibility**: High-throughput inference ($<3$ ms per query) with zero external API key requirements.

---

## Quantitative Benchmarks (Leave-One-Out Cross-Validation)

Evaluated on user-item interaction benchmarks using Leave-One-Out (LOO) positive ground truth ($r \ge 4.0$):

| Metric | Collaborative (SVD) | Semantic (LSI + MMR) | Hybrid Fusion | Advantage / Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Hit Rate@5** | 25.0% | 100.0% | **100.0%** | **+75.0%** over sparse CF alone |
| **Precision@5** | 5.0% | 20.0% | **20.0%** | **4x** higher precision |
| **Recall@5** | 25.0% | 100.0% | **100.0%** | Robust target recovery |
| **NDCG@5** | 0.188 | 0.891 | **0.891** | **+0.704** rank quality gain |
| **Catalog Coverage** | 80.0% | 100.0% | **90.0%** | Broad, diversified discovery |
| **Mean Latency** | 2.38 ms | 0.73 ms | **1.70 ms** | Ultra-low sub-millisecond CPU |

*Full benchmark documentation is available in [`evals/BENCHMARK_RESULTS.md`](evals/BENCHMARK_RESULTS.md).*

---

## Tech Stack

- **Recommendation & ML**: NumPy, Pandas, Scikit-Learn (TruncatedSVD, TfidfVectorizer, cosine_similarity)
- **Conversational Intelligence**: Regex tokenizers, Calibrated Intent Classifier, Dialogue State Tracking
- **Interactive UI**: Streamlit, Custom CSS
- **Testing & Verification**: Pytest, Pytest-Cov (42 unit tests, 93% code coverage)
- **CLI & Automation**: Python `argparse`, JSON Lines

---

## Repository Structure

```text
.
├── app.py                           # Customer storefront: Bookworm Studio (Streamlit)
├── app2.py                          # ML workbench: Recommender Intelligence Lab (Streamlit)
├── cli.py                           # Production CLI (recommend, chat, benchmark)
├── data/
│   ├── sample_books.csv             # Curated catalog dataset (Titles, Authors, Categories, Synopses)
│   ├── sample_ratings.csv           # User-item interaction ratings (sparse matrix)
│   └── intents.json                 # Dialogue intent patterns and responses
├── evals/
│   ├── __init__.py                  # Evaluation package marker
│   ├── runner.py                    # Automated Leave-One-Out benchmark harness
│   └── BENCHMARK_RESULTS.md         # Generated quantitative performance report
├── notebooks/
│   ├── 04- books-recommendation-system.ipynb
│   └── 04- chatbot-bookshop.ipynb
├── src/
│   ├── collaborative.py             # SVD Matrix Factorization with user/item bias modeling
│   ├── conversational_assistant.py  # Multi-turn assistant with dialogue memory & NER
│   ├── hybrid.py                    # Adaptive dynamic fusion & explainability engine
│   ├── semantic_recommender.py      # Multi-field TF-IDF + LSI + MMR diversification
│   ├── recommender.py               # Backward-compatible baseline recommender
│   └── chatbot_utils.py             # Backward-compatible NLP tokenization utilities
├── tests/
│   ├── test_cli.py                  # CLI subcommand & interactive session tests
│   ├── test_collaborative.py        # Collaborative SVD recommender tests
│   ├── test_conversational_assistant.py # Assistant memory & intent extraction tests
│   ├── test_evals.py                # Benchmark harness unit tests
│   ├── test_hybrid.py               # Hybrid weighting & explanation tests
│   ├── test_recommender.py          # Legacy recommender & utility tests
│   └── test_semantic_recommender.py # Semantic LSI & MMR diversification tests
├── requirements.txt                 # Pinned dependencies
├── LICENSE                          # MIT License
└── README.md                        # Documentation
```

---

## Setup & Quickstart

### 1. Prerequisites & Environment Setup

```bash
git clone https://github.com/your-username/literary-recsys-chatbot.git
cd literary-recsys-chatbot

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch Streamlit Applications

```bash
# Customer Storefront (Bookworm Studio)
streamlit run app.py

# ML Diagnostics Lab (Recommender Intelligence Lab)
streamlit run app2.py
```

### 3. Command-Line Interface (CLI)

```bash
# 1. Hybrid Recommendation by Seed Title with Explanations
python cli.py recommend --title "Dune" --top-k 3 --explain

# 2. Semantic Discovery by Natural Language Query
python cli.py recommend --strategy semantic --query "totalitarian surveillance dystopia" --genre "Classics"

# 3. Personalized Collaborative Recommendation for User ID
python cli.py recommend --strategy collaborative --user-id 1 --top-k 5

# 4. Single-Turn Chatbot Interaction
python cli.py chat --message "Can you recommend an epic fantasy adventure?"

# 5. Interactive Terminal Chatbot Session
python cli.py chat --interactive

# 6. Quantitative Evaluation Benchmark
python cli.py benchmark --top-k 5 --json
```

### 4. Running Verification Tests & Coverage

```bash
pytest --cov=src --cov=cli --cov=evals.runner --cov-report=term-missing tests/
```

---

## Python API Usage

```python
from src.hybrid import HybridBookRecommender
from src.conversational_assistant import ConversationalBookstoreAssistant

# 1. Hybrid Recommender
recommender = HybridBookRecommender()

# Recommend for an existing user based on a seed title
recommendations = recommender.recommend(user_id=1, seed_title="1984", top_k=3)
for book in recommendations:
    print(f"[{book['hybrid_score']:.3f}] {book['title']} ({book['category']})")
    print(f"  Rationale: {book['explanation']}")

# 2. Conversational Assistant with State Memory
assistant = ConversationalBookstoreAssistant()

turn1 = assistant.chat("I love classic sci-fi books like Dune")
print(turn1["response"])

turn2 = assistant.chat("Give me one more suggestion, but something lighter")
print(turn2["response"])
```

---

## License

Released under the [MIT License](LICENSE).
