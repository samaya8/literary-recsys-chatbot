# Enterprise Recommendation Systems & Conversational Bookstore Intelligence

> **Domain**: Recommender Systems, Collaborative Filtering & Natural Language Understanding (NLU)  
> **Target System**: Hybrid Matrix Factorization, Latent Semantic Discovery & Stateful Conversational Bookstore Assistant

---

## 1. Executive Summary & Problem Statement

Modern e-commerce and content platforms rely on two foundational pillars for user retention, customer lifetime value (LTV), and conversion rate optimization:
1. **Recommender Systems**: Surfacing personalized, diverse, and relevant catalog items through collaborative interaction signals and semantic content affinities.
2. **Conversational Chatbots**: Assisting customers 24/7 with inquiries, inventory lookups, stateful dialogue navigation, and personalized reading recommendations.

This project delivers an enterprise-grade platform combining:
- **Collaborative Filtering**: SVD Matrix Factorization with explicit user and item bias modeling ($\hat{r}_{u,i} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i$).
- **Semantic Discovery**: Multi-field weighted TF-IDF (Title $3\times$, Genres $2\times$, Authors $2\times$, Description $1\times$) projected through TruncatedSVD Latent Semantic Indexing (LSI) with Maximal Marginal Relevance (MMR) diversification.
- **Adaptive Dynamic Fusion**: Smoothly scales collaborative weight $\alpha(u) \in [0.0, 0.65]$ based on user interaction maturity, guaranteeing cold-start immunity.
- **Stateful Conversational Assistant**: Sliding-window dialogue memory tracking active genres, negative constraints (*"no romance"*), and entity recognition for seamless multi-turn guidance.
- **Dual Streamlit Applications**: Customer storefront (`app.py`) and ML diagnostic lab (`app2.py`).
- **Production CLI**: High-throughput terminal interface (`cli.py`).

---

## 2. System Architecture & Workflows

### 2.1 Hybrid Recommendation Fusion Pipeline
```text
  [User ID: u, Seed Title: i, Query: q]
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  [Collaborative CF]      [Semantic Engine]
  • Bias Modeling         • Multi-Field TF-IDF
  • Latent SVD Factor     • Latent Semantic Indexing
  • Cosine Similarity     • MMR Diversification (λ=0.7)
         │                       │
         ▼                       ▼
  [Score S_CF ∈ [0,1]]    [Score S_Sem ∈ [0,1]]
         │                       │
         └───────────┬───────────┘
                     ▼
         [Adaptive Dynamic Weight]
         α = 0.00 if cold-start
         α = 0.40 if emerging
         α = 0.65 if mature
                     │
                     ▼
       S_Final = α·S_CF + (1 - α)·S_Sem
                     │
                     ▼
        [Explanation Generator]
  ("Recommended because you enjoyed Dune; shares high sci-fi affinity")
                     │
                     ▼
         [Top-K Ranked Recommendations]
```

### 2.2 Conversational Assistant Architecture
```text
  [Customer Utterance] ──► [Dialogue State Tracker / Memory]
                                   │
                                   ▼
                       [Intent Classifier (TF-IDF)]
               ├── Recommendation Request   ├── Book Information
               ├── Catalog Inventory Query  ├── Greetings / Smalltalk
                                   │
                                   ▼
                       [Entity Extraction Engine]
               ├── Known Titles (Fuzzy & Exact Match)
               ├── Preferred Categories & Subgenres
               └── Negative Constraints ("not horror", "no romance")
                                   │
                                   ▼
                       [Contextual Dispatcher]
               ├── Queries Hybrid Engine with accumulated constraints
               ├── Generates conversational natural response
               └── Returns structured recommendation payload
```

---

## 3. Mathematical Foundations

### 3.1 Collaborative Filtering with Bias Modeling
Predicted rating $\hat{r}_{u,i}$ for user $u$ and item $i$:
$$\hat{r}_{u,i} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i$$
Where:
- $\mu$: Global average rating across all interactions.
- $b_u = \frac{\sum_{i \in R_u} (r_{u,i} - \mu)}{|R_u| + \lambda_u}$: User rating propensity bias with L2 shrinkage.
- $b_i = \frac{\sum_{u \in R_i} (r_{u,i} - \mu - b_u)}{|R_i| + \lambda_i}$: Item popularity bias with L2 shrinkage.
- $\mathbf{p}_u \in \mathbb{R}^d, \mathbf{q}_i \in \mathbb{R}^d$: Dense latent factor vectors learned via Singular Value Decomposition (SVD).

### 3.2 Maximal Marginal Relevance (MMR)
Given query representation $\mathbf{q}$, candidate document set $C$, and selected set $S$:
$$\text{MMR} = \arg\max_{d_i \in C \setminus S} \left[ \lambda \cdot \text{Sim}(d_i, \mathbf{q}) - (1 - \lambda) \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]$$
Setting $\lambda = 0.70$ balances relevance with intra-list novelty, preventing near-duplicate author recommendations.

---

## 4. Component Directory & File Inventory

```text
literary-recsys-chatbot/
├── app.py                           # Customer storefront: Bookworm Studio (Streamlit)
├── app2.py                          # ML workbench: Recommender Intelligence Lab (Streamlit)
├── cli.py                           # Enterprise CLI (recommend, chat, benchmark)
├── data/
│   ├── sample_books.csv             # 10 curated books with rich metadata
│   ├── sample_ratings.csv           # 25 user interaction ratings
│   └── intents.json                 # Dialogue schemas, patterns & replies
├── evals/
│   ├── __init__.py                  # Package marker
│   ├── runner.py                    # Automated Leave-One-Out evaluation harness
│   └── BENCHMARK_RESULTS.md         # Generated benchmark report
├── notebooks/
│   ├── 04- books-recommendation-system.ipynb
│   └── 04- chatbot-bookshop.ipynb
├── src/
│   ├── collaborative.py             # SVD Matrix Factorization with bias modeling
│   ├── conversational_assistant.py  # Multi-turn assistant with dialogue memory & NER
│   ├── hybrid.py                    # Adaptive dynamic fusion & explainability engine
│   ├── semantic_recommender.py      # Multi-field TF-IDF + LSI + MMR diversification
│   ├── recommender.py               # Legacy baseline recommender
│   └── chatbot_utils.py             # Legacy NLP utilities
├── tests/
│   ├── test_cli.py                  # CLI tests (15 test cases)
│   ├── test_collaborative.py        # Collaborative SVD tests (5 test cases)
│   ├── test_conversational_assistant.py # Assistant tests (5 test cases)
│   ├── test_evals.py                # Benchmark tests (4 test cases)
│   ├── test_hybrid.py               # Hybrid tests (4 test cases)
│   ├── test_recommender.py          # Legacy tests (3 test cases)
│   └── test_semantic_recommender.py # Semantic LSI/MMR tests (6 test cases)
├── requirements.txt                 # Pinned dependencies
├── LICENSE                          # MIT License
└── README.md                        # Primary documentation
```

---

## 5. Verification & Testing

### Automated Test Suite
The project includes 42 automated unit tests across 7 test suites covering 93% of codebase statements:
```bash
pytest --cov=src --cov=cli --cov=evals.runner --cov-report=term-missing tests/
```

### Benchmark Reproduction
Run the Leave-One-Out quantitative evaluation benchmark:
```bash
python evals/runner.py
# Or via CLI
python cli.py benchmark --top-k 5
```

---

## 6. License
Released under the [MIT License](LICENSE).
