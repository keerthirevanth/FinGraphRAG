# Knowledge-Graph RAG over SEC 10-K Filings

A retrieval-augmented question-answering system that answers questions about
the relationships between companies in the AI / semiconductor / cloud
ecosystem, built from their SEC 10-K filings. It compares two retrieval
strategies head to head: a conventional vector-search baseline and a
knowledge-graph approach, and evaluates both against ground truth derived from
the graph itself.

The central idea: the questions that matter about a company sector are rarely
answerable from a single document. "How is Tesla connected to TSMC?" or "Which
companies are exposed to a disruption at Samsung?" require connecting facts
that are scattered across dozens of separate filings. Conventional chunk-based
vector search retrieves passages that resemble the question but does not model
the relationships between them, so it cannot assemble a chain of facts that
lives in different documents. A knowledge graph stores those relationships
explicitly and can traverse them.

## Results

Both systems were evaluated on 76 questions in three categories, scored on
three metrics:

- **Entity recall** — the fraction of the ground-truth entities (the companies
  the reference answer names) that the system's answer actually mentions.
- **Correctness** — an LLM judge rates the answer against the reference, from
  0 to 1.
- **Faithfulness** — an LLM judge rates whether every claim in the answer is
  supported by the context the system retrieved.

| Metric | Question type | Vector RAG | GraphRAG |
|---|---|---|---|
| Entity recall | simple | 0.16 | **0.67** |
| | multi-hop | 0.03 | **0.87** |
| | global | 0.13 | **0.98** |
| Correctness | simple | 0.22 | **0.73** |
| | multi-hop | 0.03 | **0.77** |
| | global | 0.28 | **0.97** |
| Faithfulness | all | **1.00** | 0.97 |

Two findings stand out:

- **GraphRAG's advantage is largest exactly where it should be** — on
  multi-hop and global questions, which typically have no single supporting
  passage. On those, vector search is close to unusable (multi-hop
  correctness 0.03).
- **Vector RAG declined 47 of 76 questions** ("the excerpts do not mention a
  connection"), because the answer was not present in any one retrieved
  passage. GraphRAG declined only 2. Vector search stays marginally more
  faithful precisely because it only ever repeats text it retrieved verbatim.

## How it works

The pipeline runs in five stages, each a self-contained module.

1. **Ingestion** (`src/ingest`). Downloads the latest 10-K for 32 companies
   from SEC EDGAR and parses each filing's HTML into clean, section-labelled
   text. Real filings use several heading conventions and one inline-XBRL
   layout that hides the narrative behind a cross-reference index; the parser
   handles all of them, falling back to the whole document when item
   segmentation fails.

2. **Extraction** (`src/extract`). Splits the relationship-bearing sections
   (Business, Risk Factors, MD&A) into chunks and prompts an LLM to extract
   typed relationship triples against a fixed vocabulary of seven relations
   (supplier_of, customer_of, competitor_of, partner_of, subsidiary_of,
   depends_on, invests_in). Every triple carries an evidence quote from the
   source text. Extracted names are normalised so that one company is one
   node, and generic categories ("OEMs", "regulators") are rejected. Every
   response is cached on disk, so the pipeline is fully resumable and never
   pays twice for the same chunk.

3. **Graph construction** (`src/graph`). Merges the roughly 785 extracted
   triples into a directed knowledge graph, folding inverse relations
   (A customer_of B and B supplier_of A become one edge) and recording every
   filing that corroborates each edge. The result is 414 nodes and 656 edges,
   of which 21 are asserted by more than one company's filing.

4. **Retrieval and answering** (`src/rag`, `src/vector`). Two systems answer
   questions over the same corpus. The vector baseline splits the filings into
   about 6,900 passages of roughly 1,200 characters, embeds them locally with
   the `all-MiniLM-L12-v2` sentence-transformer, and retrieves the top five by
   cosine similarity before answering. GraphRAG links the companies named in a
   question to graph nodes, collects the paths and neighbourhood connecting
   them, and answers from those facts.

5. **Evaluation** (`src/eval`). Generates a question set whose answers are read
   directly from graph structure, then scores both systems. Deriving the
   reference answers from the graph makes the evaluation deterministic and
   reproducible rather than hand-invented. It is important to be precise about
   what this measures: both systems are compared on how well they retrieve and
   use the *same extracted knowledge base*, not on absolute factual correctness
   against the real world (see the limitations below).

## The API layer

The language model is accessed through an OpenAI-compatible client that
rotates across a pool of endpoints (`src/llm_client.py`). Free API tiers cap
usage per provider and per model per day, so the pool spreads one extraction
run across several provider/model combinations to multiply the usable daily
budget, disables an endpoint for the session when it reports a daily cap, and
stops gracefully when the whole pool is exhausted, with all completed work
cached. Switching providers is a change to one line of configuration.

Embeddings run locally on CPU, so retrieval costs nothing and the API budget
is reserved for extraction and generation.

## Repository layout

```
config/companies.json      the 32-company corpus definition
src/
  config.py                environment-driven configuration
  llm_client.py            OpenAI-compatible client with endpoint-pool rotation
  ingest/                  filing download and parsing
  extract/                 triple extraction, normalisation, caching
  graph/                   knowledge-graph construction
  vector/                  local vector index (baseline)
  rag/                     vector_rag and graph_rag answering systems
  eval/                    question generation and scoring harness
scripts/                   connection check, partial-triples pruner
app/streamlit_app.py       side-by-side UI, interactive graph, evaluation table
data/                      triples, graph, and evaluation outputs
```

## Setup

Requires Python 3.10+ and an API key from any OpenAI-compatible provider
(the project uses the free tiers of Groq and Google Gemini).

```
python -m venv .venv
.venv/Scripts/activate            # on Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
cp .env.example .env              # then fill in LLM_ENDPOINTS and SEC_EMAIL
```

Verify the model endpoints are reachable:

```
python -m scripts.check_llm_connection
```

## Running the pipeline

```
python -m src.ingest.download_filings     # download 10-Ks from EDGAR
python -m src.ingest.parse_filing         # parse into labelled sections
python -m src.extract.extractor           # extract relationship triples
python -m src.graph.build_graph           # assemble the knowledge graph
python -m src.vector.index                # build the vector baseline index
python -m src.eval.generate_questions     # generate the evaluation set
python -m src.eval.run_eval               # score both systems
```

Then launch the interface:

```
streamlit run app/streamlit_app.py
```

## Design decisions and limitations

- **NetworkX rather than a graph database.** At a few hundred nodes an
  in-memory graph is the right tool: no service to run and the whole project
  is reproducible from a clone. A server-based store such as Neo4j would only
  pay off at much larger scale.

- **Extraction yield reflects disclosure style, not just the extractor.**
  Semiconductor-equipment and analog firms tend to describe counterparties
  generically ("our ten largest customers") rather than naming them, so they
  contribute fewer edges. This is a property of the source data, and inventing
  edges to fill the gap would defeat the purpose.

- **The ground truth is graph-derived, so the evaluation is not fully
  independent.** The reference answers come from the same graph that GraphRAG
  retrieves over, and that graph was itself built by an LLM. The benchmark
  therefore measures which retrieval strategy best recovers what the knowledge
  base contains, not absolute real-world correctness. This favours the graph
  system by construction on the simpler questions, so those margins should be
  read with that in mind. The multi-hop and global results are more robust:
  the vector baseline fails there because it cannot combine facts across
  documents at all, regardless of where the ground truth comes from. A fully
  independent benchmark would use hand-labelled reference answers, which is the
  natural next step.

- **The evaluation surfaced a judge bias, which was corrected.** The LLM judge
  initially rewarded a system for correctly declining to answer, even when the
  question had a real answer, which inflated the weaker system's score. A
  deterministic rule now counts a refusal to an answerable question as
  incorrect, gated on entity recall so that a correct indirect answer is not
  penalised. The correction widened the true gap between the systems.

- **Some relationship directions are noisy.** A handful of extracted edges
  point the wrong way (a customer labelled a supplier). Because both systems
  are scored against the same ground truth, this affects them equally and does
  not change the comparison.

## Technology

Python, SEC EDGAR, sentence-transformers (local embeddings), NetworkX,
Streamlit and pyvis (interface and graph visualisation), and OpenAI-compatible
LLM APIs for extraction, generation, and evaluation.
