# PMC Indexer — Parallel Python + Kosha Implementation

## Context

Full RAG pipeline over PMC papers (XML + figures + tables). A standalone Python script is built first for testing, then ported to TypeScript/Electron inside Kosha.

Kosha already has: `corpus_documents`, `corpus_chunks`, Ollama embedding (nomic-embed-text), FTS search. The Python SQLite schema mirrors Kosha's data model so the eventual port is mechanical.

**Input:** `PMC{id}/` folders produced by `pubmed_review_downloader.py --format tgz`

---

## Phase 1: Python standalone (`pmc_indexer.py`)

### Subcommands
```bash
python pmc_indexer.py index  --input-dir ./reviews [options]
python pmc_indexer.py search --db pmc_index.db --query "drug resistance"
```

### CLI flags
| Flag | Default | Description |
|---|---|---|
| `--input-dir` | required | Folder of `PMC{id}/` subdirectories |
| `--db` | `./pmc_index.db` | SQLite output |
| `--ollama-url` | `http://localhost:11434` | Ollama base URL |
| `--vision-model` | `llava` | Model for figure interpretation |
| `--embed-model` | `nomic-embed-text` | Model for text embeddings |
| `--chunk-size` | `400` | Max words per chunk |
| `--chunk-overlap` | `50` | Word overlap between chunks |
| `--skip-vision` | flag | Skip figure interpretation |
| `--reindex` | flag | Reprocess already-indexed papers |
| `--top-k` | `10` | Results for search subcommand |

### SQLite Schema (mirrors Kosha)

```sql
-- mirrors corpus_documents
CREATE TABLE papers (
  id TEXT PRIMARY KEY,      -- PMC numeric ID
  title TEXT, authors TEXT, year TEXT, xml_path TEXT, indexed_at INTEGER
);

-- mirrors corpus_chunks
CREATE TABLE chunks (
  id TEXT PRIMARY KEY,
  paper_id TEXT,
  chunk_type TEXT,          -- section | figure_legend | figure_interp | table_caption | table_content
  section_title TEXT,
  element_id TEXT,          -- XML id of parent fig/table (for xref linking)
  text TEXT,
  embedding BLOB,           -- struct.pack('Nf', ...) float32
  FOREIGN KEY (paper_id) REFERENCES papers(id)
);

-- new in Kosha: pmc_figures
CREATE TABLE figures (
  id TEXT PRIMARY KEY,
  paper_id TEXT,
  fig_id TEXT,              -- XML id attr e.g. "fig1"
  label TEXT,               -- "Figure 1"
  caption TEXT,
  image_path TEXT,
  interpretation TEXT,      -- Ollama vision output
  FOREIGN KEY (paper_id) REFERENCES papers(id)
);

-- new in Kosha: pmc_tables
CREATE TABLE tables (
  id TEXT PRIMARY KEY,
  paper_id TEXT,
  table_id TEXT,
  label TEXT,
  caption TEXT,
  content_markdown TEXT,
  FOREIGN KEY (paper_id) REFERENCES papers(id)
);

-- new in Kosha: pmc_xrefs
CREATE TABLE xrefs (
  chunk_id TEXT,
  ref_type TEXT,            -- fig | table
  target_id TEXT,           -- fig_id or table_id
  PRIMARY KEY (chunk_id, ref_type, target_id)
);
```

### Pipeline Steps

#### 1. `parse_jats(xml_path, paper_dir)`
JATS/NLM XML structure used:
- **Metadata**: `<front><article-meta>` → title, authors, year
- **Sections**: recurse `<body><sec>` tree. Each `<p>` → `itertext()` for mixed content. Track every `<xref ref-type="fig|table" rid="...">` inside each paragraph for cross-reference linking.
- **Figures**: all `<fig>` elements → `{id, label, caption, image_path}`. Resolve `<graphic xlink:href>` against `paper_dir` (try .jpg .tif .tiff .png suffixes).
- **Tables**: all `<table-wrap>` → `{id, label, caption, content_markdown}`. Render `<thead>/<tbody>` rows as markdown `| col | col |`.

#### 2. `interpret_figure(image_path, model, ollama_url)`
- Base64-encode image
- POST `/api/generate` with `{"model": model, "images": [b64], "prompt": "..."}`
- Prompt: *"Describe this scientific figure: its type (graph/microscopy/diagram/etc.), key data or findings shown, axes and labels if present, and the main conclusion visible from the figure."*

#### 3. `chunk_words(text, size=400, overlap=50)`
Word-boundary splitting. Applied to section paragraphs, figure captions, table captions, figure interpretations, table markdown.

#### 4. `embed(text, model, ollama_url)`
POST `/api/embeddings` → list[float] → `struct.pack(f'{n}f', *v)` blob.

#### 5. `store(db, paper_data)`
Single transaction: papers → figures → tables → chunks → xrefs.

#### 6. `cosine_search(query, db, embed_model, top_k)`
- Embed query
- Load all chunk embeddings, compute cosine similarity (numpy)
- For top-k results, JOIN xrefs → figures/tables
- Print ranked results with linked figures/tables inline

### Search output format
```
[1] PMC7654321 | Results > Drug Treatment   score=0.91
    "Cells were treated with rifampicin (Fig. 3A) at concentrations..."
    -> Figure 3 (fig3a): Bar graph showing dose-response curve.
       Image: /reviews/PMC7654321/fig3a.jpg

[2] PMC7654321 | Results                    score=0.84
    "Table 2 summarises MIC values across strains..."
    -> Table 2: | Strain | MIC (ug/mL) | Resistance |
```

### Dependencies
```
pip install requests numpy
# All other imports (struct, sqlite3, base64, uuid, tarfile) are stdlib
```

---

## Phase 2: Kosha integration (TypeScript/Electron — future)

### New DB tables (add to `electron/main.cjs` DDL + `src/db/schema.ts`)
```sql
pmc_figures, pmc_tables, pmc_xrefs  -- same schema as above
-- extend corpus_chunks with: element_id TEXT, chunk_type TEXT
```

### New source files
| File | Purpose |
|---|---|
| `src/db/repos/pmcRepo.ts` | CRUD for pmc_figures, pmc_tables, pmc_xrefs |
| `src/pages/PmcImportPage.tsx` | UI: folder picker, progress bar, model selector |
| `src/stores/pmcStore.ts` | Zustand: index progress, search results |

### IPC handlers in `electron/main.cjs`
- `pmc:indexFolder(folderPath, options)` — full pipeline server-side (Node + better-sqlite3)
- `pmc:search(query, topK)` — embed query, cosine search, xref join
- Figure interpretation via same Ollama HTTP calls already used for embeddings

### Corpus integration
- Each paper → `corpus_documents` row (enables existing FTS search)
- Each chunk → `corpus_chunks` row (enables existing vector search)
- Figures/tables in dedicated tables, joined at search time via `pmc_xrefs`

---

## Verification

```bash
# Fast test (no vision)
python pmc_indexer.py index --input-dir ./reviews --skip-vision --reindex
python pmc_indexer.py search --query "drug resistance mechanism" --top-k 3

# Full pipeline with figure interpretation
python pmc_indexer.py index --input-dir ./reviews --vision-model llava
python pmc_indexer.py search --query "figure showing MIC distribution"
```
