# City of Water and Ink: Decoding Venice through Multi-Modal Semantic Search

This repository contains the high-performance retrieval engine for the project. The backend is engineered to bridge historical Venetian cartography and archival records through advanced vector search and multimodal AI inference.


## Functional Modules

The system provides three core retrieval capabilities designed for historical research and spatial analysis:

### 1. Text Search

The system employs a parallel dual-stream architecture to handle natural-language queries.

* **Mechanism**: Upon receiving a text query, the system generates two distinct embeddings: a semantic vector via MiniLM for archival documents and a visual-aligned vector via the Perception Encoder (PE) for map tiles.
* **Optimization**: Retrieval is executed concurrently using a `ThreadPoolExecutor` across independent Qdrant collections to minimize total latency.
* **Score Fusion**: Z-score normalization z = (x - μ) / σ to is applied to both result sets to reconcile heterogeneous similarity distributions before final ranking.

### 2. Image Search

This module facilitates both Image-to-Image (I2I) and Image-to-Text (I2T) discovery using user-provided visual inputs.

* **Mechanism**: The Perception Encoder extracts a 1024-dimensional feature vector from the uploaded image.
* **Retrieval**: This single vector is used to simultaneously identify visually similar map fragments and archival records that share comparable visual characteristics encoded in their `pe_vector` fields.
* **Consistency**: Image results undergo statistical normalization to ensure the merged ranked list is mathematically coherent.

### 3. 3D Heatmap Visualization

The heatmap service provides a lightweight data stream specifically designed for 3D visualization of information density and query relevance.

* **Search Mode**: When a query is provided, the system performs a vector search across both collections and returns latitude, longitude, and relevance-weighted scores to visualize hotspots of historical interest.


## System Performance

The backend is optimized for sub-second responsiveness to support an exploratory research workflow.

* **End-to-End Latency**: Achieved an average response time of **288 ms** over a 100-iteration functional benchmark.
* **Database Efficiency**: Optimized HNSW indexing in Qdrant enables raw retrieval times between 7 ms and 51 ms.
* **Concurrency**: Sustains a 0% error rate under high-frequency sequential testing.


## Repository Structure

The project follows a modular design pattern to separate concerns between API routing, business logic, and data persistence:

```text
backend/
└── app/
    ├── api/v1/           # API versioning and route definitions (search.py)
    ├── core/             # Global configuration (config.py)
    ├── repository/       # Data access layer for Qdrant (qdrant_repo.py)
    ├── schema/           # Pydantic models for request/validation (search.py)
    ├── service/          # Business logic: Score normalization (search_service.py)
    ├── utils/            # AI model loaders and feature extraction logic
    └── main.py           # Application entry point

```


## Technology Stack

* **Framework**: FastAPI (Python 3.10+)
* **Vector Database**: Qdrant (HNSW Indexing)
* **AI Models**: PyTorch-based MiniLM and Perception Encoder


## Installation

1. **Clone the Repository**
```bash
git clone https://github.com/SheEagle/Urban-Semantic-Search-Backend.git
cd Urban-Semantic-Search-Backend

```


2. **Install Dependencies**
```bash
pip install -r requirements.txt

```


3. **Run the Server**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000

```
