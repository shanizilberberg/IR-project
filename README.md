<div align="center">

  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-2.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white" alt="GCP" />
  
  <br />
  <br />

  <h1 align="center">🔎 Wikipedia Search Engine</h1>

  <p align="center">
    A scalable, high-performance Information Retrieval system built on Google Cloud Platform.
    <br />
    Retrieves results from the entire English Wikipedia corpus using inverted indexes and composite ranking.
    <br />
    <br />
    <a href="http://YOUR_ENGINE_IP_HERE:8080/search?query=computer">
      <img src="https://img.shields.io/badge/🚀_Launch_The_Engine-FF0000?style=for-the-badge" alt="Launch Engine" />
    </a>
    <br />
    <br />
    <a href="#api-reference"><strong>Explore the Docs »</strong></a>
    <br />
    <br />
    <a href="#ranking-algorithm">View Logic</a>
    ·
    <a href="#architecture">Architecture</a>
    ·
    <a href="#issues">Report Bug</a>
  </p>
</div>

---

<details>
  <summary><strong>📖 Table of Contents</strong></summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#ranking-algorithm">Ranking Algorithm</a></li>
    <li><a href="#api-reference">API Reference</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
  </ol>
</details>

---

## ⚡ About The Project

This project implements a search engine backend capable of retrieving and ranking Wikipedia articles. It is designed to handle large-scale data using efficient index structures stored on **Google Cloud Storage (GCS)**.

The system features a **Flask** REST API that exposes various search methods, including retrieval by body text, title, and anchor text, along with auxiliary data like PageRank and PageViews.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🛠️ Architecture

The engine initializes by loading several optimized data structures from GCS bucket `badashbucket1`:

* **Inverted Indexes:** Separate indexes for Body, Title, and Anchor text.
    * *Body Index:* Reads posting lists from partitioned `.bin` files using offsets.
* **PageRank:** A dictionary mapping document IDs to their global importance score.
* **PageViews:** A dataset of monthly traffic (August 2021) used to boost popular pages.
* **ID Mappings:** Lookup tables for converting Doc IDs to Titles.

---

## 🧮 Ranking Algorithm

The main search functionality (`/search`) uses a composite scoring formula to rank relevance. The score for a document $d$ given a query $q$ is calculated as:

$$Score(d,q) = \sum_{term \in q} \left( W_{title} \times \mathbb{1}_{title} \right) + \left( W_{PR} \times \log(PR_d + 1) \right) + \left( W_{PV} \times \log(PV_d + 1) \right)$$

**Ranking Weights:**
* **Title Match ($W_{title}$):** `5.0` (Heaviest weight for direct title relevance).
* **PageRank ($W_{PR}$):** `2.0` (Log-smoothed importance).
* **PageViews ($W_{PV}$):** `0.001` (Log-smoothed traffic signal).

*Additional Logic:*
* **Title Boost:** If the exact query appears in the title, the score is multiplied by 3.
* **Body Search:** Uses Cosine Similarity on TF-IDF vectors.

---

## 📡 link to the engine 

### 1. Main Search
Returns the top results using the composite ranking algorithm.
http://34.172.174.63:8080/search?query=hello
