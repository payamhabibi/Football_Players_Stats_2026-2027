# ⚽ Advanced Football Analytics: Unsupervised Player Role Discovery & Scouting Engine

An end-to-end Machine Learning project applying **Unsupervised Learning** and **High-Dimensional Analytics** to European football match statistics (2026/2027 season). Rather than relying on traditional pitch positions (Defender, Midfielder, Attacker), this pipeline identifies **latent tactical profiles** from 81 granular performance metrics and implements an **AI-driven player recommendation engine** for scouting.

---

## 📌 Project Overview
In modern football, tactical responsibilities transcend nominal positions. A nominal midfielder may operate as a deep-lying playmaker or a box-to-box ball-winner. 

This repository delivers:
1. **Feature Engineering & Imputation:** Cleaned and imputed missing entries across 81 numerical attributes for 2,034 professional players.
2. **Feature Scaling:** Applied `StandardScaler` to align heterogeneous features (e.g., passing volume vs. goal tallies).
3. **Exploratory Data Analysis:** Extracted feature interdependencies using correlation heatmaps.
4. **Tactical Clustering (K-Means):** Unsupervised grouping of players into 5 distinct behavioral profiles.
5. **Dimensionality Reduction (PCA):** Compressed the 81-dimensional metric space to 2 components for spatial validation.
6. **Scouting Engine (Cosine Similarity):** Computed pairwise similarity across all 2,034 players to discover statistical twins for transfer scouting.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (`StandardScaler`, `KMeans`, `PCA`, `cosine_similarity`)
- **Data Visualization:** `matplotlib`, `seaborn`

---

## 📊 Pipeline & Experimental Results

### 1. Data Cleaning & Feature Space
- **Total Players:** 2,034
- **Numerical Features Selected:** 81 attributes
- **Missing Value Handling:** Median replacement across skewed performance indicators.

### 2. Exploratory Data Analysis (Correlation Heatmap)
A targeted correlation analysis evaluated the interaction between offensive creation (shots, key passes) and defensive volume (tackles, interceptions).
*(Place your generated heatmap image here: `images/correlation_heatmap.png`)*

### 3. K-Means Clustering & Tactical Distribution
The model identified 5 distinct operational clusters across the league:

| Cluster ID | Player Count | Tactical Interpretation |
| :---: | :---: | :--- |
| **Cluster 0** | 291 | Tactical Anchors & High-Volume Midfield Engine |
| **Cluster 1** | 681 | Positional Defenders & Structural Specialists |
| **Cluster 2** | 521 | Progressive Playmakers & Creative Ball-Carriers |
| **Cluster 3** | 516 | High-Intensity Pressing & Transition Profiles |
| **Cluster 4** | 25 | Unique Specialists (Outlier Distributions & Goalkeepers) |

### 4. Dimensionality Reduction (PCA)
Using Principal Component Analysis, 81 statistical metrics were projected onto two principal axes (`PCA_1` and `PCA_2`), successfully mapping players into coherent spatial neighborhoods based on their cluster tags.
*(Place your generated PCA scatter plot here: `images/pca_clusters.png`)*

### 5. Scouting Recommender System (Cosine Similarity)
The recommendation system computes pairwise cosine similarity across all scaled features (matrix shape: `2034 × 2034`). 

#### Real Output Test Case:
- **Target Athlete:** `Brenden Aaronson`
- **Assigned Tactical Cluster:** `Cluster 3`

Top 5 statistical matches retrieved by the system:

| # | Player | Assigned Cluster | Squad | Age | Position | Similarity Score |
| :-: | :--- | :-: | :--- | :-: | :-: | :-: |
| 1 | **Jeremie Boga** | 3 | Juventus | 29.0 | MF | **74.9%** |
| 2 | **Marco Brescianini** | 3 | Fiorentina | 26.0 | MF | **73.7%** |
| 3 | **Nathaniel Brown** | 3 | Bayern Munich | 23.0 | MF, DF | **73.6%** |
| 4 | **Tyler Adams** | 3 | Bournemouth | 27.0 | MF | **68.8%** |
| 5 | **Maghnes Akliouche** | 3 | Paris SG | 24.0 | FW | **67.9%** |

*All retrieved players belong to Cluster 3, validating the mathematical consistency between K-Means clustering and vector similarity.*

---

## 💻 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/payamhabibi/football-analytics-clustering.git](https://github.com/payamhabibi/football-analytics-clustering.git)
   cd football-analytics-clustering
