# Movie Taste Map

Hybrid movie recommendation system combining LSH, K-Means clustering, and collaborative filtering. Built for DTU course 02807 (Computational Tools for Data Science).

## Approach

- **LSH** — finds similar movies using feature overlap (cast, genre, director)
- **Clustering** — groups movies semantically using SBERT embeddings
- **Collaborative filtering** — predicts ratings based on user behavior

The system adapts its weighting based on how much rating data is available for each user.

## Dataset

[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle 

