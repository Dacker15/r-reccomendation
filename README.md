# Recommendation Systems in R

In this project, we compared the results of different recommendation systems using the [recommenderlab](https://cran.r-project.org/web/packages/recommenderlab/index.html) library based on the [MovieLens Small](https://grouplens.org/datasets/movielens/latest/) dataset.

The recommendation systems used are based on two main types:

- UBCF (User Based Collaborative Filtering)
- IBCF (Item Based Collaborative Filtering)

trained on the same dataset, but separated with different techniques:

- Split
- Bootstrap (sampling with replacement)
- Cross Validation

with different comparison measures between items:

- Cosine Similarity
- Pearson Correlation

and with different measures for normalizing items:

- Center
- Z-Score

The aforementioned recommendation systems were also trained on a dataset where the average ratings are weighted using the timestamps of individual ratings.

Additionally, a binary recommendation system was implemented, where the values are converted from the scale $[0, 5]$ to the scale $[0, 1]$, using a threshold value of $3$. For this system, the only comparison measure is Jaccard similarity and there are no normalization metrics.

The complete report, which includes a comprehensive analysis of the data and a detailed explanation of each recommendation system, is available as a PDF inside the repository. The report is written in Italian.