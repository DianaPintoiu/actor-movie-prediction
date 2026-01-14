import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

EDGES_PATH = "actor_movie_edges.csv"
AY_PATH = "actor_year_features.csv"
STATS_PATH = "actor_coverage_stats.csv"

TOP_GENRES = 8
GEN_WINDOW_YEARS = 5
NEXT_YEARS = 3


def split_pipe(s):
    if pd.isna(s):
        return []
    return [x.strip() for x in str(s).split("|") if x.strip()]

def success_level(r, p):
    if pd.isna(r) or pd.isna(p):
        return None
    if (p < 45) or (r < 5.8):
        return "Miss"
    if (p >= 75) and (r >= 7.0):
        return "Hit"
    return "Mid"

def bo_outcome(roi):
    if pd.isna(roi) or np.isinf(roi):
        return None
    if roi < 1.0:
        return "Loss"
    if roi < 2.5:
        return "Break-even"
    return "Success"

def mode_next_k(series, i, k):
    w = series.iloc[i+1:i+1+k].dropna()
    if len(w) == 0:
        return None
    return w.value_counts().index[0]

def any_next_k(series, i, k, label):
    w = series.iloc[i+1:i+1+k].dropna()
    if len(w) == 0:
        return None
    return int((w == label).any())


# load data
edges = pd.read_csv(EDGES_PATH)
ay = pd.read_csv(AY_PATH)
stats = pd.read_csv(STATS_PATH)

edges["release_year"] = pd.to_numeric(edges["release_year"], errors="coerce").astype("Int64")
ay["year"] = pd.to_numeric(ay["year"], errors="coerce").astype("Int64")


# filter actors by coverage
stats["n_movies"] = pd.to_numeric(stats["n_movies"], errors="coerce")
stats["span_years"] = pd.to_numeric(stats["span_years"], errors="coerce")

keep = stats[(stats["n_movies"] >= 5) & (stats["span_years"] >= 5)]["actor"]
edges = edges[edges["actor"].isin(keep)]
ay = ay[ay["actor"].isin(keep)]


# build actr_year labels
edges["genres_list"] = edges["genres"].apply(split_pipe)
edges_gen = edges.explode("genres_list").rename(columns={"genres_list": "genre"})

actor_year_gen = (
    edges_gen.groupby(["actor", "release_year"])["genre"]
    .agg(lambda s: s.value_counts().index[0] if len(s) else None)
    .reset_index()
    .rename(columns={"release_year": "year", "genre": "dominant_genre"})
)

edges["rating"] = pd.to_numeric(edges["rating"], errors="coerce")
edges["popularity_index"] = pd.to_numeric(edges["popularity_index"], errors="coerce")
edges["success_level_movie"] = [
    success_level(r, p) for r, p in zip(edges["rating"], edges["popularity_index"])
]

actor_year_success = (
    edges.groupby(["actor", "release_year"])["success_level_movie"]
    .agg(lambda s: s.dropna().value_counts().index[0] if len(s.dropna()) else None)
    .reset_index()
    .rename(columns={"release_year": "year", "success_level_movie": "success_level"})
)

edges["budget"] = pd.to_numeric(edges["budget"], errors="coerce")
edges["box_office"] = pd.to_numeric(edges["box_office"], errors="coerce")
edges["roi"] = edges["box_office"] / edges["budget"]
edges["bo_outcome_movie"] = edges["roi"].apply(bo_outcome)

actor_year_bo = (
    edges.groupby(["actor", "release_year"])["bo_outcome_movie"]
    .agg(lambda s: s.dropna().value_counts().index[0] if len(s.dropna()) else None)
    .reset_index()
    .rename(columns={"release_year": "year", "bo_outcome_movie": "box_office_outcome"})
)


# merge all labels
data = ay.merge(actor_year_gen, on=["actor", "year"], how="left")
data = data.merge(actor_year_success, on=["actor", "year"], how="left")
data = data.merge(actor_year_bo, on=["actor", "year"], how="left")
data = data.sort_values(["actor", "year"])

# reduce genres to top N + Other
top_genres = set(data["dominant_genre"].value_counts().head(TOP_GENRES).index)
data["dominant_genre"] = data["dominant_genre"].apply(
    lambda g: g if g in top_genres else "Other"
)

# build targets
def build_targets(g):
    g = g.sort_values("year").reset_index(drop=True)

    g["dominant_genre_next3y"] = [
        mode_next_k(g["dominant_genre"], i, NEXT_YEARS) for i in range(len(g))
    ]
    g["hit_next3y"] = [
        any_next_k(g["success_level"], i, NEXT_YEARS, "Hit") for i in range(len(g))
    ]
    g["box_success_next3y"] = [
        any_next_k(g["box_office_outcome"], i, NEXT_YEARS, "Success") for i in range(len(g))
    ]
    return g

data = data.groupby("actor", group_keys=False).apply(build_targets)

# get features and targets
feature_cols = [
    c for c in [
        "n_movies", "lead_ratio", "avg_rating",
        "avg_votes", "avg_popularity", "n_genres",
        "avg_budget", "avg_box_office", "avg_roi"
    ] if c in data.columns
]

targets = [
    "dominant_genre_next3y",
    "hit_next3y",
    "box_success_next3y"
]

ml = data.dropna(subset=feature_cols + targets)

# split train/test
split_year = int(ml["year"].quantile(0.8))
train = ml[ml["year"] <= split_year]
test = ml[ml["year"] > split_year]

X_train, X_test = train[feature_cols], test[feature_cols]

# logistic regression
for target in targets:
    print(f"\n=== LOGISTIC REGRESSION — {target} ===")

    le = LabelEncoder()
    y_train = le.fit_transform(train[target].astype(str))
    y_test = le.transform(test[target].astype(str))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
        ))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(preds),
        zero_division=0
    ))
