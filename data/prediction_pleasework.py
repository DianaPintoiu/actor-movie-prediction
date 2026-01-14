import pandas as pd
import numpy as np

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

# =========================
# Paths
# =========================
EDGES_PATH = "actor_movie_edges.csv"
AY_PATH = "actor_year_features.csv"
STATS_PATH = "actor_coverage_stats.csv"

TOP_GENRES = 8          # restul -> Other
GEN_WINDOW_YEARS = 5    # preferințe de gen pe ultimii 5 ani


def split_pipe(s):
    if pd.isna(s):
        return []
    return [x.strip() for x in str(s).split("|") if x.strip()]


def success_level(r, p):
    """
    Rebalansat față de varianta ta inițială, ca să nu ai Miss ~ 0 exemple.
    Ajustează pragurile dacă vrei.
    """
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


# =========================
# Load
# =========================
edges = pd.read_csv(EDGES_PATH)
ay = pd.read_csv(AY_PATH)
stats = pd.read_csv(STATS_PATH)

edges["release_year"] = pd.to_numeric(edges["release_year"], errors="coerce").astype("Int64")
ay["year"] = pd.to_numeric(ay["year"], errors="coerce").astype("Int64")

# =========================
# Filter actors (no leakage: only filtering)
# =========================
stats["span_years"] = pd.to_numeric(stats["span_years"], errors="coerce")
stats["n_movies"] = pd.to_numeric(stats["n_movies"], errors="coerce")
keep = stats[(stats["n_movies"] >= 5) & (stats["span_years"] >= 5)]["actor"]

edges = edges[edges["actor"].isin(keep)].copy()
ay = ay[ay["actor"].isin(keep)].copy()

# =========================
# Build actor-year labels from edges
# =========================
edges["genres_list"] = edges["genres"].apply(split_pipe)

# dominant genre per actor-year
edges_gen = edges.explode("genres_list").rename(columns={"genres_list": "genre"})
actor_year_gen = (
    edges_gen.groupby(["actor", "release_year"])["genre"]
    .agg(lambda s: s.value_counts().index[0] if len(s) else None)
    .reset_index()
    .rename(columns={"release_year": "year", "genre": "dominant_genre"})
)

# success level per actor-year
edges["rating"] = pd.to_numeric(edges["rating"], errors="coerce")
edges["popularity_index"] = pd.to_numeric(edges["popularity_index"], errors="coerce")
edges["success_level_movie"] = [success_level(r, p) for r, p in zip(edges["rating"], edges["popularity_index"])]

actor_year_success = (
    edges.groupby(["actor", "release_year"])["success_level_movie"]
    .agg(lambda s: s.dropna().value_counts().index[0] if len(s.dropna()) else None)
    .reset_index()
    .rename(columns={"release_year": "year", "success_level_movie": "success_level"})
)

# box office outcome per actor-year (dacă ai budget/box_office)
has_bo = ("budget" in edges.columns) and ("box_office" in edges.columns)
actor_year_bo = None

if has_bo:
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
else:
    print("⚠️ budget/box_office not found in edges -> box office target will be skipped.")


# =========================
# Merge labels onto actor-year features
# =========================
data = ay.merge(actor_year_gen, on=["actor", "year"], how="left")
data = data.merge(actor_year_success, on=["actor", "year"], how="left")
if actor_year_bo is not None:
    data = data.merge(actor_year_bo, on=["actor", "year"], how="left")

data = data.sort_values(["actor", "year"])

# =========================
# Reduce rare genres: top N + Other
# =========================
genre_counts = data["dominant_genre"].value_counts(dropna=True)
top_genres = set(genre_counts.head(TOP_GENRES).index)

def map_genre(g):
    if pd.isna(g):
        return None
    return g if g in top_genres else "Other"

data["dominant_genre"] = data["dominant_genre"].apply(map_genre)

# =========================
# Add genre-preference features (shares in last 5 years)
# =========================
edges_gen["genre"] = edges_gen["genre"].apply(lambda g: g if g in top_genres else "Other")

gy = (
    edges_gen.groupby(["actor", "release_year", "genre"])
    .size()
    .rename("cnt")
    .reset_index()
    .rename(columns={"release_year": "year"})
)

gy = gy.sort_values(["actor", "year"])

gy_pivot = gy.pivot_table(index=["actor", "year"], columns="genre", values="cnt", aggfunc="sum", fill_value=0).reset_index()
genre_cols = [c for c in gy_pivot.columns if c not in ["actor", "year"]]

def roll_genre_shares(g):
    g = g.sort_values("year").copy()
    # rolling sums
    for col in genre_cols:
        g[f"{col}_cnt_last5y"] = g[col].rolling(GEN_WINDOW_YEARS, min_periods=1).sum()
    g["genre_total_last5y"] = g[[f"{c}_cnt_last5y" for c in genre_cols]].sum(axis=1)

    # shares
    for col in genre_cols:
        g[f"{col}_share_last5y"] = (
            g[f"{col}_cnt_last5y"] / g["genre_total_last5y"].replace(0, np.nan)
        ).fillna(0.0)

    keep_cols = ["actor", "year"] + [f"{c}_share_last5y" for c in genre_cols]
    return g[keep_cols]

genre_pref = gy_pivot.groupby("actor", group_keys=False).apply(roll_genre_shares)

data = data.merge(genre_pref, on=["actor", "year"], how="left")
for c in data.columns:
    if c.endswith("_share_last5y"):
        data[c] = data[c].fillna(0.0)

# =========================
# Next-year targets (t -> t+1)
# =========================
data["dominant_genre_next"] = data.groupby("actor")["dominant_genre"].shift(-1)
data["success_level_next"] = data.groupby("actor")["success_level"].shift(-1)
if actor_year_bo is not None:
    data["box_office_outcome_next"] = data.groupby("actor")["box_office_outcome"].shift(-1)

target_cols = ["dominant_genre_next", "success_level_next"] + (["box_office_outcome_next"] if actor_year_bo is not None else [])

# =========================
# Features (numeric + genre shares + (optional) financial aggregates if present in actor_year)
# =========================
base_num = ["n_movies", "lead_ratio", "avg_rating", "avg_votes", "avg_popularity", "n_genres"]
base_num = [c for c in base_num if c in data.columns]

# dacă ai în actor_year_features și financial aggregates (din scriptul nou)
financial = [c for c in ["avg_budget", "avg_box_office", "avg_roi"] if c in data.columns]

share_cols = [c for c in data.columns if c.endswith("_share_last5y")]

feature_cols = base_num + financial + share_cols

ml = data.dropna(subset=target_cols + feature_cols).copy()

# =========================
# Temporal split (train old, test recent)
# =========================
split_year = int(ml["year"].quantile(0.8))
train = ml[ml["year"] <= split_year]
test = ml[ml["year"] > split_year]

X_train, Y_train = train[feature_cols], train[target_cols]
X_test, Y_test = test[feature_cols], test[target_cols]

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Targets:", target_cols)
print("Top genres:", sorted(list(top_genres)))

# =========================
# One single model (multi-output)
# Using boosting (usually better than RF on tabular)
# =========================
model = MultiOutputClassifier(
    HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        random_state=42
    )
)

model.fit(X_train, Y_train)
pred = model.predict(X_test)

# =========================
# Reports
# =========================
for i, col in enumerate(target_cols):
    print("\n==============================")
    print("Target:", col)
    print(classification_report(Y_test[col], pred[:, i], zero_division=0))
