import re
import pandas as pd
import numpy as np


INPUT_PATH = "all_movies.csv"
OUT_EDGES = "actor_movie_edges.csv"
OUT_ACTOR_YEAR = "actor_year_features.csv"
OUT_ACTOR_STATS = "actor_coverage_stats.csv"


def split_pipe(value) -> list[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]


def extract_imdb_id(url: str) -> str | None:
    if pd.isna(url):
        return None
    m = re.search(r"(tt\d+)", str(url))
    return m.group(1) if m else None


def box_office_outcome(roi: float) -> str | None:
    if pd.isna(roi) or np.isinf(roi):
        return None
    # Loss / Break-even / Success
    if roi < 1.0:
        return "Loss"
    if roi < 2.5:
        return "Break-even"
    return "Success"


def main():
    df = pd.read_csv(INPUT_PATH)

    required_cols = [
        "title", "url", "rating", "votes", "release_year",
        "genres", "runtime_minutes", "director",
        "actors", "roles", "popularity_index",
        # NEW:
        "budget", "box_office"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Lipsesc coloane obligatorii în CSV: {missing}")

    df = df.copy()

    # Parse imdb_id + lists
    df["imdb_id"] = df["url"].apply(extract_imdb_id)
    df["actors_list"] = df["actors"].apply(split_pipe)
    df["roles_list"] = df["roles"].apply(split_pipe)

    # Validate alignment actors <-> roles
    n_bad = (df["actors_list"].apply(len) != df["roles_list"].apply(len)).sum()
    if n_bad:
        bad_rows = df[df["actors_list"].apply(len) != df["roles_list"].apply(len)][
            ["title", "url", "actors", "roles"]
        ].head(10)
        raise ValueError(
            f"Am găsit {n_bad} rânduri unde #actors != #roles. "
            f"Primele exemple:\n{bad_rows.to_string(index=False)}"
        )

    # Explode to actor-movie edges (NOW includes budget + box_office)
    edges = (
        df[
            [
                "imdb_id", "title", "release_year", "genres", "director",
                "rating", "votes", "popularity_index", "runtime_minutes",
                "budget", "box_office",
                "actors_list", "roles_list"
            ]
        ]
        .explode(["actors_list", "roles_list"], ignore_index=True)
        .rename(columns={"actors_list": "actor", "roles_list": "role"})
    )

    # Cast order inside each movie
    edges["cast_order"] = edges.groupby("imdb_id").cumcount() + 1

    # Type cleanup
    edges["release_year"] = pd.to_numeric(edges["release_year"], errors="coerce").astype("Int64")
    edges["rating"] = pd.to_numeric(edges["rating"], errors="coerce")
    edges["votes"] = pd.to_numeric(edges["votes"], errors="coerce")
    edges["popularity_index"] = pd.to_numeric(edges["popularity_index"], errors="coerce")
    edges["runtime_minutes"] = pd.to_numeric(edges["runtime_minutes"], errors="coerce")

    # NEW numeric cleanup
    edges["budget"] = pd.to_numeric(edges["budget"], errors="coerce")
    edges["box_office"] = pd.to_numeric(edges["box_office"], errors="coerce")

    # NEW: ROI + Box office outcome
    edges["roi"] = edges["box_office"] / edges["budget"]
    edges["bo_outcome"] = edges["roi"].apply(box_office_outcome)

    # Save edges
    edges.to_csv(OUT_EDGES, index=False)
    print(f"[OK] Saved edges: {OUT_EDGES}  (rows={len(edges)})")

    # Actor-year aggregation
    def count_unique_genres(series: pd.Series) -> int:
        all_g = set()
        for g in series.dropna().astype(str):
            for part in g.split("|"):
                part = part.strip()
                if part:
                    all_g.add(part)
        return len(all_g)

    actor_year = (
        edges.groupby(["actor", "release_year"])
        .agg(
            n_movies=("imdb_id", "nunique"),
            lead_ratio=("role", lambda s: (s == "Principal").mean()),
            avg_rating=("rating", "mean"),
            avg_votes=("votes", "mean"),
            avg_popularity=("popularity_index", "mean"),
            n_genres=("genres", count_unique_genres),
            top_director=("director", lambda s: s.dropna().astype(str).value_counts().index[0] if len(s.dropna()) else None),

            # NEW: financial aggregates per actor-year
            avg_budget=("budget", "mean"),
            avg_box_office=("box_office", "mean"),
            avg_roi=("roi", "mean"),
            bo_outcome_mode=("bo_outcome", lambda s: s.dropna().value_counts().index[0] if len(s.dropna()) else None),
        )
        .reset_index()
        .rename(columns={"release_year": "year"})
    )

    actor_year.to_csv(OUT_ACTOR_YEAR, index=False)
    print(f"[OK] Saved actor-year features: {OUT_ACTOR_YEAR}  (rows={len(actor_year)})")

    # Actor coverage stats (useful to filter actors with enough history)
    actor_stats = (
        edges.groupby("actor")
        .agg(
            n_movies=("imdb_id", "nunique"),
            first_year=("release_year", "min"),
            last_year=("release_year", "max"),
            principal_ratio=("role", lambda s: (s == "Principal").mean()),
        )
        .reset_index()
    )
    actor_stats["span_years"] = actor_stats["last_year"].astype("Int64") - actor_stats["first_year"].astype("Int64")

    actor_stats.to_csv(OUT_ACTOR_STATS, index=False)
    print(f"[OK] Saved actor stats: {OUT_ACTOR_STATS}  (rows={len(actor_stats)})")

    # Quick summary
    print("\n--- Summary ---")
    print(f"Unique movies: {edges['imdb_id'].nunique()}")
    print(f"Unique actors: {edges['actor'].nunique()}")
    print(f"Total actor-movie rows: {len(edges)}")
    print(f"Actors with >=5 movies: {(actor_stats['n_movies'] >= 5).sum()}")
    print(f"Actors with span >=5 years: {(actor_stats['span_years'] >= 5).sum()}")

    # Extra sanity checks
    print("\n--- Financial coverage ---")
    print("Budget missing rate:", edges["budget"].isna().mean())
    print("Box office missing rate:", edges["box_office"].isna().mean())
    print("ROI valid rate:", edges["roi"].replace([np.inf, -np.inf], np.nan).notna().mean())


if __name__ == "__main__":
    main()
