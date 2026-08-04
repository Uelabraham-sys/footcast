# FootCast

FootCast is a production-oriented football analytics and probabilistic match
forecasting platform.

It collects historical and current Premier League match data, preserves raw
source records, creates leakage-safe team-strength features, trains
probabilistic models and serves fixture forecasts through an interactive
dashboard.

## Initial prediction problem

For every scheduled Premier League fixture, FootCast estimates:

- probability of a home win;
- probability of a draw;
- probability of an away win;
- expected home and away goals;
- most likely scoreline.

The model must use only information that was available before kick-off.

## Initial project scope

Version 0.1 covers:

- one competition: the English Premier League;
- historical and current fixtures and results;
- Bronze, Silver and Gold data layers;
- rolling form and Elo-rating features;
- chronological model validation;
- Poisson, logistic-regression and gradient-boosting models;
- a Streamlit dashboard;
- automated tests, CI and Docker execution.

## Architecture

```text
Historical data       Current football API
       │                       │
       └──────────┬────────────┘
                  ▼
            Bronze raw data
                  ▼
        Silver canonical matches
                  ▼
      Gold features and predictions
                  ▼
       Models and dashboard
## Current implementation status

### Completed

- project structure and environment configuration;
- historical Premier League CSV ingestion;
- current fixtures, results and standings API ingestion;
- raw CSV and JSON preservation;
- Bronze Parquet datasets;
- ingestion manifests and SHA-256 checksums;
- automated validation and Bronze audit reporting;
- unit tests, static type checking and code formatting.

### Next

- canonical team identity mapping;
- historical and current match deduplication;
- Silver match table;
- leakage-safe rolling form features;
- chronological Elo ratings.