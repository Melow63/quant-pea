"""
NLP sentiment engine — v2.
- FinBERT (ProsusAI/finbert) pour sentiment financier
- Source: Google News RSS (gratuit, stable, pas de blocage)
- Fallback: analyse par mots-clés financiers FR/EN
"""
from __future__ import annotations

import time
import urllib.parse
from datetime import date
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger


_pipeline = None

def get_finbert_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline
        logger.info("Chargement FinBERT (première fois ~1 min)...")
        _pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT chargé")
        return _pipeline
    except Exception as e:
        logger.warning(f"FinBERT indisponible: {e} — fallback mots-clés")
        return None


def score_texts(texts: list[str]) -> list[dict]:
    pipe = get_finbert_pipeline()
    results = []
    if pipe is not None:
        try:
            preds = pipe(texts, batch_size=8, truncation=True)
            for pred in preds:
                label = pred["label"].lower()
                score = pred["score"]
                numeric = score if label == "positive" else (-score if label == "negative" else 0.0)
                results.append({"label": label, "score": score, "numeric": numeric})
            return results
        except Exception as e:
            logger.warning(f"FinBERT erreur: {e}")
    return [_keyword_sentiment(t) for t in texts]


def _keyword_sentiment(text: str) -> dict:
    t = text.lower()
    pos = ["hausse", "progression", "croissance", "bénéfice", "profit", "record",
           "fort", "solide", "relèvement", "optimiste", "rebond", "surperformance",
           "buy", "upgrade", "beat", "raised", "growth", "strong", "acquisition",
           "dividende", "rachat", "commande", "contrat", "partenariat"]
    neg = ["baisse", "chute", "perte", "recul", "avertissement", "risque", "crise",
           "abaissement", "pessimiste", "vente", "sell", "miss", "cut", "downgrade",
           "weak", "decline", "loss", "litige", "amende", "fraude", "dette",
           "restructuration", "licenciement", "faillite"]
    p = sum(1 for w in pos if w in t)
    n = sum(1 for w in neg if w in t)
    total = p + n
    if total == 0:
        return {"label": "neutral", "score": 0.5, "numeric": 0.0}
    numeric = (p - n) / total
    label = "positive" if numeric > 0.1 else "negative" if numeric < -0.1 else "neutral"
    return {"label": label, "score": abs(numeric), "numeric": numeric}


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
}

def fetch_google_news_rss(query: str, max_results: int = 10, lang: str = "fr") -> list[dict]:
    """Google News RSS — gratuit, fiable, pas de blocage."""
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl={lang}&gl=FR&ceid=FR:{lang.upper()}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "xml")
        items = soup.find_all("item")
        results = []
        for item in items[:max_results]:
            title = item.find("title")
            source = item.find("source")
            if title and title.get_text(strip=True):
                raw = title.get_text(strip=True)
                clean = raw.rsplit(" - ", 1)[0] if " - " in raw else raw
                results.append({
                    "headline": clean,
                    "date": date.today().isoformat(),
                    "source": source.get_text(strip=True) if source else "google_news",
                })
        logger.debug(f"Google News '{query}': {len(results)} articles")
        return results
    except Exception as e:
        logger.warning(f"Google News RSS '{query}': {e}")
        return []


def get_news_for_security(
    security_id: str,
    asset_name: str,
    yahoo_ticker: Optional[str] = None,
    max_results: int = 8,
) -> list[dict]:
    all_news = []
    queries = [asset_name]
    if yahoo_ticker:
        base = yahoo_ticker.replace(".PA", "").replace(".SG", "")
        if base not in asset_name:
            queries.append(f"{base} bourse action")
    seen = set()
    for query in queries[:2]:
        news = fetch_google_news_rss(query, max_results=max_results)
        for item in news:
            h = item["headline"]
            if h not in seen:
                seen.add(h)
                item["security_id"] = security_id
                all_news.append(item)
        time.sleep(0.3)
    return all_news[:max_results]


def compute_portfolio_sentiment(
    security_master: pd.DataFrame,
    use_finbert: bool = True,
) -> pd.DataFrame:
    results = []
    for _, row in security_master.iterrows():
        sec_id = str(row.get("security_id", ""))
        name = str(row.get("asset_name", sec_id))
        ticker = str(row.get("yahoo_ticker", "")) if row.get("yahoo_ticker") else None
        asset_class = str(row.get("asset_class", "")).lower()

        if "fund" in asset_class or "fcp" in asset_class:
            results.append({"security_id": sec_id, "sentiment_score": 0.0,
                            "sentiment_label": "neutral", "n_articles": 0,
                            "confidence": 0.0, "sample_headline": "—"})
            continue

        logger.info(f"News: {name}")
        news = get_news_for_security(sec_id, name, yahoo_ticker=ticker)

        if not news:
            results.append({"security_id": sec_id, "sentiment_score": 0.0,
                            "sentiment_label": "neutral", "n_articles": 0,
                            "confidence": 0.0, "sample_headline": "Aucune news"})
            continue

        headlines = [n["headline"] for n in news]
        scores = score_texts(headlines) if use_finbert else [_keyword_sentiment(h) for h in headlines]
        numerics = [s["numeric"] for s in scores]
        avg_score = float(sum(numerics) / len(numerics))
        avg_conf = float(sum(s["score"] for s in scores) / len(scores))
        label = "positive" if avg_score > 0.05 else "negative" if avg_score < -0.05 else "neutral"

        results.append({
            "security_id": sec_id,
            "sentiment_score": round(avg_score, 4),
            "sentiment_label": label,
            "n_articles": len(headlines),
            "confidence": round(avg_conf, 4),
            "sample_headline": (headlines[0][:80] + "...") if len(headlines[0]) > 80 else headlines[0],
        })
        logger.success(f"{sec_id}: {label} ({avg_score:.3f}) — {len(headlines)} articles")

    return pd.DataFrame(results)
