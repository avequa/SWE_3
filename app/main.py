"""Сервис рекомендаций для маркетплейса

Каждый адрес это блок на сайте магазина:
    /recommendations/{user_id}       «Вам может понравиться»
    /popular                         «Популярное»
    /users/{user_id}/recently-viewed «Вы недавно смотрели»
    /items/{item_id}/bought-together «С этим товаром покупают»
    /events/click                    клик покупателя по рекомендации

Сервис знает историю только до последних 4 недель

Запуск:        uvicorn app.main:app --port 8001
Документация:  http://127.0.0.1:8001/docs
"""

import logging
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

from recommenders import Recommender, load, split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("recsys")

app = FastAPI(title="RecSys API", description="Рекомендации товаров для маркетплейса", version="1.0.0")

history, _ = split(load())
rec = Recommender(history)
log.info("Загружено: %d покупателей, %d товаров", len(rec.user_index), len(rec.items))

Block = Literal["personal", "popular", "recently_viewed", "bought_together"]

# метрики для Prometheus

REQUESTS = Counter("recsys_requests_total", "Запросы к блокам рекомендаций", ["block"])
LATENCY = Histogram("recsys_request_seconds", "Время подбора рекомендаций", ["block"])
IMPRESSIONS = Counter("recsys_impressions_total", "Сколько товаров показано в блоке", ["block"])
CLICKS = Counter("recsys_clicks_total", "Клики по рекомендованным товарам", ["block"])
SOURCE = Counter("recsys_personal_source_total", "Каким методом собраны персональные рекомендации", ["source"])
COLD_START = Counter("recsys_cold_start_total", "Запросы от новых покупателей без истории")
COVERAGE = Gauge("recsys_catalog_coverage", "Доля каталога, хоть раз попавшая в рекомендации")

shown_items = set()


def track(block, start, items):
    """Записать метрики блока и вернуть товары в виде обычных чисел для JSON"""
    LATENCY.labels(block).observe(time.perf_counter() - start)
    REQUESTS.labels(block).inc()
    IMPRESSIONS.labels(block).inc(len(items))
    shown_items.update(items)
    COVERAGE.set(len(shown_items) / len(rec.items))
    return [int(item) for item in items]


# адреса API

@app.get("/health")
def health():
    return {"status": "ok", "users": len(rec.user_index), "items": len(rec.items)}


@app.get("/recommendations/{user_id}")
def recommendations(user_id: int, k: int = Query(10, ge=1, le=50)):
    """«Вам может понравиться»: гибрид item-based, content-based и популярного"""
    start = time.perf_counter()
    items, source = rec.hybrid(user_id, k)
    if user_id not in rec.user_index:
        COLD_START.inc()
    SOURCE.labels(source).inc()
    return {"user_id": user_id, "source": source, "items": track("personal", start, items)}


@app.get("/popular")
def popular(k: int = Query(10, ge=1, le=50)):
    """«Популярное»: больше всего заказов за последнюю неделю"""
    start = time.perf_counter()
    return {"items": track("popular", start, rec.popular(None, k))}


@app.get("/users/{user_id}/recently-viewed")
def recently_viewed(user_id: int, k: int = Query(10, ge=1, le=50)):
    """«Вы недавно смотрели»"""
    start = time.perf_counter()
    return {"user_id": user_id, "items": track("recently_viewed", start, rec.recently_viewed(user_id, k))}


@app.get("/items/{item_id}/bought-together")
def bought_together(item_id: int, k: int = Query(10, ge=1, le=50)):
    """«С этим товаром покупают» для карточки товара"""
    if item_id not in rec.item_index:
        raise HTTPException(404, "Товар не найден")
    start = time.perf_counter()
    return {"item_id": item_id, "items": track("bought_together", start, rec.bought_together(item_id, k))}


class Click(BaseModel):
    user_id: int
    item_id: int
    block: Block


@app.post("/events/click")
def click(event: Click):
    """Покупатель кликнул по рекомендованному товару"""
    CLICKS.labels(event.block).inc()
    log.info("click: user=%d item=%d block=%s", event.user_id, event.item_id, event.block)
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
