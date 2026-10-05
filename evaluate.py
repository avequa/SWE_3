"""Сравнение методов рекомендаций

Методы видят всё до последних 4 недель, а проверяем их на заказах последних 4 недель.
Каждому покупателю рекомендуем 10 товаров и смотрим, что из этого он действительно заказал потом

Метрики:
    Precision@10 — доля угаданных среди 10 рекомендованных
    Recall@10    — доля будущих заказов, попавших в рекомендации
    HitRate@10   — доля покупателей, у которых угадан хотя бы один товар
    Coverage     — какая доля каталога хоть раз попала в рекомендации

Запуск:  python evaluate.py
"""

import random
import time

from recommenders import ORDER, Recommender, load, split

K = 10
N_USERS = 2000

history, future = split(load())
print(f"История: {len(history)} действий, проверка: {len(future)} действий (последние 4 недели)")

start = time.perf_counter()
rec = Recommender(history)
print(f"Подготовка методов: {time.perf_counter() - start:.1f} с")

future_orders = future[future["event"] == ORDER].groupby("user_id")["item_id"].agg(set)
new_users = 0
new_items = set()
truth = {}
for user, items in future_orders.items():
    if user not in rec.user_index:
        new_users += 1
        continue
    items = items - rec.ordered.get(user, set())
    new_items |= {item for item in items if item not in rec.item_index}
    items = {item for item in items if item in rec.item_index}
    if items:
        truth[user] = items

print(f"Покупателей с заказами в проверочный период: {len(future_orders)}, "f"из них новых (холодный старт): {new_users}")
print(f"Товаров, которых не было в истории: {len(new_items)}")
print(f"Покупателей для оценки: {len(truth)}, берём случайных {min(N_USERS, len(truth))}")

random.seed(42)
users = random.sample(sorted(truth), min(N_USERS, len(truth)))

methods = {
    "Популярное": lambda u: rec.popular(u, K),
    "Недавно смотрели": lambda u: rec.recently_viewed(u, K),
    "Content-based": lambda u: rec.content_based(u, K),
    "Item-based CF": lambda u: rec.item_based(u, K),
    "Гибрид": lambda u: rec.hybrid(u, K)[0],
}

print(f"\n{'Метод':<20}{'Precision':>11}{'Recall':>9}{'HitRate':>9}{'Coverage':>10}{'мс':>8}")
for name, recommend in methods.items():
    hits_total = recall_total = hit_users = 0
    shown = set()
    start = time.perf_counter()
    for user in users:
        items = recommend(user)
        hits = len(set(items) & truth[user])
        hits_total += hits
        recall_total += hits / len(truth[user])
        hit_users += hits > 0
        shown |= set(items)
    ms = (time.perf_counter() - start) / len(users) * 1000
    n = len(users)
    print(f"{name:<20}{hits_total / (n * K):>11.3f}{recall_total / n:>9.3f}"
          f"{hit_users / n:>9.3f}{len(shown) / len(rec.items):>10.3f}{ms:>8.1f}")
