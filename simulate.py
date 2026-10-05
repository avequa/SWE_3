"""Имитация покупателей, чтобы на графиках Grafana появились метрики рекомендаций

Скрипт берёт покупателей, которые делали заказы в эпоследнние 4 недели, 
запрашивает для них рекомендации и отправляет «клик», если покупатель потом действительно заказал рекомендованный товар. 
CTR считается по реальному поведению людей

Запуск:  python simulate.py
"""

import random
import time

import httpx2

from recommenders import ORDER, load, split

URL = "http://localhost:8001"
STEPS = 500

history, future = split(load())
ordered_later = future[future["event"] == ORDER].groupby("user_id")["item_id"].agg(set).to_dict()
seen_before = history.groupby("user_id")["item_id"].agg(list).to_dict()
regular = [u for u in ordered_later if u in seen_before]        # покупатели с историей
newcomers = [u for u in ordered_later if u not in seen_before]  # новые , т.е. холодный старт

random.seed(42)


def show_and_click(user, response, block):
    """Покупатель видит блок и кликает по товарам, которые потом заказал"""
    for item in response.json().get("items", []):
        if item in ordered_later.get(user, set()):
            httpx2.post(f"{URL}/events/click", json={"user_id": int(user), "item_id": item, "block": block})


for step in range(1, STEPS + 1):
    chance = random.random()
    if chance < 0.5:
        # постоянный покупатель на главной странице
        user = random.choice(regular)
        show_and_click(user, httpx2.get(f"{URL}/recommendations/{user}"), "personal")
    elif chance < 0.6 and newcomers:
        # новый покупатель - истории нет, сервис отдаст популярное
        user = random.choice(newcomers)
        show_and_click(user, httpx2.get(f"{URL}/recommendations/{user}"), "personal")
    elif chance < 0.85:
        # покупатель открыл карточку товара, который смотрел раньше
        user = random.choice(regular)
        item = random.choice(seen_before[user])
        show_and_click(user, httpx2.get(f"{URL}/items/{item}/bought-together"), "bought_together")
    else:
        user = random.choice(regular)
        show_and_click(user, httpx2.get(f"{URL}/users/{user}/recently-viewed"), "recently_viewed")
        show_and_click(user, httpx2.get(f"{URL}/popular"), "popular")

    if step % 50 == 0:
        print(f"Шаг {step} из {STEPS}")
    time.sleep(random.uniform(0.05, 0.2))
