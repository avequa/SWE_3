"""Подготовка данных MegaMarket

Датасет - данные маркетплейса Мегамаркет (196 млн действий 2,7 млн покупателей за 4 месяца)
Каждая строка — кто (user_id), когда (datetime), что сделал (event), с каким товаром (item_id),
категория товара (category_id) и цена (price, измененная)

Берём выборку каждого 64-го покупателя по номеру, результат в data/sample.parquet - с ним работают остальные скрипты

Запуск:  python prepare_data.py
"""

import glob

import pyarrow.compute as pc
import pyarrow.dataset as ds

EVERY = 64
EVENTS = {0: "заказ", 1: "избранное", 2: "просмотр", 3: "корзина"}

files = [f for f in glob.glob("data/**/*.parquet", recursive=True) if not f.endswith("sample.parquet")]
if not files:
    raise SystemExit("Не найден файл датасета: скачайте MegaMarket и положите .parquet в папку data/")

print(f"Читаю {files[0]} ...")
every_64th = pc.equal(pc.bit_wise_and(ds.field("user_id"), EVERY - 1), 0)
df = ds.dataset(files[0]).to_table(filter=every_64th).to_pandas()
if df.empty:
    raise SystemExit("В выборку не попал ни один покупатель — проверьте файл датасета")

print(f"Строк: {len(df)}")
print(f"Покупателей: {df['user_id'].nunique()}, товаров: {df['item_id'].nunique()}, "
      f"категорий: {df['category_id'].nunique()}")
print(f"Период: {df['datetime'].min():%d.%m.%Y} – {df['datetime'].max():%d.%m.%Y}")
print("Действия:")
for event, count in df["event"].value_counts().sort_index().items():
    print(f"  {EVENTS[event]:<10} {count}")

df.to_parquet("data/sample.parquet", index=False)
print("Сохранено в data/sample.parquet")