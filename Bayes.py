import csv
import pandas as pd
import re

# Зчитування файлу з даними
data = pd.read_csv("sets/spam.csv", usecols=[0, 1], names=["SpamOrNot", "text"])
# Підрахунок статистики
stat = dict()
for text in data[data["SpamOrNot"] == "spam"]["text"]:
    text1 = re.sub(r'[^\w\s]', ' ', text.lower())
    for word in text1.split(" "):
        if len(word) > 2:
            if stat.get(word):
                stat[word][0] += 1
            else:
                stat[word] = [0, 0]
                stat[word][0] = 1

for text in data[data["SpamOrNot"] == "ham"]["text"]:
    text1 = re.sub(r'[^\w\s]', ' ', text.lower())
    for word in text1.split(" "):
        if len(word) > 2:
            if stat.get(word):
                stat[word][1] += 1
            else:
                stat[word] = [0, 0]
                stat[word][1] = 1

stat1 = pd.DataFrame.from_dict(stat, orient='index', columns=["spam", "ham"])
stat1["sum"] = stat1.sum(axis=1)

# Ймовірність слів
stat1["spam"] = stat1["spam"] / stat1["sum"]
stat1["ham"] = stat1["ham"] / stat1["sum"]

# Нормована ймовірність слів
stat1["spam"] = (stat1["sum"] * stat1["spam"] + 0.5) / (stat1["sum"] + 1)
stat1["ham"] = (stat1["sum"] * stat1["ham"] + 0.5) / (stat1["sum"] + 1)

# Статистика по словах
print(stat1)

# Зчитування тестових даних
test = pd.read_csv("sets/test.csv", usecols=[0, 1], names=["SpamOrNot", "text"])
test["spam"] = 0
test["ham"] = 0
test["prediction"] = ""
test["correct"] = 0
for i, text in test.iterrows():
    text1 = re.sub(r'[^\w\s]', ' ', text["text"].lower())
    spam = 0.5
    ham = 0.5
    for word in text1.split(" ")[:1]:
        if len(word) > 2:
            spam *= stat1['spam'][word]
            ham *= stat1['ham'][word]

    test.loc[i, "spam"] = spam
    test.loc[i, "ham"] = ham

    if ham >= spam:
        test.loc[i, "prediction"] = "ham"
    else:
        test.loc[i, "prediction"] = "spam"
    if test.loc[i, "SpamOrNot"] == test.loc[i, "prediction"]:
        test.loc[i, "correct"] = 1

# Кількість коректно предугатаних повідомлень
print(test.groupby("correct").count())
print(test)
test.to_csv("sets/result.csv")

