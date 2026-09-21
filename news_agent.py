import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")

def get_news(query, limit=5):

    # Hard safety limit
    limit = min(limit, 10)

    url = "https://newsdata.io/api/1/latest"

    params = {
        "apikey": API_KEY,
        "q": query,
        "language": "en",
        "size": limit
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    articles = data.get("results", [])

    # Extra safety: never return more than requested
    return articles[:limit]


if __name__ == "__main__":
    articles = get_news("Indian Economy", limit=5)

    for i, article in enumerate(articles, 1):
        print(f"{i}. {article.get('title')}")
        print(article.get("link"))
        print()