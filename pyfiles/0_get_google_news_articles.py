import os
import json
import argparse
from typing import Dict, List, Set, Tuple, Any
from multiprocessing import Pool
from contextlib import suppress
from itertools import product

from GoogleNews import GoogleNews
from newspaper import Article
from newspaper.article import ArticleException
from tqdm import tqdm, trange

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, help='Name of the city to download articles for.')
    parser.add_argument('--save_path', type=str, help='path to the folder to save the articles to.')
    parser.add_argument('--load_path', type=str, help='path of the folder to load the JSON data from.')

    return parser.parse_args()


def get_news(query: str, pages: int=35) -> List[Dict[str, Any]]:
    """
    Search news defined by query.
    Returns a list of search results.
    
    Parameters
    ----------
    query: str
        The news search query to use.
        
    Returns
    -------
    news: list of news items.
        News list, each element in the list is a dictionary containing news details like title, date, URL etc.
    """
    
    googlenews = GoogleNews(start='01/01/2010',end='01/01/2015')
    googlenews.search(query)
    news = []
    for page in tqdm(range(pages), leave=False):
        googlenews.get_page(page)
        news += googlenews.results()
        
    return news
    
def get_article(news_item: Dict[str, Any], save_path: str) -> None:
    """
    Downloads a item from the URL provided by the news_item dict.
    
    Parameters
    ----------
    news_item: Dict[str, any]
        A single news_item which contains fields like: date, link, title etc. etc.

    save_path: str
        Location to save the news article to.
    """
    with suppress(ArticleException):
        article = Article(news_item['link'])
        article.download()
        article.parse()
        with open(os.path.join(save_path, f'{news_item["title"].replace("/", " ")}.txt'), 'w') as f:
            f.write(article.text)
            
            
def load_city_json(path: str, city: str) -> Dict[str, List[str]]:
    """
    Loads a city JSON file.
    
    Parameters
    ----------
    path: str
        Path to the JSON file.
        
    city: str
        name of the city JSON to load.
        
    Returns
    -------
    locations_json: Dict[str, List[str]]
        Dictionary that maps areas to list of neighborhoods in the area.
    """
    with open(os.path.join(path, f'{city}.json'), 'r') as f:
        locations_json = json.loads(f.read())
    
    return locations_json
    
def get_articles(city_locations: Dict[str, List[str]], city: str, save_path: str) -> None:
    
    """
    Fetch all articles for all neighborhoods in the city defined by the city_locations JSON.
    
    Parameters
    ----------
    city_locations: Dict[str, List[str]]
        Dictionary that maps areas to list of neighborhoods in the area.
        
    city: str
        name of the city we are getting articles for.
        
    save_path: str
        Inital save path for the articles.
    """
    
    for area, neighborhoods in tqdm(city_locations.items(), leave=False):
        
        for neighborhood in tqdm(neighborhoods, leave=False):
            path = os.path.join(save_path, city, area, neighborhood)
            if not os.path.exists(path):
                os.makedirs(path)
            news_articles = get_news(f'{city} {neighborhood}')
            with Pool(5) as p:
                p.starmap(get_article, product(news_articles, [path]))


def main():

    parser = parse_args()

    city_areas = load_city_json(parser.load_path, parser.city)
    
    get_articles(city_areas, parser.city, parser.save_path)
 
    
if __name__ == "__main__":
    main()
