import pandas as pd
from duckduckgo_search import DDGS


def search_query(query):
    results = DDGS().text(
        keywords = str(query),
        max_results = 5,
        region = 'wt-wt',
        timelimit = '7d',
        safesearch = 'on' 
    )

    results_df = pd.DataFrame(results)
    return results_df
    # results_df.to_csv('search_results.csv', index=False)


# search_query(search_query)
