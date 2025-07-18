##################################################
# This script is part of a chatbot project that provides information about anime.
# It includes functions to retrieve anime data based on various criteria.
# The dataset is preprocessed to ensure data quality and relevance.
# The chatbot can handle requests for top anime, genre-specific anime, seasonal anime, and more.
# The OpenAI API is used as base for the chatbot's functionality.
##################################################


##################################################
# Preliminary setup
##################################################

# Import necessary libraries and set up the OpenAI client
import json  # Import the json module for handling JSON data
import pandas as pd  # Import pandas for data manipulation
import matplotlib.pyplot as plt  # Import matplotlib for creating plots
import numpy as np  # Import numpy for numerical operations
from openai import OpenAI  # Import the OpenAI class to interact with the OpenAI API
from dotenv import load_dotenv # Import load_dotenv to load environment variables from a .env file
import os  # Import os to interact with the operating system

load_dotenv() # Load environment variables from the .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Retrieve the OpenAI API key from the environment variable
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=OPENAI_API_KEY) # Create an instance of the OpenAI client using the API key


##################################################
# Loading of the dataset and preprocessing
##################################################

# Load the anime dataset from a csv file
df = pd.read_csv(r'popular_anime.csv')

# Clean the dataset by filling empty values
# Create a copy of the DataFrame to avoid modifying the original
df_clean = df.copy()
# Fill all empty values in all numeric columns with 0
for column in df_clean.select_dtypes(include=['float']).columns:
    df_clean[column] = df_clean[column].fillna(0)
# Fill all empty values in all string columns with 'Unknown'
for column in df_clean.select_dtypes(include=['object']).columns:
    df_clean[column] = df_clean[column].fillna('Unknown')

# Check that the aired_from and aired_to columns are in datetime format (only if not 'Unknown')
df_clean['aired_from'] = pd.to_datetime(df_clean['aired_from'], errors='coerce')
df_clean['aired_to'] = pd.to_datetime(df_clean['aired_to'], errors='coerce')

# Extract the year from the aired_from column and create a new column aired_from_year
df_clean['aired_from_year'] = df_clean['aired_from'].dt.year

# Create a column for the season of the aried_from date
df_clean['aired_from_season'] = df_clean['aired_from'].dt.month % 12 // 3 + 1
season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
df_clean['aired_from_season'] = df_clean['aired_from_season'].map(season_map)

# create a list of all the genres in the dataset
all_genres = set()
for genres in df_clean['genres'].str.split(', '):
    all_genres.update(genres)
all_genres = sorted(all_genres)
# ['Action', 'Adventure', 'Avant Garde', 'Award Winning', 'Boys Love', 'Comedy', 'Drama', 'Ecchi', 'Erotica', 'Fantasy', 'Girls Love', 
# 'Gourmet', 'Hentai', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Slice of Life', 'Sports', 'Supernatural', 'Suspense', 'Unknown']

# Filter the DataFrame
# Remove age rating 'Rx - Hentai'
df_filtered = df_clean[df_clean['rating'] != 'Rx - Hentai']

# Remove rows with 'Unknown' genres
df_filtered = df_filtered[df_filtered['genres'] != 'Unknown']
# Remove rows with 'Hentai' or 'Erotic' in genres
df_filtered = df_filtered[~df_filtered['genres'].str.contains('Hentai|Erotic', case=False, na=False)]

# Remove less relevant types ('Unknown', 'PV', 'CM', 'Music')
less_relevant_types = ['Unknown', 'PV', 'CM', 'Music']
df_filtered = df_filtered[~df_filtered['type'].isin(less_relevant_types)]

# Remove anime with a score of 0
df_filtered = df_filtered[df_filtered['score'] > 0]


##################################################
# Definitions of functions and tools for the chatbot
##################################################

# Start definitions of functions for the chatbot
def get_top_anime_by_column(data=df_filtered.to_dict(orient='records'), n=10, column='score', ascending=False):
    """
    Get the top n anime for a given column.
    :param data: dictionary containing anime data
    :param n: Number of top anime to return
    :param column: Column to sort by (default is 'score')
    :param ascending: Sort order (default is descending)
    :return: dictionary with the top n anime
    """
    data = pd.DataFrame(data)
    if column not in data.columns:
        raise ValueError(f"Column '{column}' does not exist in the data.")
    data = data.sort_values(by=column, ascending=ascending).head(n)
    top_anime = data.to_dict(orient='records')
    return top_anime

def get_anime_by_genre(genre, data=df_filtered.to_dict(orient='records')):
    """
    Get anime by genre.
    :param genre: Genre to filter by
    :return: dictionary with anime of the specified genre
    """
    data = pd.DataFrame(data)
    return data[data['genres'].str.contains(genre, case=False, na=False)]

def get_anime_by_year(year, data=df_filtered.to_dict(orient='records')):
    """
    Get anime by year.
    :param year: Year to filter by
    :return: dictionary with anime released in the specified year
    """
    data = pd.DataFrame(data)
    return data[data['aired_from_year'] == year]

def get_anime_by_season(season, year=None, data=df_filtered.to_dict(orient='records')):
    """
    Get anime by season and year.
    :param season: Season to filter by (e.g., 'Winter', 'Spring', 'Summer', 'Fall')
    :param year: Year to filter by
    :return: dictionary with anime released in the specified season and year
    """
    data = pd.DataFrame(data)
    if year is None:
        return data[data['aired_from_season'] == season]
    
    # Filter by both season and year
    return data[(data['aired_from_season'] == season) & (data['aired_from_year'] == year)]

def get_anime_by_type(anime_type, data=df_filtered.to_dict(orient='records')):
    """
    Get anime by type.
    :param anime_type: Type of anime to filter by (e.g., 'TV', 'Movie')
    :return: dictionary with anime of the specified type
    """
    data = pd.DataFrame(data)
    return data[data['type'] == anime_type]

def get_anime_by_score_range(min_score, max_score, data=df_filtered.to_dict(orient='records')):
    """
    Get anime by score range.
    :param min_score: Minimum score
    :param max_score: Maximum score
    :return: dictionary with anime within the specified score range
    """
    data = pd.DataFrame(data)
    return data[(data['score'] >= min_score) & (data['score'] <= max_score)]

def get_anime_by_rating(rating, data=df_filtered.to_dict(orient='records')):
    """
    Get anime by rating.
    :param rating: Rating to filter by (e.g., 'PG-13', 'R')
    :return: dictionary with anime of the specified rating
    """
    data = pd.DataFrame(data)
    return data[data['rating'] == rating]

def get_anime_by_name(name, data=df_filtered.to_dict(orient='records')):
    """
    Get anime by name.
    :param name: Name of the anime to filter by
    :return: dictionary with anime matching the specified name
    """
    data = pd.DataFrame(data)
    return data[data['name'].str.contains(name, case=False, na=False)]

def get_anime_by_synopsis(keyword, data=df_filtered.to_dict(orient='records')):
    """
    Get anime by synopsis keyword.
    :param keyword: Keyword to search in the synopsis
    :return: dictionary with anime containing the specified keyword in the synopsis
    """
    data = pd.DataFrame(data)
    return data[data['synopsis'].str.contains(keyword, case=False, na=False)]

def get_anime_by_producer(producer, data=df_filtered.to_dict(orient='records')):
    """
    Get anime by producer.
    :param producer: Producer to filter by
    :return: dictionary with anime produced by the specified producer
    """
    data = pd.DataFrame(data)
    return data[data['producers'].str.contains(producer, case=False, na=False)]

def get_anime_by_studio(studio, data=df_filtered.to_dict(orient='records')):
    """
    Get anime by studio.
    :param studio: Studio to filter by
    :return: dictionary with anime produced by the specified studio
    """
    data = pd.DataFrame(data)
    return data[data['studios'].str.contains(studio, case=False, na=False)]

def get_anime_info(anime_id=-1, anime_name=None, data=df_filtered.to_dict(orient='records')):
    """
    Get detailed information about a specific anime by its ID or name.
    If both ID and name are provided, ID takes precedence.
    :param anime_id: ID of the anime to retrieve information for
    :param anime_name: Name of the anime to retrieve information for
    :param data: dictionary containing anime data
    :return: Series with detailed information about the specified anime
    """
    data = pd.DataFrame(data)
    if anime_id != -1:
        # If anime_id is provided, filter by ID
        result = data[data['id'] == anime_id]
    elif anime_name:
        # If anime_name is provided, filter by name
        result = data[data['name'] == anime_name]
    else:
        raise ValueError("Either anime_id or anime_name must be provided.")
    if result.empty:
        return None
    else:
        return result.iloc[0].squeeze()  # Return the first row as a Series
    

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_top_anime_by_column",
            "description": "Get the top n anime for a given column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anime_id": {"type": "integer"},
                                "name": {"type": "string"},
                                "score": {"type": "number"},
                                "genres": {"type": "string"},
                                "aired_from_year": {"type": "integer"},
                                "aired_from_season": {"type": "string"},
                                "type": {"type": "string"},
                                "rating": {"type": "string"},
                                "synopsis": {"type": "string"},
                                "producers": {"type": "string"},
                                "studios": {"type": "string"},
                                # Add other relevant fields as needed
                            },
                            "required": ["anime_id", "name", "score", "genres", "aired_from_year", "aired_from_season", "type", "rating", "synopsis", "producers", "studios"],
                        },
                    },
                    "n": {"type": "integer", "description": "Number of top anime to return"},
                    "column": {"type": "string", "description": "Column to sort by (default is 'score')"},
                    "ascending": {"type": "boolean", "description": "Sort order (default is descending)"},
                },
                "required": ["n"],
            },
        },
    },
    # Add more tools as needed
    {
        "type": "function",
        "function": {
            "name": "get_anime_by_genre",
            "description": "Get anime by genre.",
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {"type": "string", "description": "Genre to filter by. List of genres in the dataset: 'Action', 'Adventure', 'Avant Garde', 'Award Winning', 'Boys Love', 'Comedy', 'Drama', 'Ecchi', 'Fantasy', 'Girls Love', 'Gourmet', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Slice of Life', 'Sports', 'Supernatural', 'Suspense'"},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anime_id": {"type": "integer"},
                                "name": {"type": "string"},
                                "score": {"type": "number"},
                                "genres": {"type": "string"},
                                "aired_from_year": {"type": "integer"},
                                "aired_from_season": {"type": "string"},
                                "type": {"type": "string"},
                                "rating": {"type": "string"},
                                "synopsis": {"type": "string"},
                                "producers": {"type": "string"},
                                "studios": {"type": "string"},
                                # Add other relevant fields as needed
                            },
                            "required": ["anime_id", "name", "score", "genres", "aired_from_year", "aired_from_season", "type", "rating", "synopsis", "producers", "studios"],
                        },
                    },
                },
                "required": ["genre"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anime_by_year",
            "description": "Get anime by year.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "Year to filter by"},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anime_id": {"type": "integer"},
                                "name": {"type": "string"},
                                "score": {"type": "number"},
                                "genres": {"type": "string"},
                                "aired_from_year": {"type": "integer"},
                                "aired_from_season": {"type": "string"},
                                "type": {"type": "string"},
                                "rating": {"type": "string"},
                                "synopsis": {"type": "string"},
                                "producers": {"type": "string"},
                                "studios": {"type": "string"},
                                # Add other relevant fields as needed
                            },
                            "required": ["anime_id", "name", "score", "genres", "aired_from_year", "aired_from_season", "type", "rating", "synopsis", "producers", "studios"],
                        },
                    },
                },
                "required": ["year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anime_by_season",
            "description": "Get anime by season and year.",
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {"type": "string", "description": "Season to filter by (e.g., 'Winter', 'Spring', 'Summer', 'Fall')"},
                    "year": {"type": ["integer", "null"], "description": "Year to filter by (optional)"},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anime_id": {"type": "integer"},
                                "name": {"type": "string"},
                                "score": {"type": "number"},
                                "genres": {"type": "string"},
                                "aired_from_year": {"type": "integer"},
                                "aired_from_season": {"type": "string"},
                                "type": {"type": "string"},
                                "rating": {"type": "string"},
                                "synopsis": {"type": "string"},
                                "producers": {"type": "string"},
                                "studios": {"type": "string"},
                                # Add other relevant fields as needed
                            },
                            "required": ["anime_id", "name", "score", "genres", "aired_from_year", "aired_from_season", "type", "rating", "synopsis", "producers", "studios"],
                        },
                    },
                },
                "required": ["season"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anime_by_type",
            "description": "Get anime by type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "anime_type": {"type": "string", "description": "Type of anime to filter by (e.g., 'TV', 'Movie', 'OVA', 'ONA', 'Special')"},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anime_id": {"type": "integer"},
                                "name": {"type": "string"},
                                "score": {"type": "number"},
                                "genres": {"type": "string"},
                                "aired_from_year": {"type": "integer"},
                                "aired_from_season": {"type": "string"},
                                "type": {"type": "string"},
                                "rating": {"type": "string"},
                                "synopsis": {"type": "string"},
                                "producers": {"type": "string"},
                                "studios": {"type": "string"},
                                # Add other relevant fields as needed
                            },
                            "required": ["anime_id", "name", "score", "genres", "aired_from_year", "aired_from_season", "type", "rating", "synopsis", "producers", "studios"],
                        },
                    },
                },
                "required": ["anime_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anime_by_score_range",
            "description": "Get anime by score range. Renge is from 0 to 10.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_score": {"type": "number", "description": "Minimum score"},
                    "max_score": {"type": "number", "description": "Maximum score"},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anime_id": {"type": "integer"},
                                "name": {"type": "string"},
                                "score": {"type": "number"},
                                "genres": {"type": "string"},
                                "aired_from_year": {"type": "integer"},
                                "aired_from_season": {"type": "string"},
                                "type": {"type": "string"},
                                "rating": {"type": "string"},
                                "synopsis": {"type": "string"},
                                "producers": {"type": "string"},
                                "studios": {"type": "string"},
                                # Add other relevant fields as needed
                            },
                            "required": ["anime_id", "name", "score", "genres", "aired_from_year", "aired_from_season", "type", "rating", "synopsis", "producers", "studios"],
                        },
                    },
                },
                "required": ["min_score", "max_score"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anime_by_rating",
            "description": "Get anime by age rating. Ratings include 'G - All Ages', 'PG - Children', 'PG-13 - Teens', 'R - 17+ (violence & profanity)', 'R+ - Mild Nudity'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rating": {"type": "string", "description": "Age Rating to filter by"},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anime_id": {"type": "integer"},
                                "name": {"type": "string"},
                                "score": {"type": "number"},
                                "genres": {"type": "string"},
                                "aired_from_year": {"type": "integer"},
                                "aired_from_season": {"type": "string"},
                                "type": {"type": "string"},
                                "rating": {"type": "string"},
                                "synopsis": {"type": "string"},
                                "producers": {"type": "string"},
                                "studios": {"type": "string"},
                                # Add other relevant fields as needed
                            },
                            "required": ["anime_id", "name", "score", "genres", "aired_from_year", "aired_from_season", "type", "rating", "synopsis", "producers", "studios"],
                        },
                    },
                },
                "required": ["rating"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anime_by_name",
            "description": "Get anime by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the anime to filter by"},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anime_id": {"type": "integer"},
                                "name": {"type": "string"},
                                "score": {"type": "number"},
                                "genres": {"type": "string"},
                                "aired_from_year": {"type": "integer"},
                                "aired_from_season": {"type": "string"},
                                "type": {"type": "string"},
                                "rating": {"type": "string"},
                                "synopsis": {"type": "string"},
                                "producers": {"type": "string"},
                                "studios": {"type": "string"},
                                # Add other relevant fields as needed
                            },
                            "required": ["anime_id", "name", "score", "genres", "aired_from_year", "aired_from_season", "type", "rating", "synopsis", "producers", "studios"],
                        },
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anime_by_synopsis",
            "description": "Get anime by synopsis keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Keyword to search in the synopsis"},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anime_id": {"type": "integer"},
                                "name": {"type": "string"},
                                "score": {"type": "number"},
                                "genres": {"type": "string"},
                                "aired_from_year": {"type": "integer"},
                                "aired_from_season": {"type": "string"},
                                "type": {"type": "string"},
                                "rating": {"type": "string"},
                                "synopsis": {"type": "string"},
                                "producers": {"type": "string"},
                                "studios": {"type": "string"},
                                # Add other relevant fields as needed
                            },
                            "required": ["anime_id", "name", "score", "genres", "aired_from_year", "aired_from_season", "type", "rating", "synopsis", "producers", "studios"],
                        },
                    },
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anime_by_producer",
            "description": "Get anime by producer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "producer": {"type": "string", "description": "Producer to filter by"},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anime_id": {"type": "integer"},
                                "name": {"type": "string"},
                                "score": {"type": "number"},
                                "genres": {"type": "string"},
                                "aired_from_year": {"type": "integer"},
                                "aired_from_season": {"type": "string"},
                                "type": {"type": "string"},
                                "rating": {"type": "string"},
                                "synopsis": {"type": "string"},
                                "producers": {"type": "string"},
                                "studios": {"type": "string"},
                                # Add other relevant fields as needed
                            },
                            "required": ["anime_id", "name", "score", "genres", "aired_from_year", "aired_from_season", "type", "rating", "synopsis", "producers", "studios"],
                        },
                    },
                },
                "required": ["producer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anime_by_studio",
            "description": "Get anime by studio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "studio": {"type": "string", "description": "Studio to filter by"},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anime_id": {"type": "integer"},
                                "name": {"type": "string"},
                                "score": {"type": "number"},
                                "genres": {"type": "string"},
                                "aired_from_year": {"type": "integer"},
                                "aired_from_season": {"type": "string"},
                                "type": {"type": "string"},
                                "rating": {"type": "string"},
                                "synopsis": {"type": "string"},
                                "producers": {"type": "string"},
                                "studios": {"type": "string"},
                                # Add other relevant fields as needed
                            },
                            "required": ["anime_id", "name", "score", "genres", "aired_from_year", "aired_from_season", "type", "rating", "synopsis", "producers", "studios"],
                        },
                    },
                },
                "required": ["studio"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anime_info",
            "description": "Get detailed information about a specific anime by its ID or name. If both ID and name are provided, ID takes precedence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "anime_id": {"type": "integer", "description": "ID of the anime to retrieve information for (optional)"},
                    "anime_name": {"type": "string", "description": "Name of the anime to retrieve information for (optional)"},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anime_id": {"type": "integer"},
                                "name": {"type": "string"},
                                "score": {"type": "number"},
                                "genres": {"type": "string"},
                                "aired_from_year": {"type": "integer"},
                                "aired_from_season": {"type": "string"},
                                "type": {"type": "string"},
                                "rating": {"type": "string"},
                                "synopsis": {"type": "string"},
                                "producers": {"type": "string"},
                                "studios": {"type": "string"},
                                # Add other relevant fields as needed
                            },
                            "required": ["anime_id", "name", "score", "genres", "aired_from_year", "aired_from_season", "type", "rating", "synopsis", "producers", "studios"],
                        },
                    },
                },
                "required": [],
            },
        },
    },    
]

# Function to handle the chatbot's tool calls
def function_calling(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        print(f"Calling tool: {tool_name} with arguments: {tool_call.function.arguments}") # Debugging line to see the tool call details
        params = json.loads(tool_call.function.arguments)
        if 'data' not in params:
            params['data'] = df_filtered.to_dict(orient='records')  # Convert dictionary to list of dicts
        if tool_name == "get_top_anime_by_column":
            result = get_top_anime_by_column(**params)
        elif tool_name == "get_anime_by_genre":
            if 'genre' not in params:
                raise ValueError("Missing 'genre' parameter for get_anime_by_genre")
            result = get_anime_by_genre(**params)
        elif tool_name == "get_anime_by_year":
            if 'year' not in params:
                raise ValueError("Missing 'year' parameter for get_anime_by_year")
            result = get_anime_by_year(**params)
        elif tool_name == "get_anime_by_season":
            if 'season' not in params:
                raise ValueError("Missing 'season' parameter for get_anime_by_season")
            result = get_anime_by_season(**params)
        elif tool_name == "get_anime_by_type":
            if 'anime_type' not in params:
                raise ValueError("Missing 'anime_type' parameter for get_anime_by_type")
            result = get_anime_by_type(**params)
        elif tool_name == "get_anime_by_score_range":
            if 'min_score' not in params or 'max_score' not in params:
                raise ValueError("Missing 'min_score' or 'max_score' parameter for get_anime_by_score_range")
            result = get_anime_by_score_range(**params)
        elif tool_name == "get_anime_by_rating":
            if 'rating' not in params:
                raise ValueError("Missing 'rating' parameter for get_anime_by_rating")
            result = get_anime_by_rating(**params)
        elif tool_name == "get_anime_by_name":
            if 'name' not in params:
                raise ValueError("Missing 'name' parameter for get_anime_by_name")
            result = get_anime_by_name(**params)
        elif tool_name == "get_anime_by_synopsis":
            if 'keyword' not in params:
                raise ValueError("Missing 'keyword' parameter for get_anime_by_synopsis")
            result = get_anime_by_synopsis(**params)
        elif tool_name == "get_anime_by_producer":
            if 'producer' not in params:
                raise ValueError("Missing 'producer' parameter for get_anime_by_producer")
            result = get_anime_by_producer(**params)
        elif tool_name == "get_anime_by_studio":
            if 'studio' not in params:
                raise ValueError("Missing 'studio' parameter for get_anime_by_studio")
            result = get_anime_by_studio(**params)
        elif tool_name == "get_anime_info":
            if 'anime_id' not in params and 'anime_name' not in params:
                raise ValueError("Either 'anime_id' or 'anime_name' must be provided for get_anime_info")
            if 'anime_id' not in params:
                params['anime_id'] = -1
            if 'anime_name' not in params:
                params['anime_name'] = None
            result = get_anime_info(**params)
        else:
            result = None
        results.append(str(result) if result is not None else f"No results for tool {tool_name} with parameters {params}")
    return "\n\n".join(results)  # Restituisce i risultati concatenati



##################################################
# Main loop to interact with the chatbot
##################################################

if __name__ == "__main__":
    
    # Initialize the chatbot
    messages = [
        {
            "role": "system",
            "content": "Sei un assistente esperto in anime. Puoi rispondere a domande su anime, generi, stagioni, anni di uscita, produttori, studi di animazione e altro ancora. Utilizza le funzioni disponibili (anche più di una alla volta) per fornire risposte accurate o per dare consigli sugli anime migliori per l'utente. Fai attenzione a flitrare le risposte in base al genere specifico richiesto.",
        },
        {
            "role": "user",
            "content": "Ciao! Chi sei?",
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages
    )

    print("Avvio del chatbot...\n")
    print(response.choices[0].message.content)
    messages.append(response.choices[0].message)
    
    # Main loop 
    while True:
        print("\n"+"------"*8+"\n")
        Query = input("Inserisci la tua richiesta (o 'exit' per uscire):\n")
        if Query.lower() == "exit":
            print("Uscita dal programma.")
            break

        messages.append(
            {
                "role": "user",
                "content": Query,
            }
        )

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages[:20],  # Limit to the last 20 messages
            tools=tools,
            tool_choice="auto",
        )
        
        print("\nRisposta del modello:")
        if response.choices[0].message.tool_calls:
            tool_result = function_calling(response.choices[0].message.tool_calls)
            # Add the result as an assistant message
            messages.append({"role": "assistant", "content": f"{tool_result}\nScrivi una risposta in base ai risultati ottenuti."})
            # Ask the model to generate a reply based on that result
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages[:20],
                tools=tools,
                tool_choice="auto",
            )
        print(response.choices[0].message.content)
        if not response.choices[0].message.content:
            print("No response generated by the model. Printing the last lines of tool_calls results:\n")
            print(tool_result[-50:])  # Print the last 50 lines of the tool calls result
        else:
            # Add the model's reply to the conversation
            messages.append(response.choices[0].message)
