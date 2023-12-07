# Headline HUB | News Aggregator 

## Overview

This is a News Aggregator, a web application designed to provide users with personalized news suggestions based on various metrics such as text length, sentiment score, bias score, sensationalism score, readability, and clickbait check. Furthermore, the web application offers a variety of analysis tools to form an in-depth opinion about different news sites and topics.

## Features

- **Personalized Suggestions:** Receive news suggestions tailored to your preferences.
- **Metric-Based Filtering:** Explore news articles based on text length, sentiment, bias, sensationalism, readability, and clickbait check.
- **Multiple Profiles:** Choose from different user profiles, each emphasizing specific metrics and preferences.
- **Analysis Tools:** Use different Analysis Tools like a Tableau Dashboard, Word Freqeuncy Comparer and the "Explorer", which gives you all kinds of statistics for an article after being given an article url

## Getting Started

### Prerequisites

- Python (>=3.6)
- Required Python packages (install with `pip install -r requirements.txt`)

### Installation
- fork and clone this repo
- web app: -streamlit run app.by in the command line after going to the streamlit directory to start the web app
- login: the first user login can made with username: user1 , password: abc
- accounts: more users can be added in the config.yaml file, remember to add the passwords as hashed passwords. This can be done with the generate_keys.py file
- updating: run the update_articles.ipynb and table_tuning.ipynb (sequentially) to get the newest articles directly into your web app (this can take up to 15 minutes)
