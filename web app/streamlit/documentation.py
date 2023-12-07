import streamlit as st
import pandas as pd
from PIL import Image
def show_documentation():
    st.title("Documentation")
    st.write("---")
    col1, col2 = st.columns([3,1])
    col1.write("**At HeadlineHUB different Metrics are used in order to give users better insight about each articles.**")
    col1.write("**These Metrics are Text length, Sentiment, Readability, Bias Score, Sensationalism Score, Clickbait Probability and Fake News Check**")
    col1.write("**Text length:** The Text length is given by the number of Characters in the Article Text. This Value usually lies between has a very big deviation with a mean of ca 4200 chars.")
    col1.write("**Sentiment:** The Sentiment Score (-1 to 1) is calculated with NLTK SentimentIntensityAnalyzer. This Score is then grouped into Positive(0.35<x<1), Neutral(-0.35<x<0.35) and Negative(-1<x<-0.35)")
    col1.write("**Readability:** The Readability is calculated with the Coleman Liau Index. It approximates the US Grade level thought necessary to comprehend the text.")
    col1.write("**Bias Score:** This metric is calculated with the TextBlob sentiment.subjectivity Tool. The value is then squared to disregard positive or negative sentiment and only focus on general bias.")
    col1.write("**Sensationalism Score:** This Score looks for a words for such as superlatives and other ones that are associated with sensationalism. The number of occurences of those words is then normalized with the Text lenth.")
    col1.write("**Clickbait Probability:** To predict wheter an Article is likely to be clickbait a LogisticRegression model with 97% accuracy is applied to title to look for features often associated with clickbait (e.g uppercased, exclamation marks etc.)")
    image = Image.open("./pictures/newspaper.png")
    col2.image(image, use_column_width=True)
    st.write("---")
    st.title("How our Suggestions are generated")
    col3, col4 = st.columns([3,1])
    col3.write("Headline HUB uses the above explainend metrics to calculate a total score for each article, then the articles with the highest score are shown first on our suggestion tab.")
    col3.write("For the final score we rewards different Attributes for each feature. The features have different weights in the calculation for the total score.")
    col3.write("These can be seen here")
    weights = pd.DataFrame({"Feature" :["Date", "Sentiment", "Sensationalism", "Readability", "Bias", "Clickbait", "Categories"],"Searching For": ["Newest", "Most Neutral", "low use of sensationalism language", "Readability around 11th grade", "Most unbiased", "unlikeliest to have clickbait", "most new articles published"],"Weight":[0.25, 0.1, 0.1, 0.15, 0.1, 0.1, 0.2]})
    st.table(weights)
    st.write("---")
    st.title("How the Suggestions for each Profile are generated")
    col5, col6 = st.columns([3,1])
    col5.write("Every Profile searches for different Attributes in the features to make accurate Suggestions.")
    col5.write("These Attribute can be found here")
    my_dict = {
        'Profile': ['In-Depth Explorer', 'Quick Briefing', 'Positive Vibes', 'Balanced Perspective', 'In-Depth Analysis', 'Critical Thinker'],
        'Text Length': ['Long', 'Short', 'Moderate', 'Moderate', 'Long', 'Moderate to Long'],
        'Sentiment Score': ['Neutral to Positive', 'Neutral', 'Positive', 'Neutral', 'Neutral to Negative', 'no preference'],
        'Readability': ['Moderate to Challenging', 'High', 'Moderate to High', 'Moderate', 'Moderate to Challenging', 'Moderate'],
        'Sensationalism Score': ['Low', 'Low', 'Moderate', 'Low to Moderate', 'Low to Moderate', 'Moderate'],
        'Bias Score': ['Balanced', 'Balanced', 'Balanced', 'Low', 'Moderate', 'Moderate to High'],
        'Clickbait Check': ['Low', 'Low', 'no preference', 'Low', 'Low', 'Low']
    }
    df = pd.DataFrame(my_dict)
    df.set_index('Profile', inplace=True)
    st.table(df)