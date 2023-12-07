
# --- Load Libraries ---
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pickle
import streamlit_authenticator as stauth
import pandas as pd
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
from datetime import date, timedelta
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
lemmatizer = WordNetLemmatizer()
import re
from documentation import show_documentation
import ast

# --- PAGE CONFIG ---
st.set_page_config(page_title="Headline HUB",
                    page_icon=":newspaper:")

# --- Load Models ---
with open('bow_vect.pkl', 'rb') as file:
    bow_vect = pickle.load(file)
with open('cat_model.pkl', 'rb') as file:
    cat_model = pickle.load(file)

with open('bow_vect_clickbait.pkl', 'rb') as file:
    bow_vect_clickbait = pickle.load(file)
with open('clickbait_model.pkl', 'rb') as file:
    clickbait_model = pickle.load(file)


# --- Load Data ---
# all articles
df = pd.read_csv("C:/Users/Jacob/Desktop/Final Project/datasets/all_articles.csv")
# suggestions
ranked_df = pd.read_csv("C:/Users/Jacob/Desktop/Final Project/datasets/ranked_df.csv")
# profiles
in_depth_explorer = pd.read_csv("./data/In-DepthExplorer.csv")
quick_briefing = pd.read_csv("./data/QuickBriefing.csv")
positive_vibes = pd.read_csv("./data/PositiveVibes.csv")
balanced_perspective = pd.read_csv("./data/BalancedPerspektive.csv")
in_depth_analysis = pd.read_csv("./data/InDepthAnalysis.csv")
critical_thinker = pd.read_csv("./data/CriticalThinker.csv")

#categories
politics = pd.read_csv("./data/politics_df.csv")
entertainment = pd.read_csv("./data/entertainment_df.csv")
international = pd.read_csv("./data/international_df.csv")
sports = pd.read_csv("./data/sports_df.csv")
climate = pd.read_csv("./data/climate_df.csv")
health_tech = pd.read_csv("./data/health_tech_df.csv")
justice = pd.read_csv("./data/justice_df.csv")
buisness = pd.read_csv("./data/buisness_df.csv")
cookies = pd.read_csv("./data/user_cookies.csv")

# --- Functions ---
def display_articles(df, c):
    for index, row in df.iterrows():
        col1, col2 = st.columns([8,2])
        if c==0:
            col1.write(f"**Date:** {row['Date']} | **Source:** {row['Source']} | **Author:** {row['Author']}")
        if c==1:
            col1.write(f"**Date:** {row['Date']} | **Source:** {row['Source']} | **Author:** {row['Author']} | **Category:** {row['cat_label']}")
        # define clickbait
        if row["isClickbait"]==0:
            click = "Unlikely"
        else:
            click = "Probable"
        # define sentiment
        if row["sentiment_score"]>0.45:
            sentiment = "Positive"
        elif row["sentiment_score"]<-0.45:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        # define bias
        if row["bias_score"]<0.34:
            bias ="low"
        elif row["bias_score"]>0.42:
            bias = "high"
        else:
            bias = "moderate"
        col1.write(f"**Text Length:** {row['Text lenght']} | **Sentiment:** {sentiment} | **Readability:** {round(row['Readability'],2)} | **Bias:** {bias} | **Sensationalism:** {round(row['sensationalism_score'],2)} | **Clickbait:** {click}")
        title_link = f'<a href="{row["URL"]}" target="_blank" style="font-size: 45px; color: black;">{row["Title"]}</a>'
        col1.markdown(title_link, unsafe_allow_html=True)

        col2.image(row['imageURL'], use_column_width=True)
        st.write("---")

def dates_list(s_dates):
    date_list = []
    current_date = s_dates[0]
    end = s_dates[1]
    while current_date <= end:
        date_list.append(current_date)
        current_date += timedelta(days=1)
    return date_list

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word], lang='eng')[0][1][0].upper() # gets first letter of POS categorization
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def word_tool(df, word):
    """Receives an Word as input and returns Chart with Frequency of the Word used per News Site"""
    sources = ['CNN', 'Fox News', 'ABC', 'Politico', 'Washington Post', 'New York Times', 'MSNBC', 'Reuters', 'NPR', 'Chicago Tribune', 'USA TODAY']
    texts = {}
    for source in sources:
        texts[source] = df[df["Source"]==source]["Text"].str.lower()
    
    word = word.lower()
    word_count = {}
    index = 0
    for source, text in texts.items():
        total_count = text.str.count(word).sum()
        word_count[source] = total_count/len(text)
        index += 1

    word_count = pd.DataFrame.from_dict(word_count, orient="index")
    word_count = word_count.reset_index()
    word_count.columns = ["Source", "Count"]
    st.title(word)
    return st.bar_chart(word_count.set_index('Source')['Count'])


def clean_html_tags(html_text):
    # Use BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(html_text, 'html.parser')
    text_content = soup.get_text()
    cleaned_text = re.sub(r'\s+', ' ', text_content).strip()
    return cleaned_text

def web_scrape(url):
    r = requests.get(url)
    if r.status_code != 200:
        driver = webdriver.Edge()
        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html)
        title = soup.find("h1").get_text().strip()
        title = clean_html_tags(title)

        text = []
        paragraphs = soup.find_all("p")
        for p in paragraphs:
            y = p.get_text().strip()
            text.append(y)
        text = "".join(text)
        text = clean_html_tags(text)
    else:
        soup = BeautifulSoup(r.content)
        title = soup.find("h1").get_text().strip()
        title = clean_html_tags(title)

        text = []
        paragraphs = soup.find_all("p")
        for p in paragraphs:
            y = p.get_text().strip()
            text.append(y)
        text = "".join(text)
        text = clean_html_tags(text)
    return title, text

def sum_stats_2(title, text):
    text_lower = text.lower()
    # text length

    length = len(text)
    # sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment_score =  sia.polarity_scores(text)["compound"]
    if sentiment_score >0.45:
        sentiment = "Positive"
    elif sentiment_score<-0.45:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    # sensationalism
    sensationalism_score = 0 # base value
    sensationalism_keywords = ['breaking', 'exclusive', 'shocking', 'explosive', 'revealed', 'urgent', 'unbelievable', 'mind-blowing', 'scandalous', 'outrageous', 'sensational', 'never-before-seen', 'dramatic', 'jaw-dropping', 'killer', 'massive', 'insane', 'terrifying', 'banned', 'controversial', 'secret', 'conspiracy', 'nightmare', 'apocalyptic', 'incredible', 'forbidden', 'sinister', 'catastrophic', 'shock', 'danger', 'fear', 'panic', 'monster', 'death-defying', 'omg', 'alarming', 'tremendous', 'never', 'deadly', 'hellish', 'paranormal', 'bewildering', 'menacing', 'twisted', 'kill', 'wild', 'devious', 'exposed', 'unprecedented', 'crisis', 'apocalypse', 'bizarre', 'explosive revelation', 'apocalyptic', 'deadly virus', 'unearthed', 'censored', 'life-threatening', 'cataclysmic', 'underground', 'illegal', 'menace', 'gruesome', 'intense', 'unholy', 'insidious', 'doomsday', 'untold', 'mysterious', 'censored', 'horrifying', 'unexplained', 'phenomenal', 'surreal', 'freakish', 'clash', 'supernatural', 'reckless', 'banned', 'taboo', 'untamed', 'monstrous', 'forbidden', 'pandemonium', 'infernal', 'no holds barred', 'ruthless', 'ghostly', 'frightening', 'dangerous liaison', 'freak accident', 'aberration', 'anarchy', 'untouchable', 'eerie', 'conspiracy theory', 'lost civilization', 'apocalyptic nightmare', 'inexplicable']
    for keyword in sensationalism_keywords:
        keyword_count = text_lower.count(keyword.lower()) 
        sensationalism_score += 0.3 * keyword_count  # Add to score based on keyword occurrences
    divisor = len(text) / 1000  
    sensationalism_score /= divisor

    # readability
    words = word_tokenize(text)
    words_count = len(words)
    sentences = sent_tokenize(text)
    sentences = len(sentences)
    letters = sum(len(letter) for word in words for letter in word)
    L = (letters / words_count) * 100
    S = (sentences / words_count) * 100
    readability = 0.0588 * L - 0.296 * S - 15.8

    # bias score
    analysis = TextBlob(text)
    bias = analysis.sentiment.subjectivity
    bias = bias**2
    bias = np.sqrt(bias)
    # define bias
    if bias<0.34:
        bias ="low"
    elif bias>0.42:
        bias = "high"
    else:
        bias = "moderate"
    # category
    lemmatized = [lemmatizer.lemmatize(word,get_wordnet_pos(word)) for word in words]
    remove_sw = list(set(lemmatized).difference(stopwords.words()))
    clean_blob = " ".join(remove_sw)
    my_dict = {0:{"clean_blob":clean_blob}}
    empty = pd.DataFrame.from_dict(my_dict, orient="index")
    X = bow_vect.transform(empty["clean_blob"]).toarray()
    category = cat_model.predict(X)

    # clickbait
    title_lenght = len(title)   # characters of title
    uppercased = (len([l for l in title if l.isupper()])) # num of uppercased
    my_dict3 = {0:{"title_lenght":title_lenght, "uppercased":uppercased}}
    features = pd.DataFrame.from_dict(my_dict3, orient="index")
    my_dict2 = {0:{"title":title}}    
    empty2 = pd.DataFrame.from_dict(my_dict2, orient="index")
    vs = bow_vect_clickbait.transform(empty2["title"]).toarray()
    vs = pd.DataFrame(vs, columns=bow_vect_clickbait.get_feature_names_out())
    X = pd.concat([features, vs], axis=1)
    X.columns = X.columns.astype(str)
    clickbait = clickbait_model.predict(X)
    if clickbait==0:
        click = "No"
    else:
        click = "Yes"
    st.write(f"**Category:** {category[0]} | **Text length:** {length} | **Sentiment:** {sentiment} | **Sensationalism:** {round(sensationalism_score,2)} | **Readability:** {round(readability,2)} | **Bias:** {round(bias,3)} | **Clickbait:** {click}")


# --- USER AUTHENTICATION ---
import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)
name, authentication_status, username = authenticator.login('Login', 'main')


# --- MAIN PAGE ---
if authentication_status==True:
    local_image_path = "./pictures/logo3.png"
    image = Image.open(local_image_path)
    st.image(image, use_column_width=True)
    st.sidebar.header("Menu")
    st.sidebar.write("---")
    selected_page = st.sidebar.radio("Page", ["Home", "Documentation","User Profile", "Articles"],0)
    for i in range(18):
        st.sidebar.write(" ")
    authenticator.logout('Logout', 'sidebar', key='unique_key')

    if selected_page == "Home":
        image = Image.open("./pictures/flyer.png")
        st.image(image, use_column_width=True)
    if selected_page == "Documentation":
        show_documentation()
    
    if selected_page =="User Profile":
        st.title("Profile")
        st.write("---")
        st.title(f"Welcome Back {name}!")
        rating = st.slider(
                label="How Happy are you with  HeadlineHUB at the moment?",
                min_value=1,
                max_value=5,
                value=(3), key="sdsaem"
                )
        st.text_input("Please leave your Feedback here: ", max_chars=250, key="msdwyalokol")
        st.write("---")
        st.title("Your News Profile")
        sources = ["CNN", "Fox News", "ABC", "Politico", "Washington Post", "New York Times", "MSNBC", "Reuters", "USA TODAY", "NPR", "Chicago Tribune"]
        categories = ['Politics', 'Entertainment and Lifestyle', 'International News','Business and Economy', 'Sports', 'Climate and Environment','Health | Science | Technology', 'Law and Justice']
        profiles = ["In-Depth Explorer", "Quick Briefing", "Positive Vibes", "Balanced Perspective", "In-Depth Analysis", "Critical Thinker"]
        col1, col2 = st.columns([2,1]) 
        col1.write("Choose how you want to consume your daily dose of news here by selecting the profile that suits you best")
        col1.write("**In-Depth Explorer:** This profile caters to users who prefer detailed and comprehensive news articles with a balanced and neutral tone.")
        col1.write("**Quick Briefing:** For users who prefer concise, to-the-point news that provides a quick overview of current events without unnecessary details.")
        col1.write("**Positive Vibes:** Tailored for users who want news articles with a positive and uplifting tone.")
        col1.write("**Balanced Perspective:** Aimed at users who value balanced perspectives and want to avoid highly biased or sensationalized content.")
        col1.write("**In-Depth Analysis:** For users who enjoy in-depth analysis and are open to a moderate level of bias.")
        col1.write("**Critical Thinker:** Catering to users who appreciate thought-provoking articles with a moderate level of bias. ")
        st.write(" ")
        st.write("You can find a more detailed Description of the calculations used for each profile on the Documentation side.")
        image = Image.open("./pictures/profiles.png")
        col2.image(image, width=200)
        options_from = st.form("options_form")
        cats_cookies = str(cookies["categories"][0])
        cats_cookies = ast.literal_eval(cats_cookies)
        source_cookies = str(cookies["sources"][0])
        source_cookies = ast.literal_eval(source_cookies)
        profile_index = profiles.index(cookies["profile"][0])
        categories_user = options_from.multiselect("Select the Categories that interest you", categories,default=cats_cookies)
        sources_user = options_from.multiselect ("Select the News Site you like",sources, default=source_cookies, key="mczwexaüö")
        selected_profile = options_from.radio("Select your News Profile here: ",profiles , profile_index)
        submit = options_from.form_submit_button("submit")
        if submit:
            cookies["categories"] = [categories_user]
            cookies["sources"] = [sources_user]
            cookies["profile"] = selected_profile
            cookies.to_csv("./data/user_cookies.csv", index=False)

    if selected_page =="Articles":
        selected = option_menu(
        menu_title=None,
        options=["Suggestions", "Your Picks", "Categories", "Analyzer", "Search"],
        icons = ["arrow-up-circle", "lightning","list-columns-reverse","pie-chart", "cursor"],
        menu_icon = "cast",
        default_index=0,
        orientation="horizontal",
        )

        if selected == "Suggestions":
            st.write("---")
            st.title("Look here for our special Suggestions")
            n = 60
            st.write("---")
            display_articles(ranked_df[:n], c=1)
            if st.button("Show more articles"):
                st.write("---")
                on = n
                n = n*2
                display_articles(ranked_df[n:on*3], c=1)


        if selected == "Your Picks":
            st.write("---")
            st.title("Best News Articles fitted to your Preferences")
            st.write("  - You can change your preferences on the Profile page.")
            st.write("  - More Information regarding the used metrics can be found in the Documentation tab")
            st.write("---")
            cats_cookies = str(cookies["categories"][0])
            cats_cookies = ast.literal_eval(cats_cookies)
            source_cookies = str(cookies["sources"][0])
            source_cookies = ast.literal_eval(source_cookies)
            if cookies["profile"][0] == "In-Depth Explorer":
                ide_df_1 = in_depth_explorer[in_depth_explorer["cat_label"].isin(cats_cookies)]
                ide_df = ide_df_1[ide_df_1["Source"].isin(source_cookies)]
                display_articles(ide_df[:100],c=1)
            if cookies["profile"][0] == "Quick Briefing": 
                qb_df_1 = quick_briefing[quick_briefing["cat_label"].isin(cats_cookies)]
                qb_df = qb_df_1[qb_df_1["Source"].isin(source_cookies)]
                display_articles(qb_df[:100],c=1)
            if cookies["profile"][0] == "Positive Vibes": 
                pv_df_1 = positive_vibes[positive_vibes["cat_label"].isin(cats_cookies)]
                pv_df = pv_df_1[pv_df_1["Source"].isin(source_cookies)]
                display_articles(pv_df[:100],c=1)
            if cookies["profile"][0] == "Balanced Perspective": 
                bp_df_1 = balanced_perspective[balanced_perspective["cat_label"].isin(cats_cookies)]
                bp_df = bp_df_1[bp_df_1["Source"].isin(source_cookies)]
                display_articles(bp_df[:100],c=1)
            if cookies["profile"][0] == "In-Depth Analysis": 
                ida_df_1 = in_depth_analysis[in_depth_analysis["cat_label"].isin(cats_cookies)]
                ida_df = ida_df_1[ida_df_1["Source"].isin(source_cookies)]
                display_articles(ida_df[:100],c=1)
            if cookies["profile"][0] == "Critical Thinker": 
                ct_df_1 = critical_thinker[critical_thinker["cat_label"].isin(cats_cookies)]
                ct_df = ct_df_1[ct_df_1["Source"].isin(source_cookies)]
                display_articles(ct_df[:100],c=1)


        if selected == "Categories":
            st.write("---")
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Politics", "International News", "Buisness and Economy", "Entertainment and Lifestyle", "Sports", "Enviroment and Climate", "Health | Science | Technology", "Law and Justice"])
            sources = ["CNN", "Fox News", "ABC", "Politico", "Washington Post", "New York Times", "MSNBC", "Reuters", "USA TODAY", "NPR", "Chicago Tribune"]
            with tab1:
                st.write("**Filter here:**")
                start_date = date(2023, 11, 27)
                end_date = date.today()
                end_date = max(start_date + timedelta(days=1), end_date)
                d_pol = st.slider(
                    label="Select a date",
                    min_value=start_date,
                    max_value=end_date,
                    value=(start_date, end_date),
                    key="asndjndjn"
                )
                
                dates_pol = dates_list(d_pol)
                politics["Day"] = pd.to_datetime(politics["Day"])
                politics_filtered_1 = politics[politics["Day"].isin(dates_pol)]
                cola, colb = st.columns([1,2])
                sources_p = colb.multiselect ("Select the News Site you want to be shown",sources, default=sources)
                politics_filtered = politics_filtered_1[politics_filtered_1["Source"].isin(sources_p)]
                n_articles_p = cola.slider("Select the amount of articles to be shown",0,len(politics_filtered),int(len(politics_filtered)/2))
                st.write("---")
                display_articles(politics_filtered[:n_articles_p], c=0)

            with tab2:
                st.write("**Filter here:**")
                start_date = date(2023, 11, 27)
                end_date = date.today()
                end_date = max(start_date + timedelta(days=1), end_date)
                d_inter = st.slider(
                    label="Select a date",
                    min_value=start_date,
                    max_value=end_date,
                    value=(start_date, end_date),
                    key="asndjnxaedjn"
                )
                dates_inter = dates_list(d_inter)
                international["Day"] = pd.to_datetime(international["Day"])
                international_filtered_1 = international[international["Day"].isin(dates_inter)]
                cola, colb = st.columns([1,2])
                sources_i = colb.multiselect ("Select the News Site you want to be shown",sources, default=sources, key="andjasndsd")
                international_filtered = international_filtered_1[international_filtered_1["Source"].isin(sources_i)]
                n_articles_i = cola.slider("Select the amount of articles to be shown",0,len(international_filtered),int(len(international_filtered)/2), key="asndjnasj")
                st.write("---")
                display_articles(international_filtered[:n_articles_i], c=0)

            with tab3:
                st.write("**Filter here:**")
                start_date = date(2023, 11, 27)
                end_date = date.today()
                end_date = max(start_date + timedelta(days=1), end_date)
                d_buis = st.slider(
                    label="Select a date",
                    min_value=start_date,
                    max_value=end_date,
                    value=(start_date, end_date),
                    key="asndjnjnaedjn"
                )
                dates_buis = dates_list(d_buis)
                buisness["Day"] = pd.to_datetime(buisness["Day"])
                buisness_filtered_1 = buisness[buisness["Day"].isin(dates_buis)]
                cola, colb = st.columns([1,2])
                sources_b = colb.multiselect ("Select the News Site you want to be shown",sources, default=sources, key="asmfkasmkf")
                buisness_filtered = buisness_filtered_1[buisness_filtered_1["Source"].isin(sources_b)]
                n_articles_b = cola.slider("Select the amount of articles to be shown",0,len(buisness_filtered),int(len(buisness_filtered)/2), key="sssfajnfj")
                st.write("---")
                display_articles(buisness_filtered[:n_articles_b], c=0)

            with tab4:
                st.write("**Filter here:**")
                start_date = date(2023, 11, 27)
                end_date = date.today()
                end_date = max(start_date + timedelta(days=1), end_date)
                d_enter = st.slider(
                    label="Select a date",
                    min_value=start_date,
                    max_value=end_date,
                    value=(start_date, end_date),
                    key="asndjnnknh"
                )
                dates_enter = dates_list(d_enter)
                entertainment["Day"] = pd.to_datetime(entertainment["Day"])
                entertainment_filtered_1 = entertainment[entertainment["Day"].isin(dates_buis)]
                cola, colb = st.columns([1,2])
                sources_e = colb.multiselect ("Select the News Site you want to be shown",sources, default=sources, key="asmfaxnnhnasmkf")
                entertainment_filtered = entertainment_filtered_1[entertainment_filtered_1["Source"].isin(sources_e)]
                n_articles_e = cola.slider("Select the amount of articles to be shown",0,len(entertainment_filtered),int(len(entertainment_filtered)/2), key="sssfaploxfj")
                st.write("---")
                display_articles(entertainment_filtered[:n_articles_e], c=0)

            with tab5:
                st.write("**Filter here:**")
                start_date = date(2023, 11, 27)
                end_date = date.today()
                end_date = max(start_date + timedelta(days=1), end_date)
                d_sport = st.slider(
                    label="Select a date",
                    min_value=start_date,
                    max_value=end_date,
                    value=(start_date, end_date),
                    key="asnopenynknh"
                )
                dates_sport = dates_list(d_sport)
                sports["Day"] = pd.to_datetime(sports["Day"])
                sports_filtered_1 = sports[sports["Day"].isin(dates_sport)]
                cola, colb = st.columns([1,2])
                sources_s = colb.multiselect ("Select the News Site you want to be shown",sources, default=sources, key="asmpolizedkasmkf")
                sports_filtered = sports[sports["Source"].isin(sources_s)]
                n_articles_s = cola.slider("Select the amount of articles to be shown",0,len(sports_filtered),int(len(sports_filtered)/2), key="spramnanfj")
                st.write("---")
                display_articles(sports_filtered[:n_articles_s], c=0)

            with tab6:
                st.write("**Filter here:**")
                start_date = date(2023, 11, 27)
                end_date = date.today()
                end_date = max(start_date + timedelta(days=1), end_date)
                d_clim = st.slider(
                    label="Select a date",
                    min_value=start_date,
                    max_value=end_date,
                    value=(start_date, end_date),
                    key="asnomjhbnknh"
                )
                dates_clim = dates_list(d_clim)
                climate["Day"] = pd.to_datetime(climate["Day"])
                climate_filtered_1 = climate[climate["Day"].isin(dates_clim)]
                cola, colb = st.columns([1,2])
                sources_c = colb.multiselect ("Select the News Site you want to be shown",sources, default=sources, key="asmfannnervthnasmkf")
                climate_filtered = climate[climate["Source"].isin(sources_c)]
                n_articles_c = cola.slider("Select the amount of articles to be shown",0,len(climate_filtered),int(len(climate_filtered)/2), key="ssslamosoxfj")
                st.write("---")
                display_articles(climate_filtered[:n_articles_c], c=0)

            with tab7:
                st.write("**Filter here:**")
                start_date = date(2023, 11, 27)
                end_date = date.today()
                end_date = max(start_date + timedelta(days=1), end_date)
                d_health = st.slider(
                    label="Select a date",
                    min_value=start_date,
                    max_value=end_date,
                    value=(start_date, end_date),
                    key="asnnjnknh"
                )
                dates_health = dates_list(d_health)
                health_tech["Day"] = pd.to_datetime(health_tech["Day"])
                health_tech_filtered_1 = health_tech[health_tech["Day"].isin(dates_health)]
                cola, colb = st.columns([1,2])
                sources_h = colb.multiselect ("Select the News Site you want to be shown",sources, default=sources, key="asmfnajavthnasmkf")
                health_tech_filtered = health_tech[health_tech["Source"].isin(sources_h)]
                n_articles_h = cola.slider("Select the amount of articles to be shown",0,len(health_tech_filtered),int(len(health_tech_filtered)/2), key="ssplantosoxfj")
                st.write("---")
                display_articles(health_tech_filtered[:n_articles_h], c=0)

            with tab8:
                st.write("**Filter here:**")
                start_date = date(2023, 11, 27)
                end_date = date.today()
                end_date = max(start_date + timedelta(days=1), end_date)
                d_just = st.slider(
                    label="Select a date",
                    min_value=start_date,
                    max_value=end_date,
                    value=(start_date, end_date),
                    key="asnonjxnknh"
                )
                dates_just = dates_list(d_just)
                justice["Day"] = pd.to_datetime(justice["Day"])
                justice_filtered_1 = justice[justice["Day"].isin(dates_just)]
                cola, colb = st.columns([1,2])
                sources_j = colb.multiselect ("Select the News Site you want to be shown",sources, default=sources, key="asmfnplcsthnasmkf")
                justice_filtered = justice[justice["Source"].isin(sources_j)]
                n_articles_j = cola.slider("Select the amount of articles to be shown",0,len(justice_filtered),int(len(justice_filtered)/2), key="ssplanmlfwoxfj")
                st.write("---")
                display_articles(justice_filtered[:n_articles_h], c=0)

        if selected == "Analyzer":
            st.write("---")
            # Embed Tableau visualization using an iframe
            tableau_embed_code = """
                <iframe src="https://public.tableau.com/views/HeadlineHub/Dashboard1?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link:showVizHome=no&:embed=true" width="1366" height="768"></iframe>
            """
            tab9, tab10, tab11 = st.tabs(["Explorer", "Dashboard", "WordFreq"])
            with tab9:
                choice = option_menu(
                menu_title=None,
                options=["Analyze Article via URL","Analyze Article via Title and Text"],
                menu_icon = "cast",
                default_index=0,
                orientation="horizontal",
            )
                if choice == "Analyze Article via URL":
                    url = st.text_input("Enter URL here: ")
                    st.write("---")
                    if len(url)>2:
                        try:
                            title, text = web_scrape(url)
                            if len(title) > 5 and len(text) > 150:
                                sum_stats_2(title, text)
                            elif len(title) ==0 or len(text)==0:
                                st.write("Please Enter Article Text and Title above! ")
                            else:
                                st.write("Ooops Something went wrong... Please try again!")
                        except Exception:
                            st.write("Please Enter a valid URL")
                if choice == "Analyze Article via Title and Text":
                    title = st.text_input("Enter the title of the Article here: ")
                    text = st.text_input("Enter the text of the Article here: ")
                    st.write("---")
                    if len(title) > 5 and len(text) > 150:
                        sum_stats_2(title, text)
                    elif len(title) ==0 or len(text)==0:
                        st.write("Please Enter Article Text and Title above! ")
                    else:
                        st.write("Ooops Something went wrong... Please try again!")
            with tab10:
                tab10.write("You can find a more specific Docomentation below!")
                tab10.markdown(tableau_embed_code, unsafe_allow_html=True)
            

            categories = ['Politics', 'Entertainment and Lifestyle', 'International News','Business and Economy', 'Sports', 'Climate and Environment','Health | Science | Technology', 'Law and Justice']
            with tab11:
                st.write("**Filter here:**")
                start_date = date(2023, 11, 27)
                end_date = date.today()
                end_date = max(start_date + timedelta(days=1), end_date)
                d_tool = st.slider(
                    label="Select a date",
                    min_value=start_date,
                    max_value=end_date,
                    value=(start_date, end_date),
                    key="asnonweznknh"
                )
                dates_tool = dates_list(d_tool)
                df["Day"] = pd.to_datetime(df["Day"])
                df_filtered_1 = df[df["Day"].isin(dates_tool)]
                cats = st.multiselect("Select the Categories to look for words for", categories, categories)
                df_filtered = df_filtered_1[df_filtered_1["cat_label"].isin(cats)]
                st.write("---")
                word = tab11.text_input("Enter a word you would like to know the frequency of in the Box below!", max_chars=40)
                st.write("---")
                word_tool(df=df_filtered, word=word)

        if selected == "Search":
            sources = ["CNN", "Fox News", "ABC", "Politico", "Washington Post", "New York Times", "MSNBC", "Reuters", "USA TODAY", "NPR", "Chicago Tribune"]
            categories = ['Politics', 'Entertainment and Lifestyle', 'International News','Business and Economy', 'Sports', 'Climate and Environment','Health | Science | Technology', 'Law and Justice']
            st.write("---")
            st.title(f"Search here for specific News Articles")
            st.write("Search via URL")
            url = st.text_input("Enter copied URL here: ") 
            if len(url)>2:
                df_search = df[df["URL"]==url]
                if len(df_search)!=0:
                    display_articles(df_search, c=1)
            else:
                st.write("No articles were found with the given URL")
            st.write("---")

            st.write("Search via Features")
            start_date = date.today()
            date_s = st.date_input("Enter Date here: ", start_date)
            date_search = [date_s]
            df["Day"] = pd.to_datetime(df["Day"])
            cats_search = st.multiselect("Select the Categories to look for in", categories, "Politics")
            df_filtered_1 = df[df["Day"].isin(date_search)]
            df_filtered_2 = df_filtered_1[df_filtered_1["cat_label"].isin(cats_search)]
            
            sources_search = st.multiselect ("Select the News Site you want to be shown",sources, default=sources, key="asmfnamnjwasmkf")
            df_filtered_3 = df_filtered_2[df_filtered_2["Source"].isin(sources_search)]
            colh, colj = st.columns(2)
            sentiment_s = colh.select_slider("Choose Sentiment", ["Positive", "Neutral", "Negative"], "Neutral")
            if sentiment_s == "Positive":
                df_filtered_4 = df_filtered_3[df_filtered_3["sentiment_score"]>0.2]
            elif sentiment_s == "Negative":
                df_filtered_4 = df_filtered_3[df_filtered_3["sentiment_score"]<0.2]
            else:
                df_filtered_4 = df_filtered_3[(df_filtered_3["sentiment_score"] > -0.2) & (df_filtered_3["sentiment_score"] <= 0.2)]
            
            clickbait_s = colj.selectbox("Choose Clickbait Probability (0 for unlikely | 1 for probable)", [0, 1], 0)
            df_filtered_5 = df_filtered_4[df_filtered_4["isClickbait"]==clickbait_s]
            readability = ["easy", "medium", "hard"]
            selected_option = colh.select_slider("Select Readbility:", readability, "medium")
            if selected_option == "easy":
                df_filtered_6 = df_filtered_5[df_filtered_5["Readability"]<8.75]
            elif selected_option == "hard":
                df_filtered_6 = df_filtered_5[df_filtered_5["Readability"]>11.25]
            else:
                df_filtered_6 = df_filtered_5[(df_filtered_5["Readability"] > 8.75) & (df_filtered_5["Readability"] <= 11.25)]

            st.write("---")
            display_articles(df_filtered_6, c=1)

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')



