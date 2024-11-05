import csv
import re
import pandas as pd
import nltk
from wordcloud import WordCloud # type: ignore
import matplotlib.pyplot as plt # type: ignore
import plotly.express as px
import plotly.graph_objects as go
from colorama import Fore, Style
from typing import Dict
import streamlit as st
from collections import Counter
import os
import google.generativeai as genai

# Set your API key
api_key = 'AIzaSyDENmkR2zOJDCdqG1roGaEznmCwLgiS6Uc'
os.environ['GOOGLE_API_KEY'] = api_key

# Configure the API client with your API key
genai.configure(api_key=api_key)

# Initialize the sentiment analyzer
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def extract_video_id(youtube_link):
    video_id_regex = r"^(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(video_id_regex, youtube_link)
    if match:
        video_id = match.group(1)
        return video_id
    else:
        return None

def analyze_sentiment(csv_file):
    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Read in the YouTube comments from the CSV file
    comments = []
    with open(csv_file, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            comments.append(row['Comment'])

    # Count the number of neutral, positive, and negative comments
    num_neutral = 0
    num_positive = 0
    num_negative = 0
    for comment in comments:
        sentiment_scores = sid.polarity_scores(comment)
        if sentiment_scores['compound'] == 0.0:
            num_neutral += 1
        elif sentiment_scores['compound'] > 0.0:
            num_positive += 1
        else:
            num_negative += 1

    # Return the results as a dictionary
    results = {'num_neutral': num_neutral, 'num_positive': num_positive, 'num_negative': num_negative}
    return results

def bar_chart(csv_file: str) -> None:
    # Call analyze_sentiment function to get the results
    results: Dict[str, int] = analyze_sentiment(csv_file)

    # Get the counts for each sentiment category
    num_neutral = results['num_neutral']
    num_positive = results['num_positive']
    num_negative = results['num_negative']

    # Create a Pandas DataFrame with the results
    df = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Number of Comments': [num_positive, num_negative, num_neutral]
    })

    # Create the bar chart using Plotly Express
    fig = px.bar(df, x='Sentiment', y='Number of Comments', color='Sentiment', 
                 color_discrete_sequence=['#87CEFA', '#FFA07A', '#D3D3D3'],
                 title='Sentiment Analysis Results')
    fig.update_layout(title_font=dict(size=20))

    # Show the chart
    st.plotly_chart(fig, use_container_width=True)

def plot_sentiment(csv_file: str) -> None:
    # Call analyze_sentiment function to get the results
    results: Dict[str, int] = analyze_sentiment(csv_file)

    # Get the counts for each sentiment category
    num_neutral = results['num_neutral']
    num_positive = results['num_positive']
    num_negative = results['num_negative']

    # Plot the pie chart
    labels = ['Neutral', 'Positive', 'Negative']
    values = [num_neutral, num_positive, num_negative]
    colors = ['yellow', 'green', 'red']
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                                 marker=dict(colors=colors))])
    fig.update_layout(title={'text': 'Sentiment Analysis Results', 'font': {'size': 20, 'family': 'Arial', 'color': 'grey'},
                              'x': 0.5, 'y': 0.9},
                      font=dict(size=14))
    st.plotly_chart(fig)

def create_scatterplot(csv_file: str, x_column: str, y_column: str) -> None:
    # Load data from CSV
    data = pd.read_csv(csv_file)

    # Create scatter plot using Plotly
    fig = px.scatter(data, x=x_column, y=y_column, color='Category')

    # Customize layout
    fig.update_layout(
        title='Scatter Plot',
        xaxis_title=x_column,
        yaxis_title=y_column,
        font=dict(size=18)
    )

    # Display plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def generate_word_cloud(csv_file: str) -> None:
    # Assuming your CSV file has a 'comments' column containing text data
    df = pd.read_csv(csv_file)
    
    # Concatenate all comments into a single string
    all_comments = ' '.join(df['Comment'].dropna().astype(str).tolist())
    
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)
    
    # Plot the word cloud using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Comments')
    
    # Show the plot in Streamlit
    st.pyplot(plt)

def print_sentiment(csv_file: str) -> None:
    # Call analyze_sentiment function to get the results
    results: Dict[str, int] = analyze_sentiment(csv_file)

    # Get the counts for each sentiment category
    num_neutral = results['num_neutral']
    num_positive = results['num_positive']
    num_negative = results['num_negative']

    # Determine the overall sentiment
    if num_positive > num_negative:
        overall_sentiment = 'POSITIVE'
        color = Fore.GREEN
    elif num_negative > num_positive:
        overall_sentiment = 'NEGATIVE'
        color = Fore.RED
    else:
        overall_sentiment = 'NEUTRAL'
        color = Fore.YELLOW

    # Print the overall sentiment in color
    print('\n' + Style.BRIGHT + color + overall_sentiment.upper().center(50, ' ') + Style.RESET_ALL)

def generate_reaction_paragraph(csv_file: str) -> None:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Ensure the correct column name is used
    column_name = 'Comment'  # Adjust this to match your actual column name
    
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in the CSV file.")
    
    # Concatenate all comments into a single string
    all_comments = ' '.join(df[column_name].dropna().astype(str).tolist())
    
    # Tokenize the comments into words
    words = all_comments.split()
    
    # Count frequencies of words
    word_freq = Counter(words)
    
    # Get the 10 most common words
    common_words = word_freq.most_common(10)
    
    # Combine the key words into a paragraph using a basic template
    reaction_paragraph = f"Based on the analysis of comments, people reacted with {' '.join([word for word, _ in common_words])}."

    # Print or display the generated paragraph
    st.text(reaction_paragraph)

def get_comments_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        comments = df['Comment'].tolist()  # Assuming the column containing comments is named 'Comment'
        return comments
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return []

def generate_summary_prompt(comments):
    comments_text = " ".join(map(str, comments))
    prompt = f"Summarize the following comments in a single cohesive paragraph and also mention the probable age group who reacted more on this video:\n\n{comments_text}"
    return prompt

def generate_text(prompt):
    # Choose a model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Generate text
    response = model.generate_content(prompt)
    
    # Return the generated response text
    return response.text

def generate_summary(csv_file: str) -> None:
    # Get comments from the CSV file
    comments = get_comments_from_csv(csv_file)

    if comments:
        # Generate the summary prompt
        summary_prompt = generate_summary_prompt(comments)
        
        # Generate summary using Gemini API
        summary = generate_text(summary_prompt)

        summary_lines = summary.split('. ')  # Splitting by '. ' (period followed by a space)
        st.text("Generated Summary:")
        for line in summary_lines:
            st.write(line)

    else:
        st.text("No comments to summarize.")
