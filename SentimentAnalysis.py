#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:56:15 2024

@author: rabbiyayounas
"""

import pandas as pd

import numpy as np

#to make the graphics
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

import nltk

# to check the sentimentaal value of a sentence
from nltk.sentiment import SentimentIntensityAnalyzer

#to use the progress bar. Use the progress bar in loops or functions to visualize the progress of your code execution.
from tqdm import tqdm

nltk.download('vader_lexicon')

#punkt is a pre-trained tokenizer model provided by the Natural Language Toolkit (NLTK) library. 
nltk.download('punkt')

#used for making the Part of Speech for each token
nltk.download('averaged_perceptron_tagger')
#read in data 

df= pd.read_csv('Reviews.csv')

print(df.head())


# this command shows value 0 of the column Text
# we will run our sentiment analysis on thos 
print(df['Text'].values[0])

# Get the shape of the DataFrame
dataframe_shape = df.shape

# Print the shape
print("The shape of the DataFrame is:", dataframe_shape)

# now we have to downsize the dataset. This is a huge dataset.
# we have donwsied the dataframe to only 500 lines.
df=df.head(500)

# Get the shape of the DataFrame
dataframe_shape = df.shape

# Print the shape
print("The shape of the DataFrame is:", dataframe_shape)

#get a quick summary of the reviews
# first we will get the score colum. 
#we will get value count of the score (rating) column. 
#it means how many times each rating happened
#then we deploy a plot on it to visualize it 

ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
         title='Count of Reviews by Stars',
         figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()

# implement nltk on one text/ review from the file 
example= df['Text'].values[50]
#print(example)

print(example)

#tokenize one text/review  
tokens = nltk.word_tokenize(example)
#print(tokens)

#part of speech for each tokem
tagged=nltk.pos_tag(tokens)
#print(tagged)



#Doing Sentiment AWnalysis using VADER

#sia is a object created 
sia = SentimentIntensityAnalyzer()

# check this on any othr example 
P1= sia.polarity_scores('Rabbiya is a horrible PErson')
print(P1)

P2= sia.polarity_scores(example)
print(P2)


#Run the porality on the entire dataset
#total=len(df)) = total length of dataset
#df.iterrows() =

DataFrameText= df['Text']


# List to store the results
res = {}

# Iterate over the rows of the DataFrame with a progress bar
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

# Print the stored records
print("Sentiment Scores for each record:")
for myid, scores in res.items():
    print(f"ID: {myid}, Scores: {scores}")
    
 

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


# Now we have sentiment score and metadata
print(vaders.head())
    

#Plot VADER results

ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()
    


fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()
    
# Calculate average sentiment scores
average_scores = vaders[['neg', 'neu', 'pos', 'compound']].mean()


# Determine which score is the highest and which is the lowest
highest_score = average_scores.idxmax()
lowest_score = average_scores.idxmin()


# Print summary
print("Average Sentiment Scores:")
print(average_scores)
print(f"\nHighest average score: {highest_score} ({average_scores[highest_score]})")
print(f"Lowest average score: {lowest_score} ({average_scores[lowest_score]})")

# Print the first few rows of the resulting DataFrame
print("\nSentiment Scores for each record:")
print(vaders.head())
    


