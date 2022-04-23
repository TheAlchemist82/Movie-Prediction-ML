import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

md = pd.read_csv("movies.csv")
#printing first five rows of the dataset
# print(md.head())
# print(md.shape)
sf = ['genres', 'keywords', 'cast', 'director', 'original_title', 'tagline']

# replacing null values 
for f in sf:
    md[f] = md[f].fillna('')

#combining relevant features
cf = md['genres']+' '+md['keywords']+' '+md['cast']+' '+md['director']+' '+md['original_title']+' '+md['tagline']
# print(cf)

#converting text to feature vectors

v = TfidfVectorizer()
fv = v.fit_transform(cf)
# print(fv)

#getting similarity score using cosine_similarity

s = cosine_similarity(fv)
# print(s.shape)

#user input of movie name
mn = input("Enter your favorite movie's name: ")
#creating a list with all the movies of the dataset

loat = md['title'].tolist()
# print(loat)

#finding close match for movie given by the user
fcm = difflib.get_close_matches(mn, loat)
# print(fcm)
cm = fcm[0]
print(cm)

#find the index of the movie
iom = md[md.title == cm]['index'].values[0]
# print(iom)

#getting a list of similar movies
sm  = list(enumerate(s[iom]))
# print(sm)

#sorting the movies based on the sm

ssm = sorted(sm, key = lambda x: x[1], reverse = True)
# print(ssm)A

#print the name of sm based on index

print("Suggeseted movies: \n")
i = 1
for m in ssm:
    index = m[0]
    tfi = md[md.index == index]['title'].values[0]
    if (i<31):
        print(i, '.', tfi)
        i += 1