import numpy as np
import pandas as pd
movies=pd.read_csv('movies.csv')
ratings=pd.read_csv('rating.csv')
# to find duplicates
dupl_movies=movies.groupby('title').filter(lambda x:len(x)==2)
dupl_film=dupl_movies['movieId'].values
dupl_reviews=pd.DataFrame(ratings[ratings['movieId'].isin(dupl_film)]['movieId'].value_counts())
# dupl_reviews=dupl_reviews[['movieId','count']]
dupl_reviews.reset_index(inplace=True)
dupl_reviews.columns=['movieId','count']
dupl_movies=dupl_movies[['movieId','title']]
dupl_movies=pd.DataFrame(dupl_movies)
df=pd.merge(dupl_movies,dupl_reviews,on='movieId')
df.set_index('title')
movies


# In[5]:


# HERE MOVIES TITLES GIVE US A GREAT FREATURE TO PREDICT MOVIES SO WE THEREFORE USE MOVIE TITLE TO BE RATED BY S-P-L-I-T-T-I-N-G
# THE MOVIE GENRE COLUMNS INTO UNIQUE GENRES AND CONVERTING THEM INTO CATEGORICAL COLUMNS


# In[6]:
movies=movies.loc[~movies['movieId'].isin(df)]
ratings=ratings.loc[~ratings['movieId'].isin(df)]
genres=list(set('|'.join(list(movies['genres'])).split("|")))
genres.remove('(no genres listed)')
genres
for genre in genres:
    movies[genre]=movies['genres'].map(lambda x:1 if genre in x  else 0)
movies
# filter1=movies['title']=='Toy Story (1995)'
# filter2=movies['title'].str.contains('Jumanji (1995)').as_type(str)
# movies.where(filtehttp://localhost:8888/notebooks/Movie%20recommendation%20system.ipynb#r2,inplace=True)
# movies.head(100)
# # movies['index']=movies['title'].str.find('mon',2)


# In[7]:


# HERE THE YEAR WHEN MOVIE WAS RELEASED HAS BEEN SEPERATED IN THE FORM OF DECADES(foe eg:2000-2010 movies belonging above 2000
# And below 2010 will fall into category and similarly for other decade pairs too)
import re
movies['year']=movies['title'].map(lambda val: int(re.search('\(([0-9]{4})\)',val).group(1)) 
                                     if re.search('\(([0-9]{4})\)',val)!= None 
                                     else 0)   
for decades in range(1930,2020,10):
    movies['decade'+str(decades)]=np.where((movies['year']<decades+10) &(movies['year']>=decades),1,0)
movies['decades_none']=np.where(movies['year']==0,1,0)
movies['decade_notnone']=np.where((movies['year']!=0)&(movies['year']<1930),1,0)
movie_rates=pd.merge(movies,ratings,on='movieId')
movie_rates.drop('timestamp',axis=1,inplace=True)
# set(movie_rates["userId"])
movie_rates


# In[8]:


# RECOMENDATION IS DONE BY USING MOVIES BASED ON USERS RATINGS AND BY KEEPING USER IDS AS COLUMNS[SEE PIVOT TABLE CODE LINE]
import sklearn.metrics.pairwise as pw
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
pivot_table = pd.pivot_table(movie_rates, index='userId', columns='title', values='rating')

pivot_table.fillna(0,inplace=True)
item_similarity=pivot_table.fillna(0).T
sparse_matrix=sparse.csr_matrix(item_similarity) 
cosine_sim=pw.cosine_similarity(sparse_matrix)
cosine_sim
df=pd.DataFrame(cosine_sim,columns=pivot_table.columns,index=pivot_table.columns)
df
# movie_test1='Jumanji (1995)'
# df
# cosine_df=pd.DataFrame(df[movie_test1].sort_values(ascending=False))
# cosine_df.reset_index(level=0,inplace=True)
# cosine_df.columns=['title','cosine Similarity']
# cosine_df=cosine_df.set_index("title")
# cosine_df
# pivot_table
# HERE I HAVE PICKED  A  MARVEL MOVIE CALLED "Avengers: Infinity War - Part I (2018)" AND USED THIS COSINE-SIMILARITYALGORITHM
# NOW ALL THE MOVIES RELATED TO MARVEL AND HOW MUCH CORRELATION THIS MOVIE HAS WITH OTHER MOVIES HAS BEEN LISTED IN THIS OUTPUT.
# THIS IS HOW A MOVIE RECOMMENDATION WORKS AND RECOMMENDS US MOVIES AND SERIES LIKE HOW IT IS IN NETFLIX,AMAZON PRIME etc.


# In[9]:


# def item_based_movie_recommendations(target_movie, cosine_similarity, pivot_table, top_n=10):
#     target_movie_index = pivot_table.loc(target_movie)
#     similar_movies = pd.Series(cosine_similarity[target_movie_index], index=pivot_table.index).sort_values(ascending=False)[1:]
#     target_movie_rated_movies = pivot_table.get_loc[target_movie]
#     target_movie_rated_movies = target_movie_rated_movies[target_movie_rated_movies > 0]
#     movie_recommendations = {}

#     for movie, similarity_score in similar_movies.items():
#         similar_movie_rated_movies = pivot_table.T.get_loc[movie]
#         similar_movie_rated_movies = similar_movie_rated_movies[similar_movie_rated_movies > 0]

#         # Find unrated movies that the similar movie has been rated highly
#         unrated_movies = similar_movie_rated_movies.index.difference(target_movie_rated_movies.index)
# # #         print(similar_movie_rated_movies)
# #         print(unrated_movies)
#         # Find unrated movies that the similar movie has been rated highly
#         unrated_movies =list(similar_movie_rated_movies.index.difference(target_movie_rated_movies.index))
# #         print(unrated_movies)
#         for unrated_movie in unrated_movies:
#             if unrated_movie in movie_recommendations:
#                 movie_recommendations[unrated_movie] += similarity_score * similar_movie_rated_movies.loc[unrated_movie]
#             else:
#         # If the movie is not in movie_recommendations, add it with similarity score
#                 movie_recommendations[unrated_movie] = similarity_score * similar_movie_rated_movies.loc[unrated_movie]


#     sorted_movie_recommendations = sorted(movie_recommendations.items(), key=lambda x: x[1], reverse=True)
# #     return sorted_movie_recommendations[:top_n]

# # # Example usage:
# target_movie="Avengers: Infinity War - Part I (2018)"
# top_recommendations = item_based_movie_recommendations(target_movie, cosine_sim, pivot_table)
# # # # # for movie, score in top_recommendations:
# # # # #     print(f"Movie: {movie}, Recommendation Score: {score}")
# top_recommendations


# In[10]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
user_item_matrix = pd.pivot_table(movie_rates, index='userId', columns='title', values='rating')
user_item_matrix = user_item_matrix.fillna(0)

# Calculate item-item similarity using cosine similarity
item_similarity = cosine_similarity(user_item_matrix.T)
similar_suggestions={}
# Create a DataFrame for item-item similarity
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
# target_movie_rated_movies = item_similarity_df[target_movie]
# target_movie_rated_movies = target_movie_rated_movies[target_movie_rated_movies > 0]

similar_suggestions = {}  # Initialize the dictionary

for movie, similarity_score in item_similarity_df.iterrows():
    similar_movie_rated_movies = item_similarity_df[movie]
    similar_movie_rated_movies = similar_movie_rated_movies[similar_movie_rated_movies > 0].index.tolist()  
# print(target_movie_rated_movies)

# sorted_similar_suggestions = sorted(similar_suggestions.items(), key=lambda x: x[1], reverse=True)
# sorted_similar_suggestions[:10]

# #         print(similar_suggestions)
# def item_based_movie_recommendations(target_movie,item_similarity_df,top_n=10):
#     target_movie_rated_movies =item_similarity_df[target_movie]
#     target_movie_rated_movies = target_movie_rated_movies[target_movie_rated_movies > 0]
# #     print(target_movie_rated_movies)
#     for movie, similarity_score in item_similarity_df.iterrows():
#         similar_movie_rated_movies = item_similarity_df[movie]
#         similar_movie_rated_movies = similar_movie_rated_movies[similar_movie_rated_movies > 0]
# #         print(similar_movie_rated_movies)
#         unrated_movies = similar_movie_rated_movies.index.difference(target_movie_rated_movies.index)
# # #         print(unrated_movies)
#         similar_suggestions={}
#         for unrated_movie in unrated_movies:
#             if unrated_movie in similar_suggestions:
#                 similar_suggestions[unrated_movie] += similarity_score * similar_movie_rated_movies.loc[unrated_movie]
#             else:
#                 similar_suggestions[unrated_movie] = similarity_score * similar_movie_rated_movies.loc[unrated_movie]
#         print(similar_suggestions)
#     print(unrated_movies)
#     return sorted_similar_suggestions[:top_n]
#         print(similar_suggestions)
#     sorted_movie_recommendations = sorted(similar_suggestions.items(), key=lambda x: x[1], reverse=True)
#     return similar_suggestions
# # #         print(similar_movie_target_movie = 'Star Wars: Episode IV - A New Hope (1977)'
# similar_movies =item_based_movie_recommendations(target_movie,item_similarity_df)
# print(f"Top 5 movies similar to {target_movie}:\n{similar_movies}")
# similar_moviesrated_movies
# #         print(unrated_movies)
#         # Find unrated movies that the similar movie has been rated highly
#     unrated_movies =list(similar_movie_rated_movies.index.difference(target_movie_rated_movies.index))
# def get_similar_movies(movie_title, top_n=5):
#     similar_movies = item_similarity_df[movie_title].sort_values(ascending=False).head(top_n)


# # Example: Get top 5 similar movies to
# a given movie
# target_movie = 'Star Wars: Episode IV - A New Hope (1977)'
# similar_movies =item_based_movie_recommendations(target_movie,item_similarity_df)
# # print(f"Top 5 movies similar to {target_movie}:\n{similar_movies}")
# similar_movies


# In[12]:


similar_suggestions={}
def get_similar_movies(df,movie_title,similar_rated_movies,top_n=50):
    target_movie_rated_movies =df[movie_title]
    target_movie_rated_movies =target_movie_rated_movies[target_movie_rated_movies > 0]
    unrated_movies =list(set(similar_rated_movies).difference(target_movie_rated_movies.index))
    for unrated_movie in unrated_movies:
        rating = movie_rates[movie_rates["title"] == unrated_movie]["rating"].iloc[0]
        if unrated_movie in similar_suggestions:
            similar_suggestions[unrated_movie] += similarity_score*rating
        else:
            similar_suggestions[unrated_movie] = similarity_score*rating
    similar_movies =df[movie_title].sort_values(ascending=False).head(top_n)
    return similar_movies.index
movie_title='Star Wars: Episode IV - A New Hope (1977)'
get_similar_movies(item_similarity_df,movie_title,similar_movie_rated_movies)


# In[13]:


# NOW RECOMENDATION IS DONE BY USING USER ID'S BY KEEPING MOVIES AS COLUMNS NOW 
pivot_table1=pd.pivot_table(movie_rates,index='title',columns=['userId'],values='rating').T
pivot_table1.fillna(0,inplace=True)
pivot_table1

# pivot_table1
# A USER-ID IS TAKEN AS AS SAMPLE INPUT AND THEN THE USING COSINE SIMILARITY ALGORITHM,IT RECOMMENDS MOVIES/SHOWS BASED ON 
# WHAT THE USER HAS WATCHED.


# In[14]:


sparse_matrix1=sparse.csr_matrix(pivot_table1)
sparse_matrix1
cosine_sim1=pw.cosine_similarity(sparse_matrix1)
cosine_sim1
df=pd.DataFrame(cosine_sim1,index=pivot_table1.index)
df
userid=284
cosine_df1=pd.DataFrame(df[userid].sort_values(ascending=False))
cosine_df1.reset_index()
cosine_df1.rename(columns={143:'USER CO-RELATION'},inplace=True) 
cosine_df1


# In[15]:


def user_based_recommendations(target_user,cosine_similarity,pivot_table1,top_n=10):
#     target_user=143
    similar_users = pd.Series(cosine_sim1[pivot_table1.index.get_loc(target_user)], index=pivot_table1.index).sort_values(ascending=False)[1:]
    similar_users
    target_user_rated_movies = pivot_table1.loc[target_user]
    target_user_rated_movies = target_user_rated_movies[target_user_rated_movies>0]
    recommendations={}
    for user,similarity_score in similar_users.items():
        similar_user_rated_movies=pivot_table1.loc[user]
        similar_user_rated_movies = similar_user_rated_movies[similar_user_rated_movies>0]
#     # Find u nrated movies that the similar user has rated highly
        unrated_movies = similar_user_rated_movies.index.difference(target_user_rated_movies.index)
        for movie in unrated_movies:
            if movie in recommendations:
                recommendations[movie] += similarity_score * similar_user_rated_movies.loc[movie]
            else:
            # If the movie is not in recommendations, add it with similarity score
                recommendations[movie] = similarity_score * similar_user_rated_movies.loc[movie]
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    user_df=pd.DataFrame(sorted_recommendations,columns=["Movies","similarity score"])
#     for movie, score in sorted_recommendations[:top_n+1]:
#         return f"Movie: {movie}, Recommendation Score: {score}"
    return user_df["Movies"].head(top_n)

target_user=60
print(user_based_recommendations(target_user,cosine_sim1,pivot_table1,top_n=30))


# In[16]:


movie_rates


# In[17]:


similarity_scores = []
# Preprocess genres
def preprocess_genres(genres):
    return set(genres.split())

# Calculate genre similarity using Jaccard similarity
def calculate_genre_similarity(genres1, genres2):
    intersection = len(genres1.intersection(genres2))
    union = len(genres1) + len(genres2) - intersection
    return round(intersection / union, 3)

# Calculate movie similarity based on genre similarity
def calculate_movie_similarity(target_genres, other_genres):
    return calculate_genre_similarity(target_genres, other_genres)

# Example target movie genres
def similar_genre_movies(target_movie, movie_rates):
    genre = movie_rates.loc[movie_rates['title'] == target_movie, 'genres'].iloc[0]
    target_movie_genres = preprocess_genres(genre)

    similarity_scores = []  # Initialize the list to hold similarity scores
    for index, row in movie_rates.iterrows():
        movie_title = row['title']
        movie_genres = preprocess_genres(row['genres'])

        similarity = calculate_movie_similarity(target_movie_genres, movie_genres)
        similarity_scores.append((movie_title, similarity, movie_genres))
    
    similarity_scores.sort(key=lambda x: (x[1], x[0]), reverse=True)  # Sort by similarity and then by movie title
    genre_df = pd.DataFrame(similarity_scores, columns=["movie", "similarity_score", "genre"])
    genre_based=genre_df["movie"].drop_duplicates().head(50)
    return genre_based

movie="Bungo Stray Dogs: Dead Apple (2018)"
similar_genre_movies(movie,movie_rates)


# In[ ]:


# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split

# # target_movie = 'Black Butler: Book of the Atlantic (2017)'
# genre_embeddings = pd.DataFrame(movie_rates['genre_embedding'].tolist(), index=movie_rates.genres)
# genre_embeddings
# pca = PCA(n_components=100)
# # genre_embeddings_pca = pca.fit_transform(genre_embeddings)
# # Split your data into training and validation sets
# genre_embeddings_train, genre_embeddings_val = train_test_split(genre_embeddings, test_size=0.2, random_state=42)
# genre_embeddings_train_pca = pca.fit_transform(genre_embeddings_train)
# genre_embeddings_val_pca = pca.transform(genre_embeddings_val)
# # Train the Nearest Neighbors model on genre embeddings
# nn_model = NearestNeighbors(n_neighbors=10,metric='cosine', algorithm='auto')
# nn_model.fit(genre_embeddings_train_pca)
# def recommend_movies_by_movie(movie_title, genre_embeddings_val_pca, nn_model, movie_rates, pca, k=10):
#     # Get the genre embeddings of the target movie's genres from the genre_embeddings DataFrame
#     target_movie_genres = movie_rates[movie_rates["title"] == movie_title]['genres'].iloc[0]
#     target_movie_genre_embeddings = genre_embeddings.loc[target_movie_genres]
    
# #     # Apply PCA to the target movie's genre embeddings
#     target_movie_genre_pca = pca.transform(target_movie_genre_embeddings.values.reshape(1, -1))

# #     # Find the k-nearest neighbors of the target movie's genre embeddings
# #     distances, indices = nn_model.kneighbors(target_movie_genre_pca)

# #     # Get the indices of the recommended movies in the genre_embeddings_val DataFrame
# #     recommended_movie_indices = indices[0]

# #     # Get the movie titles from the indices
# #     recommended_movie_titles = movie_rates.iloc[recommended_movie_indices]['title'].tolist()

#     return target_movie_genre_pca

# target_movie_title='Casino (1995)'
# recommended_movies = recommend_movies_by_movie(target_movie_title, genre_embeddings_val_pca, nn_model, movie_rates, pca, k=10)
# recommended_movies 
# # Perform PCA on genre_embeddings
# # genre_embeddings_df = pd.DataFrame(genre_embeddings_pca, index=genre_embeddings.index)
# # Split your data into training and validation sets
# # target_movie_genre=movie_rates[movie_rates["title"]==target_movie_title]['genres'].iloc[0]
# # target_movie_index = genre_embeddings_df.index.get_loc(target_movie_genre)
# # target_movie_index
# # genre_embeddings_train, genre_embeddings_val = train_test_split(genre_embeddings_df, test_size=0.2, random_state=42)

# # # Now you can use genre_embeddings_train for training and genre_embeddings_val for validation
# # k = 5
# # nn_model = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='auto')
# # nn_model.fit(genre_embeddings_train)
# # distances, indices = nn_model.kneighbors(genre_embeddings_val.iloc[target_movie_index].values.reshape(1, -1))

# # # Step 3: Get the movie titles corresponding to the indices of the k-nearest neighbors
# # recommended_movie_titles = filtered_movie_rates.iloc[indices[0]].index.tolist()

# # # Print the recommended movie titles
# # print("Recommended Movies:")
# # for movie_title in recommended_movie_titles:
# #     print(movie_title)


# # target_movie_index = [idx for idx, title in enumerate(movie_rates["title"]) if title in genre_embeddings_val.index]
# # target_movie_index
# # if len(target_movie_index) == 0:
# #     print("Target movie not found in the validation set.")
# #     # Handle this case as per your requirement, e.g., return an empty list or handle it in a specific way
# # else:
# #     target_movie_index = target_movie_index[0]
# #     # Get the k-nearest neighbors of the target movie
# #     distances, indices = nn_model.kneighbors(genre_embeddings_val[target_movie_index].reshape(1, -1))

# # Get the k-nearest neighbors of the target movie
# # distances, indices = nn_model.kneighbors(genre_embeddings_val[target_movie_index].reshape(1, -1))

# # Get the recommended movie titles from the indices
# # 
# # Assuming you have already created the genre_embeddings matrix and movie_ratings dataframe
# # Proceed with the rest of the code
# # recommended_movies = [genre_embeddings_val.index[i] for i in indices]

# # # Assuming you have a function get_ground_truth(target_movie) that returns the ground truth movie recommendations for a given target_movie
# # ground_truth = get_ground_truth(target_movie)

# # # Calculate precision and recall
# # precision = precision_score(ground_truth, recommended_movies, average='micro')
# # recall = recall_score(ground_truth, recommended_movies, average='micro')


# In[ ]:


# target_movie_title='Jumanji (1995)'
# # target_movie_index = [idx for idx, title in enumerate(movie_rates["title"]) if title in genre_embeddings_val.index]
# # target_movie_index
# # Assuming you have 'genre_embeddings_val', 'filtered_movie_rates', and 'k' (number of nearest neighbors) already defined

# # Step 1: Find the index of the target movie in the validation set
# genre_embeddings_val=pd.DataFrame(genre_embeddings_val)

# target_movie_index = genre_embeddings_val.index.get_loc(target_movie_title)

# # # Step 2: Use the Nearest Neighbors model to find the k-nearest neighbors of the target movie
# # distances, indices = nn_model.kneighbors(genre_embeddings_val.iloc[target_movie_index].values.reshape(1, -1))

# # # Step 3: Get the movie titles corresponding to the indices of the k-nearest neighbors
# # recommended_movie_titles = filtered_movie_rates.iloc[indices[0]].index.tolist()

# # # Print the recommended movie titles
# # print("Recommended Movies:")
# # for movie_title in recommended_movie_titles:
# #     print(movie_title)


# In[ ]:


# # !pip install keras
# from sklearn.decomposition import PCA
# target_movie='Avengers: Infinity War - Part I (2018)'
# genre_embeddings = pd.DataFrame(movie_rates['genre_embedding'].tolist(), index=movie_rates.index)
# pca = PCA(n_components=100)
# genre_embeddings_pca = pca.fit_transform(genre_embeddings)
# k=5
# nn_model = NearestNeighbors(n_neighbors=k,metric='cosine', algorithm='auto')
# nn_model.fit(genre_embeddings_pca)
# target_movie_index = 0  # Replace with the index of the target movie in the genre_embeddings matrix

# # Assuming you have already created the genre_embeddings matrix and movie_ratings dataframe
# # Split your data into training and validation sets
# # For example, using train_test_split from scikit-learn
# from sklearn.model_selection import train_test_split

# genre_embeddings_train, genre_embeddings_val = train_test_split(genre_embeddings_pca,test_size=0.2, random_state=42)

# # Create a list of hyperparameter values to experiment with
# n_components_list = [50, 75, 100, 125, 150]

# # Initialize lists to store evaluation results
# precision_scores = []
# recall_scores = []
# map_scores = []

# for n_components in n_components_list:
#     # Create and fit the PCA model with the current n_components value
#     pca = PCA(n_components=n_components)
#     genre_embeddings_train_pca = pca.fit_transform(genre_embeddings_train)
    
#     # Train the Nearest Neighbors model on the training data
#     nn_model = NearestNeighbors(metric='cosine', algorithm='auto')
#     nn_model.fit(genre_embeddings_train_pca)
    
#     # Calculate similarities and evaluate on the validation set
#     genre_embeddings_val_pca = pca.transform(genre_embeddings_val)
#     distances, indices = nn_model.kneighbors(genre_embeddings_val_pca)
    # Perform evaluation using appropriate metrics (e.g., precision, recall, MAP)
    # Calculate precision, recall, and MAP based on the recommendations and ground truth
    # Append the scores to the respective lists
    
    # Example evaluation using scikit-learn's metrics (you may need to adapt this depending on your specific evaluation requirements)
#     from sklearn.metrics import precision_score, recall_score, average_precision_score
    
#     # Assuming you have a function get_ground_truth(target_movie) that returns the ground truth movie recommendations for a given target_movie
#     ground_truth = get_ground_truth(target_movie)
    # Assuming you have the target_movie as a string

    # Or you can return an empty list as recommendations
#     recommended_movies = []

#     recommended_movies = [genre_embeddings_val.index[i] for i in indices]
    
#     precision = precision_score(ground_truth, recommended_movies, average='micro')
#     recall = recall_score(ground_truth, recommended_movies, average='micro')
#     map_score = average_precision_score(ground_truth, distances, average='micro')
    
#     precision_scores.append(precision)
#     recall_scores.append(recall)
#     map_scores.append(map_score)

# # Print or plot the evaluation results to analyze the impact of different hyperparameter settings
# print("N_Components List:", n_components_list)
# print("Precision Scores:", precision_scores)
# print("Recall Scores:", recall_scores)
# print("MAP Scores:", map_scores)

# # Choose the best hyperparameter setting based on the evaluation results
# best_n_components = n_components_list[precision_scores.index(max(precision_scores))]
# print("Best n_components:", best_n_components)

# Get indices of similar movies (excluding the target movie itself)
# similar_movie_indices = indices[0][1:]
# similar_movie_indices
# # Get the corresponding movie titles
# similar_movies = movie_rates.iloc[similar_movie_indices]
# print(similar_movies[['title', 'genres']])
# Compute cosine similarity matrix
# cosine_sim = linear_kernel(genre_embeddings, genre_embeddings)
# genre_embeddings
# # Step 4: Define a function to get movie recommendations based on content similarity
# def content_based_movie_recommendations(movie_title, cosine_sim=cosine_sim, movie_rates=movie_rates):
#     # Get the index of the movie with the given title
#     idx = movie_rates.index[movie_rates['title'] == movie_title].tolist()[0]

#     # Get the pairwsise similarity scores of all movie_rates with the given movie
#     sim_scores = list(enumerate(cosine_sim[idx]))

#     # Sort the movie_rates based on the similarity scores in descending order
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     # Get the top 10 similar movie_rates (excluding the movie itself)
#     top_similar_movie_rates = sim_scores[1:11]

#     # Get the movie titles from the indices
#     movie_indices = [movie[0] for movie in top_similar_movie_rates]
#     recommended_movie_rates = movie_rates['title'].iloc[movie_indices].tolist()

#     return recommended_movie_rates

# # Example usage:
# target_movie = "The Dark Knight (2008)"
# recommended_movie_rates = content_based_movie_recommendations(target_movie)
# print(f"Movie: {target_movie}, Content-Based Recommendations: {recommended_movie_rates}")


# In[ ]:


# def get_ground_truth(movie_title, movie_rates=movie_rates, top_n=10):
#     # Filter the movie_rates dataframe to get the ratings of the target movie
#     target_movie_ratings = movie_rates[movie_rates['title'] == movie_title]

#     if target_movie_ratings.empty:
#         # Return an empty list if the target movie is not found in the dataset
#         return []

#     # Sort the entire movie_rates dataframe by rating in descending order
#     sorted_movie_ratings = movie_rates.sort_values(by='rating', ascending=False)

#     # Get the top N rated movies (excluding the target movie itself)
#     top_rated_movies = sorted_movie_ratings['title'].iloc[:top_n+1].tolist()
#     return list(set(top_rated_movies))

# ground_truth=get_ground_truth("Deadpool 2 (2018)", movie_rates=movie_rates, top_n=10)
# ground_truth


# In[ ]:


# # from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.models import Word2Vec

# from sklearn.metrics.pairwise import linear_kernel
# from sklearn.neighbors import NearestNeighbors

# # Assuming you have a DataFrame called 'movie_rates' with columns 'title' and 'genres'

# # Step 1: Preprocess the genres column to join the genres separated by "|"
# movie_rates['genres'] = movie_rates['genres'].apply(lambda x: " ".join(x.split("|")))
# movie_rates
# # Step 2: Create a TF-IDF vectorizer to convert movie genres into numeric vectors
# model = Word2Vec(sentences=movie_rates['genres'], vector_size=100, window=5, min_count=1, workers=4)

# # Create a function to get the average word embeddings for genres
# def get_average_embedding(genres):
#     embeddings = [model.wv[genre] for genre in genres if genre in model.wv]
#     if embeddings:
#         return sum(embeddings) / len(embeddings)
#     else:
#         return None

# # Create a new column in the DataFrame to store the genre embeddings
# movie_rates['genre_embedding'] = movie_rates['genres'].apply(get_average_embedding)

# # Drop rows where no embeddings are found*

# # Convert genre embeddings to a numpy array for cosine similarity computation
# movie_rates = movie_rates.dropna(subset=['genre_embedding'])

