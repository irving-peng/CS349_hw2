import math
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer

'''
returns Euclidean distance between vectors a and b
'''
def euclidean(a,b):
    if len(a) != len(b):
        raise ValueError("Both vectors must have the same dimension")
    dist = 0
    for dimension in range(len(a)):
       dist += (b[dimension] - a[dimension]) ** 2
    dist = math.sqrt(dist)

    return(dist)
        
'''
returns Cosine Similarity between vectors and b
'''
def cosim(a,b):
    if len(a) != len(b):
        raise ValueError("Both vectors must have the same dimension")
    dot_product = 0
    magnitude_a = 0
    magnitude_b = 0
    for dimension in range(len(a)):
        dot_product += a[dimension] * b[dimension]
        magnitude_a += a[dimension] ** 2
        magnitude_b += b[dimension] ** 2
    magnitude_a = math.sqrt(magnitude_a)
    magnitude_b = math.sqrt(magnitude_b)
    dist = dot_product / magnitude_b / magnitude_a
    return(dist)

'''
return pearson correlations of  a and b.
'''
def pearson_correlation(a, b):
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    numerator = 0
    denominator_x = 0
    denominator_y = 0
    for i in range(len(a)):
        numerator += (a[i] - mean_a) * (b[i] - mean_b)
        denominator_x += (a[i] - mean_a) ** 2
        denominator_y += (b[i] - mean_b) ** 2
    denominator_y = math.sqrt(denominator_y)
    denominator_x = math.sqrt(denominator_x)
    denominator = denominator_y * denominator_x
    dist = numerator / denominator
    return dist

'''
return hammering distance of two binary vectors.
'''
def hammering_distance(a, b):
    hammer_dist = 0
    for i in range(len(a)):
        hammer_dist += abs(a[i] - b[i])
    return hammer_dist

'''
returns a list of labels for the query dataset based upon labeled observations in the train dataset.
metric is a string specifying either "euclidean" or "cosim".  
All hyper-parameters should be hard-coded in the algorithm
'''
# Define k-NN function
def knn(train, query, metri="euclidean", K=5, n_comp=50):
    # Choose distance function
    if metri.lower() == "euclidean":
        dist_func = lambda x, y: np.linalg.norm(x - y)
    else:  # Cosine similarity
        dist_func = lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    
    # Separate features and labels in the training and query datasets
    train_data = np.array([sample[1:] for sample in train])
    train_labels = np.array([sample[0] for sample in train])
    query_data = np.array([sample[1:] for sample in query])
    
    # Handle missing values by imputing with the mean
    imputer = SimpleImputer(strategy='mean')
    train_data = imputer.fit_transform(train_data)
    query_data = imputer.transform(query_data)
    
    # Apply PCA if specified
    if n_comp < 784:
        pca = PCA(n_components=n_comp)
        train_data = pca.fit_transform(train_data)
        query_data = pca.transform(query_data)
    
    # Perform k-NN classification
    predictions = []
    for q in query_data:
        # Compute distances from query to each training sample
        distances = [dist_func(q, t) for t in train_data]
        # Get indices of the K nearest neighbors
        k_nearest_indices = np.argsort(distances)[:K]
        # Find the labels of these neighbors
        k_nearest_labels = train_labels[k_nearest_indices]
        # Predict the most common label among the nearest neighbors
        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
        predictions.append(most_common_label)
    
    return predictions

# Evaluation function to generate confusion matrix
def evaluate_knn(train, test, metri="euclidean", K=5, n_comp=50):
    predictions = knn(train, test, metri, K, n_comp)
    true_labels = [sample[0] for sample in test]
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=range(10))
    cm_df = pd.DataFrame(cm, index=range(10), columns=range(10))
    print("Confusion Matrix:")
    print(cm_df)


'''
helper function: read file
'''
def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)

'''
helper function: read movielens.txt and other txts"
'''
def read_movie_data(file_name):
    data_set = []
    with open(file_name, 'rt') as f:
        # Skip the header line
        header = f.readline().strip().split('\t')
        
        for line in f:
            line = line.strip()
            tokens = line.split('\t')
            
            # Map tokens to column names for each row
            row_data = {header[i]: tokens[i] for i in range(len(header))}
            data_set.append(row_data)
    
    return data_set

'''
helper function: show files
'''
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
          
'''
helper function: load data using pd
'''
# Load dataset function
def load_data(file_path):
    return pd.read_csv(file_path, delimiter="\t")

# Example usage for loading datasets
train_a = load_data('train_a.txt')
train_b = load_data('train_b.txt')
train_c = load_data('train_c.txt')
movieLens = load_data('movielens.txt')
# Add other datasets as needed

def get_top_k_similar_users(target_user, other_users, k, metric='cosine'):
    similarities = []
    target_user_ratings_dict = dict(zip(target_user['movie_id'], target_user['rating']))
    
    for user_id, other_user in other_users.groupby('user_id'):
        other_user_ratings_dict = dict(zip(other_user['movie_id'], other_user['rating']))
        
        # Find common movies
        common_movies = set(target_user_ratings_dict.keys()).intersection(other_user_ratings_dict.keys())
        
        if common_movies:
            # Extract ratings for common movies
            target_ratings = [target_user_ratings_dict[movie] for movie in common_movies]
            other_ratings = [other_user_ratings_dict[movie] for movie in common_movies]
            
            # Calculate similarity based on the selected metric
            if metric == 'cosine':
                sim = cosim(target_ratings, other_ratings)
            elif metric == 'euclidean':
                sim = euclidean(target_ratings, other_ratings)
            elif metric == 'pearson':
                sim = pearson_correlation(target_ratings, other_ratings)
            similarities.append((user_id, sim))
    
    # Sort by similarity and select top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

def recommend_movies(target_user_id, other_users, top_k_users, threshold=4):
    recommendations = {}
    
    # Get ratings for similar users
    for user_id, _ in top_k_users:
        similar_user_ratings = other_users[other_users['user_id'] == user_id]
        for _, row in similar_user_ratings.iterrows():
            movie_id, movie_name, rating = row['movie_id'], row['title'], row['rating']
            
            # Only recommend highly-rated movies
            if rating >= threshold:
                if movie_id not in recommendations:
                    recommendations[movie_id] = {'name': movie_name, 'score': 0}
                
                # Accumulate the score based on the ratings from similar users
                recommendations[movie_id]['score'] += rating
    
    # Sort recommendations by the accumulated score from top_k similar users
    recommended_movies = sorted(recommendations.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Return the list of movie names and scores
    return [(movie_data['name'], movie_data['score']) for _, movie_data in recommended_movies]


def main():


    # show('mnist_train.csv','pixels')
    # print(knn(read_data("mnist_train.csv"), read_data("mnist_valid.csv"), "euclidean"))
    # labels=kmeans(read_data("mnist_train.csv"), read_data("mnist_test.csv"), "euclidean")
    # print("K-means cluster assignments:", labels)

    # x = [1, 5,9, 4, 7]
    # y = [2,10,25,28, 20]
    # print(hammering_distance(x, y))
    # print(read_movie_data("train_a.txt"))
    # print(user_similarity(read_data("train_a.txt"), 405 ))
    # Load training, validation, and test data
    train_data = pd.read_csv('mnist_train.csv').values.tolist()  # Replace with actual file path
    test_data = pd.read_csv('mnist_test.csv').values.tolist()    # Replace with actual file path
    validate_data = pd.read_csv('mnist_valid.csv').values.tolist()  # Replace with actual file path

    evaluate_knn(train_data, test_data, metri="euclidean", K=5, n_comp=50)
    evaluate_knn(train_data, validate_data, metri="cosine", K=5, n_comp=50)

    confusion_matrix_euclidean = evaluate_knn(train_data, test_data, 'euclidean')
    confusion_matrix_cosine = evaluate_knn(train_data, test_data, 'cosine')
    # Load target user and other users
    target_user = train_b  # Replace 405 with desired user_id
    other_users = train_c
    # Get top K similar users
    top_k_users = get_top_k_similar_users(target_user, other_users, k=10, metric='cosine')
    
    # Get movie recommendations
    recommendations = recommend_movies(405, other_users, top_k_users)
    
    # Print recommendations
    # print("Recommended movies for user 405:")
    # for movie_name, score in recommendations:
    #     print(f"Movie name: {movie_name}, Score: {score}")
if __name__ == "__main__":
    main()
    