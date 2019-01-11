
# coding: utf-8

# In[ ]:


import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
import scipy
import sklearn
from scipy.sparse import csr_matrix


# In[ ]:


user_data = pd.read_table('H://datasets/music/lastfm/usersha1-artmbid-artname-plays.tsv',
                          header = None, nrows = 2e7,
                          names = ['users', 'musicbrainz-artist-id', 'artist-name', 'plays'],
                          usecols = ['users', 'artist-name', 'plays'])


# In[ ]:



user_profiles = pd.read_table('H://datasets/music/lastfm/usersha1-profile.tsv',
                          header = None,
                          names = ['users', 'gender', 'age', 'country', 'signup'],
                          usecols = ['users', 'country'])


# In[ ]:


user_data.head()


# In[ ]:


user_profiles.head()


# In[ ]:


if user_data['artist-name'].isnull().sum()>0:
    user_data=user_data.dropna(axis=0,subset=['artist-name'])


# In[ ]:


artist_plays = (user_data.
     groupby(by = ['artist-name'])['plays'].
     sum().
     reset_index().
     rename(columns = {'plays': 'total_artist_plays'})
     #[['artist-name', 'total_artist_plays']]
    )


# In[ ]:


artist_plays.head()


# In[ ]:


user_data_with_artist_plays = user_data.merge(artist_plays, left_on = 'artist-name', right_on = 'artist-name', how = 'left')


# In[ ]:


user_data_with_artist_plays.head()


# In[ ]:


artist_plays['total_artist_plays'].describe()


# In[ ]:


artist_plays['total_artist_plays'].quantile(np.arange(.9,1,0.01))


# In[ ]:


popularity_threshold =40000
user_data_popular_artists=user_data_with_artist_plays.query('total_artist_plays>=@popularity_threshold')
user_data_popular_artists.head()


# In[ ]:


combined = user_data_popular_artists.merge(user_profiles, left_on='users',right_on='users',how='left')
uk_data=combined.query('country == \'United Kingdom\'')
uk_data.head(10)


# In[ ]:


if not uk_data[uk_data.duplicated(['users', 'artist-name'])].empty:
    initial_rows = uk_data.shape[0]
    
    print('Initial dataframe shape {0}'.format(uk_data.shape))
    uk_data = uk_data.drop_duplicates(['users', 'artist-name'])
    current_rows = uk_data.shape[0]
    print('New dataframe shape {0}'.format(uk_data.shape))
    print('Removed {0} rows'.format(initial_rows - current_rows))


# In[ ]:


wide_artist_data = uk_data.pivot(index = 'artist-name', columns = 'users', values = 'plays').fillna(0)
wide_artist_data_sparse = csr_matrix(wide_artist_data.values)


# In[ ]:


wide_artist_data.head()


# In[ ]:


wide_artist_data_zero_one = wide_artist_data.apply(np.sign)
wide_artist_data_zero_one_sparse = csr_matrix(wide_artist_data_zero_one.values)

scipy.sparse.save_npz('H://datasets/music/lastfm/lastfm_sparse_artist_matrix_binary.npz', wide_artist_data_zero_one_sparse)


# In[ ]:


wide_artist_data_zero_one.head()


# In[ ]:


from sklearn.neighbors import NearestNeighbors

model_nn_binary = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_nn_binary.fit(wide_artist_data_zero_one_sparse)


# In[ ]:


query_index = np.random.choice(wide_artist_data.shape[0])
distances, indices = model_nn_binary.kneighbors(wide_artist_data_zero_one.iloc[query_index, :].reshape(1,-1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print ('Recommendations with binary play data for {0}:\n'.format(wide_artist_data_zero_one.index[query_index]))
    else:
        print ('{0}: {1}, with distance of {2}:'.format(i, wide_artist_data_zero_one.index[indices.flatten()[i]], distances.flatten()[i]))

