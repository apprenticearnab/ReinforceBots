'''
This file loads a word2vec of user's choice

For this project :

Word2Vec model : 'word2vec-google-news-300'
'''

import torch
import gensim.downloader


# Downloads Google news word2vec model
def load_embeddings(word2vec_model='word2vec-google-news-300'):
    
    embed_model = gensim.downloader.load(word2vec_model)
    
    return embed_model
    

# Transform sentence using word2vec
def generate_word2vec_tensor(x,embed_model):
    return torch.Tensor(embed_model[x])

# Prepare batch dataset : returns tensor of dimension --> [batch_size,embedding_dimension,sentence_size]
def sentence2embeddings(sentences,embed_model,sentence_size):

    sentence_batch = []
    assert len(sentences[0]) == sentence_size
    
    for sentence in sentences:
        sentence_vector = torch.transpose(generate_word2vec_tensor(sentence,embed_model),0,1)
        sentence_vector = sentence_vector.reshape((1,sentence_vector.shape[0],sentence_vector.shape[1]))
        sentence_batch.append(sentence_vector)
    
    sentence_batch = torch.cat(sentence_batch,dim=0)

    return torch.Tensor(sentence_batch)
