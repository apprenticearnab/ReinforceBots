'''
This file contains the model for the RL based chatbot
'''

import numpy as np
import sys
import time
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
import torch
from torchsummary import summary
import torch.nn as nn
import gensim
import load_embeddings



'''
Aggregating Word Vectors into a document vector
'''


# word_vector_size --> embedding dimension (300,1)
# sentence_size --> maximum number of words in a sentence  ( shorter sentences padded with dummy words )
# document_size --> maximum number of sentences in a document  ( shorter documents padded with dummy sentences )



class aggregate_doc(nn.Module):
    def __init__(self,sentence_size,conv_filter_dim = 200,word_vector_size=300):
        super(aggregate_word2sentence,self).__init__()

        self.embed_dim = word_vector_size
        self.conv_dim = conv_filter_dim
        self.sentence_size  = sentence_size
        
    # channel 1  n_gram = 1
        self.conv1 = nn.Conv1d(300,200,1)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool1d(self.sentence_size-1+1)
        self.flat1 = nn.Flatten()
        
    # channel 2  n_gram = 2
        self.conv2 = nn.Conv1d(300,200,2)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool1d(self.sentence_size-2+1)
        self.flat2 = nn.Flatten()
        
    # channel 3  n_gram = 3
        self.conv3 = nn.Conv1d(300,200,3)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool1d(self.sentence_size-3+1)
        self.flat3 = nn.Flatten()

    # Classification layer
    	self.lin1  = nn.Linear(684,200)
    	self.lin2  = nn.Linear(200,2)
    	self.relu4 = nn.ReLU()
    	self.sig1  = nn.Sigmoid()
      
      
    def forward(self, x):                               #aggregates word vectors to a sentence vector

        assert x.shape[1] == self.embed_dim
        assert x.shape[2] == self.sentence_size
        
        x_e = x
        x_c1 = self.conv1(x_e)
        x_c2 = self.conv2(x_e)
        x_c3 = self.conv3(x_e)
          
        x_c1 = self.relu1(x_c1)
        x_c2 = self.relu2(x_c2)
        x_c3 = self.relu3(x_c3)
      
        x_m1 = self.max_pool1(x_c1)
        x_m2 = self.max_pool2(x_c2)
        x_m3 = self.max_pool3(x_c3)
          
        f1 = self.flat1(x_m1)
        f2 = self.flat2(x_m2)
        f3 = self.flat3(x_m3)

          
        output= torch.cat((f1, f2, f3), dim=1)

        return output
    
    def aggregate_sentence2doc(self,words_aggregated):  #aggregates sentence vectors to a document vector
        
        assert words_aggregated.shape[1] == 600
        
        return torch.max(words_aggregated,0).values
        
        
        
    def aggregate_mairesse2doc(self,doc_agg,mairess_feat): # Aggregates the mairesse features with the document vector

    	assert doc_agg.shape[0] == 600
    	assert mairess_feat.shape[0] == 84

    	return torch.cat(doc_agg,mairess_feat, dim=0)

    def classification(self,doc_vec): # Final classification network

    	assert doc_vec.shape[0] == 684

    	out = self.lin1(doc_vec)
    	out = self.relu4(out)
    	out = self.lin2(out)
    	out = self.sig1(out)

    	return out


        
class agent():

	def __init__(self,input_enc,hidden_enc, layer_enc, output_enc,
		         input_dec,hidden_dec, layer_dec, output_dec, debug=False):
		super(agent,self).__init__()

		self.input_enc  = input_enc
		self.hidden_enc = hidden_enc
		self.layer_enc  = layer_enc
		self.output_enc = output_enc
		self.input_dec  = input_dec
		self.hidden_dec = hidden_dec
		self.layer_dec  = layer_dec
		self.output_dec = output_dec

		
	def lstm_encoder(nn.Module):
	        
        # Building the LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm_enc = nn.LSTM(self.input_enc, self.hidden_enc, self.layer_enc, batch_first=True)
        
        # Readout layer
        self.fc_enc = nn.Linear(self.hidden_enc, self.output_enc)
	    
    def forward_enc(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################

        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        # Initialize cell state
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        # One time step
        out, (hn, cn) = self.lstm_enc(x, (h0,c0))
        
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc_enc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
		

	def lstm_decoder(nn.Module):
	        
	        # Building your LSTM
	        # batch_first=True causes input/output tensors to be of shape
	        # (batch_dim, seq_dim, feature_dim)
	        self.lstm_dec = nn.LSTM(self.input_dec, self.hidden_dec, self.layer_dec, batch_first=True)
	        
	        # Readout layer
	        self.fc_dec = nn.Linear(self.hidden_dec, self.output_dec)
	    
	def forward_dec(self, x, seq):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################

        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        # Initialize cell state
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        # One time step
        out, (hn, cn) = self.lstm(x, (h0,c0))
        
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        for i in range(seq):
        	out = torch.cat(self.fc(out[:, i, :]),dim=0) 
        # out.size() --> 100, 10
        return out



    
    
    
    

