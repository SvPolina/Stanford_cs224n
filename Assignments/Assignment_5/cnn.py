#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
from torch import nn

### YOUR CODE HERE for part 1i
class CNN(nn.Module):    

    def __init__(self,char_embed_size, num_filters,max_word_length,kernel_size=5):
        
        super(CNN, self).__init__()
        self.c_word=char_embed_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.max_word_length = max_word_length
        self.conv_layer=nn.Conv1d(in_channels=self.c_word,out_channels=self.num_filters,kernel_size=self.kernel_size, bias=True)
        self.max_pool = nn.MaxPool1d(self.max_word_length - self.kernel_size + 1)
        
    def forward(self,x_reshaped: torch.Tensor)-> torch.Tensor:
        """
        @param x_reshaped (Tensor): Tensor of shape (batch_size,char_embed_size,max_word_length) 
        @param x_conv_out (Tensor):Tensor of shape (batch_size,char_embed_size)      
        """
        
        x_conv=self.conv_layer(x_reshaped)
        x_conv_out=self.max_pool(torch.relu_(x_conv)).squeeze() 
        
        return x_conv_out
### END YOUR CODE

