#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
"""
CS224N 2018-19: Homework 5
"""
class Highway(nn.Module):
    
### YOUR CODE HERE for part 1h
    def __init__(self,word_embed_size):
        
        super(Highway, self).__init__()
        self.e_word=word_embed_size
        self.projection = nn.Linear(self.e_word,self.e_word,bias=True)
        self.gate = nn.Linear(self.e_word,self.e_word,bias=True)
        
    def forward(self,x_conv_out: torch.Tensor)-> torch.Tensor:
        """
        @param x_conv_out (Tensor): Tensor of shape (batch_size,word_embed_size) 
        @param x_highway (Tensor):Tensor of shape (batch_size,word_embed_size)
        """
        x_projection= torch.relu_(self.projection(x_conv_out))   
        x_gate= torch.sigmoid(self.gate(x_conv_out))        
        
        x_highway=x_gate * x_projection + (1 - x_gate) * x_conv_out
        
        return x_highway
### END YOUR CODE 

