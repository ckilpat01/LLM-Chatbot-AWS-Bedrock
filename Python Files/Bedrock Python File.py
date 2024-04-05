#!/usr/bin/env python
# coding: utf-8

# # LLM On AWS Bedrock Using LangChain

# In[ ]:


# Import Libraries

import os
import boto3
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# ### Function of Libraries
# 
# * Bedrock allows creation of objects with details about what FM to use, and configure model parameters, authentication, etc.
# * PromptTemplate enables creation of prompts to ingest variables into.
# * ConversationBufferMemory enables control of history or memory.
# * Conversation Chain Ties all these objects together into a chain.

# ### Create Chain Function

# In[ ]:


# Define bedrock_chain function to create bedrock object
def bedrock_chain():
    profile = os.environ["AWS_PROFILE"]

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )

    titan_llm = Bedrock(
        model_id="amazon.titan-text-express-v1", client=bedrock_runtime, credentials_profile_name=profile
    )
    titan_llm.model_kwargs = {"temperature": 0.5, "maxTokenCount": 700}
    
# Construct three part prompt template:  1. The instruction.  2. The context (history). 3. User query (input).    
    prompt_template = """System: The following is a friendly conversation between a knowledgeable helpful assistant and a customer.
    The assistant is talkative and provides lots of specific details from it's context.

    Current conversation:
    {history}

    User: {input}
    Bot:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=prompt_template
    )

# Configure the memory attribute    
    memory = ConversationBufferMemory(human_prefix="User", ai_prefix="Bot")
    
# Combine all together
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=titan_llm,
        verbose=True,
        memory=memory,
    )

    return conversation


# In[ ]:


# Function which is executed when chain is called from the Streamlit app
def run_chain(chain, prompt):
    num_tokens = chain.llm.get_num_tokens(prompt)
    return chain({"input": prompt}), num_tokens


# Clear memory
def clear_memory(chain):
    return chain.memory.clear()


# In[ ]:




