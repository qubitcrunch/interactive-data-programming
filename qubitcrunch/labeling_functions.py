from snorkel.labeling import labeling_function
from torchtext.data.utils import get_tokenizer
import pandas as pd

@labeling_function()
def lf_contains_world_term(x):
    # Return a label of 0 if the document contains a sports term
    
    if type(x) == pd.core.series.Series:
        x = x.body
    if type(x) != str:
        x = str(x)
    
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(x)
    
    sports_terms = ["israel"]
    
    for token in tokens:
        if token in sports_terms:
            return 0
        else:
            return -1
        
@labeling_function()
def lf_contains_sports_term(x):
    # Return a label of 1 if the document contains a finance term
    
    if type(x) == pd.core.series.Series:
        x = x.body
    if type(x) != str:
        x = str(x)
    
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(x)
    
    finance_terms = ["baseball"]
    
    for token in tokens:
        if token in finance_terms:
            return 1
        else:
            return -1
        
@labeling_function()
def lf_contains_business_term(x):
    # Return a label of 2 if the document contains a entertainment term
    
    if type(x) == pd.core.series.Series:
        x = x.body
    if type(x) != str:
        x = str(x)
    
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(x)
    
    entertainment_terms = ["money"]
    
    for token in tokens:
        if token in entertainment_terms:
            return 2
        else:
            return -1

@labeling_function()
def lf_contains_science_term(x):
    # Return a label of 3 if the document contains a automobile term
    
    if type(x) == pd.core.series.Series:
        x = x.body
    if type(x) != str:
        x = str(x)
    
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(x)
    
    automobile_terms = ["astronomy"]
    for token in tokens:
        if token in automobile_terms:
            return 3
        else:
            pass
        return -1