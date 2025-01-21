import io
import os
import unicodedata
import string
import glob

import torch 
import random

# Alphabet: Small + Capital letters + "'.,;"
All_Letters=string.ascii_letters + ".,;'"
N_Letters=len(All_Letters)

# Turn a unicode string to a plain ASCII
def Unicode_to_ASCII(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD' ,s)
        if unicodedata.category(c)!='Mn'
        and c in All_Letters
    )

def load_data():
    # Build the category_lines dictionary, a list of names per language
    category_lines={}
    all_categories=[]
    
    def find_files(path):
        return glob.glob(path)
    
    # read a file and Split into Lines
    def read_files_into_lines(filename):
        lines=io.open(filename,encoding='utf-8').read().strip().split('\n')
        return [Unicode_to_ASCII(line) for line in lines]
    
    for filename in find_files('data/names/*.txt'):
        category=os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        
        lines=read_files_into_lines(filename)
        category_lines[category]=lines
        
    return category_lines,all_categories

'''
To represent a single letter, we use a "one-hot vector"
of size <1, number of letters>, where the one-hot vector is filled 
with 0s  except for the index of the current letter, eg "b"=<0 1 0 0 0...>.

To make a a word, we join a bunch of those into a 2D matrix
of size <line length, 1, number of letters>.

That extra 1 dimension is because pytorch assumes everythin is in batches,
and we are just using a batch size of 1 here.
'''

# Find letter index from All_Letters , eg "a"=0
def letter_to_index(letter):
    return All_Letters.find(letter)


# Just for demonstration, turn a letter into a <1, number of letters> Tensor
def letter_to_tensor(letter):
    tensor=torch.zeros(1,N_Letters)
    tensor[0][letter_to_index(letter)]=1
    return tensor

# Turn a line into a <line length, 1, number of letters>,
# or any array of one-hot letters vectors
def line_to_tensor(line):
    tensor=torch.zeros(len(line),1,N_Letters)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)]=1
    return tensor

def random_training_example(category_lines,all_categories):
    
    def random_choice(a):
        random_idx=random.randint(0,len(a)-1)
        return a[random_idx]
    
    category=random_choice(all_categories)
    line=random_choice(category_lines[category])
    category_tensor=torch.tensor([all_categories.index(category)],dtype=torch.long)
    line_tensor=line_to_tensor(line)
    return category,line,category_tensor,line_tensor

if __name__=='__main__':
    print(All_Letters)
    print(Unicode_to_ASCII('#Slusarski'))
    
    category_lines, all_categories=load_data()
    print(category_lines['Italian'][:5])
    
    print(letter_to_tensor('J'))#[1,57]
    print(line_to_tensor('Jones').size())#[5,1,56]