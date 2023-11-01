# Copied from CHAR folder, modified to have extra tokens with better tokenization

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

char2idx = {
    '<pad>': 0,
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5,
    'f': 6,
    'g': 7,
    'h': 8,
    'i': 9,
    'j': 10,
    'k': 11,
    'l': 12,
    'm': 13,
    'n': 14,
    'o': 15,
    'p': 16,
    'q': 17,
    'r': 18,
    's': 19,
    't': 20,
    'u': 21,
    'v': 22,
    'w': 23,
    'x': 24,
    'y': 25,
    'z': 26,
    '<sow>': 27,
    '<eow>': 28,
    '<sos>': 29,
    '<eos>': 30,
    '0': 31,
    '1': 32,
    '2': 33,
    '3': 34,
    '4': 35,
    '5': 36,
    '6': 37,
    '7': 38,
    '8': 39,
    '9': 40,
    '_': 41, # fill in the blanks types of questions
}

idx2char = {v:k for k,v in char2idx.items()}

def convert_to_idx(question, char2idx):
    """
    Converts sentences to character tokens
    """
    all_tokens = []
    
    # Split words
    #words = question.lower().split()
    words = tokenizer.tokenize(question.lower())
    
    all_tokens.append(char2idx['<sos>']) # Start of sentence
    
    for w in words:
        all_tokens.append(char2idx['<sow>']) # Start of word
        # Split in characters
        for c in w:
            if c not in char2idx.keys():
                print('ERROR IN QUESTION SENTENCE:')
                print(question)
            all_tokens.append(char2idx[c])
        all_tokens.append(char2idx['<eow>']) # End of word
    
    all_tokens.append(char2idx['<eos>']) # End of sentence
        
    return all_tokens


def convert_to_char(tokens, idx2char):
    """
    Converts tokens into sentence
    """
    
    word = ''
    sentence = ''
    
    for idx in tokens:
        token = idx2char[idx]
        
        if token == '<eos>' or token == '<sos>' or token == '<sow>':
            continue
        
        if token == '<eow>':
            sentence += (word + ' ') 
            word = ''
        else:
            char = token
            #print(char, end='')
            word += char
            
    return sentence


def print_each_character(tokens, idx2char):
    """
    Prints tokens one by one (as characters)
    """
    
    for t in tokens:
        print(idx2char[t], end=' ')
        
    print()