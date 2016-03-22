# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 23:36:18 2016

@author: yiyuezhuo
"""

from __future__ import print_function

import os
import numpy as np

import jieba
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import random
import sys

#from keras.utils.data_utils import get_file

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))
    
def fancy(s,utf8_to_char):
    word_l=[word.encode('utf8') for word in jieba.cut(unicode(s,'utf8'))]
    sl=[ ''.join(([utf8_to_char[c] for c in word])) for word in word_l]
    return ' '.join(sl)
    
def fancy_test(s,ds,diff=49):
    cl=sorted(list(ds.chars))
    utf8_l=cl[84:]
    utf8_to_char={code:chr(index+diff) for index,code in enumerate(utf8_l)}
    return fancy(s,utf8_to_char)


class Extern_Monitor(object):
    def __init__(self,fname='unicode_model',command_path='command.txt',log_path='log.txt'):
        self.command_path=command_path
        self.log_path=log_path
        self.fname=fname
    def save(self,model):
        json_string = model.to_json()
        open(self.fname+'.json', 'w').write(json_string)
        model.save_weights(self.fname+'.h5')
    def check(self,model):
        f=open(self.command_path,'r')
        s=f.read()
        f.close()
        if 'pause' in s:
            self.save(model)
            self.log('pause')
            raise StopIteration
        elif 'save' in s:
            self.save(model)
            self.log('saving')
    def log(self,*argv):
        f=open(self.log_path,'a')
        print(*argv)
        for s in argv:
            f.write(str(s))
        f.write('\n')
        f.close()
        
class DataStream(object):
    def __init__(self,init_index=0,fetch_mode='random',corpus_size=100000,maxlen=20,step=3,text_path=None,
                 dir_path=None,delim='\n',load=True,cut=True):
        self.maxlen=maxlen
        self.step=step
        self.index=init_index
        self.corpus_size=corpus_size
        self.fetch_mode=fetch_mode
        
        if text_path:
            f=open(text_path,'r')
            self.text=f.read()
            f.close()
        elif dir_path:
            tl=[]
            for name in os.listdir(dir_path):
                f=open(os.path.join(dir_path,name),'r')
                tl.append(f.read())
                f.close()
                self.text=delim.join(tl)
        else:
            raise "can't bind"
        text_l=[]
        for line in self.text.split('\n'):
            line=line.strip()
            if len(line)>0:
                text_l.append(line)
        self.text='\n'.join(text_l)
        if cut:
            self.cut()
        self.chars=set(self.text)
        self.length=len(self.text)
        
        print('corpus length:', self.length)
        print('total chars:', len(self.chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def cut(self,encoding='utf8',delim=' '):
        cut_ite=jieba.cut(unicode(self.text,encoding=encoding,errors='ignore'))
        self.text=delim.join(list(cut_ite))
        self.text=self.text.encode(encoding)
    def next_text(self):
        if self.fetch_mode=='scroll':
            index=self.index
            if self.index+self.corpus_size>=self.length:
                self.index=0
            else:
                self.index+=self.corpus_size
        elif self.fetch_mode=='random':
            index=random.randint(0, self.length - self.corpus_size - 1)
        rt=self.text[index:index+self.corpus_size]
        print('new_text:'+rt[:50])
        return rt
    def fetch(self):
        # cut the text in semi-redundant sequences of maxlen characters
        #text=self.text
        text=self.next_text()
        chars=self.chars
        maxlen=self.maxlen
        step=self.step
        
        maxlen = 20
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        print('nb sequences:', len(sentences))
        
        print('Vectorization...')
        X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1
        return text,X,y

            
            

class Model(object):
    def __init__(self):
        self.model=None
    def build(self,text_data):
        chars=text_data.chars
        maxlen=text_data.maxlen
        # build the model: 2 stacked LSTM
        print('Build model...')
        model = Sequential()
        model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
        model.add(Dropout(0.2))
        model.add(LSTM(512, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(len(chars)))
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        self.model=model
    def load(self,fname):
        model = model_from_json(open(fname+'.json').read())
        model.load_weights(fname+'.h5')
        self.model=model
    def save(self,fname):
        json_string = self.model.to_json()
        open(fname+'.json', 'w').write(json_string)
        self.model.save_weights(fname+'.h5')
    def sample(self,ds,em,text=None,diversity_t=(0.2, 0.5, 1.0),test_len=400):
        maxlen=ds.maxlen
        if text==None:
            text=ds.next_text()

        start_index = random.randint(0, len(text) - maxlen - 1)
        
        for diversity in diversity_t:
            em.log()
            em.log('----- diversity:', diversity)
    
            sentence = text[start_index: start_index + maxlen]
            em.log('----- Generating with seed: "' + sentence + '"')
            
            generated=self.generate(sentence,ds,diversity=diversity,test_len=test_len)
                
            em.log()
            em.log("gemerated")
            em.log(generated)
            em.log()
    def generate(self,sentence,ds,diversity=0.5,test_len=400):
        model=self.model        
        indices_char=ds.indices_char
        char_indices=ds.char_indices
        maxlen=ds.maxlen
        chars=ds.chars
        
        generated = sentence
        sys.stdout.write(generated)

        for i in range(test_len):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        return generated
    
    def train(self,ds,em,roll=5,batch_size=128,nb_epoch=1,maxiter=30,
              diversity_t=(0.2, 0.5, 1.0),test_len=400):
        model=self.model
        
        for ii in range(maxiter):
            text,X,y=ds.fetch()
            
            for iteration in range(roll):
                em.log()
                em.log('-' * 50)
                em.log('Iteration', iteration)
                
                model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch)
                
                self.sample(ds,em,text=text,diversity_t=diversity_t,test_len=test_len)
                em.check(model)

em=Extern_Monitor(fname='shana_model')
ds=DataStream(text_path=u'灼眼的夏娜.txt')
model=Model()
'''
model.build(ds)
model.load(em.fname)

model.sample(ds,em)
model.train(ds,em)
'''