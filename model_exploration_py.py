
# coding: utf-8

# In[1]:


import os


# In[2]:

from sv_system.dnn.utils.parser import test_config
from sv_system.dnn.mo import
import sv_system.dnn.train.si_train as si_train
import sv_system.dnn.data.dataloader as dloader
from sv_system.dnn.data.dataset import SpeechDataset
import pandas as pd


# In[3]:


model = "SimpleCNN"
si_config = test_config(model)


# In[4]:


from dnn.utils import secToSample, secToFrames
si_config['splice_frames'] = secToFrames(0.1)
si_config['input_length'] = secToSample(1)
si_config['input_clip'] = True
si_config['input_format'] = 'fbank'


# In[5]:


si_model = SimpleCNN(1880, si_config['splice_frames'])
si_model.load("../interspeech2018/models/commands/equal_num_102spk_dot1.pt")


# In[6]:


si_model


# In[7]:


si_train.set_seed(si_config)
si_config['n_epochs'] = 80
si_config['output_file'] = "models/commands/test.pt"


# In[8]:


si_config['data_folder'] = "/home/muncok/DL/dataset/SV_sets/speech_commands/"


# In[9]:


iden_df = pd.read_pickle("trials/commands/final/equal_num_102spk_iden.pkl")
iden_df.file = iden_df.apply(lambda row: os.path.join(row.sent, row.file), axis=1)


# In[10]:


samples = SpeechDataset.read_df(si_config['data_folder'], iden_df)
dataset = SpeechDataset(samples,"test", si_config)


# In[12]:


loader = dloader.init_default_loader(dataset)
si_train.si_train(si_config, model=si_model, loaders=[loader, loader, loader])

