# coding: utf-8
import pandas as pd
import pickle
import sys

sys.path.append("/home/muncok/DL/projects/")


# In[3]:


from tqdm import tqdm_notebook as tqdm
import torch.nn.functional as F
from sv_system.data.dataloader import init_default_loader
from sv_system.utils.parser import get_sv_parser
from sv_system.train.si_train import set_seed
from sv_system.data.dataset import SpeechDataset


# In[4]:


from sv_system.utils import secToFrames, secToSample
from sv_system.utils.parser import test_config
si_config = test_config('tdnn')
si_config['input_clip'] = True
si_config['input_length'] = secToSample(3)
si_config['input_frames'] = secToFrames(3)
si_config['splice_frames'] = secToFrames(0.1)
si_config['input_format'] = 'fbank'
si_config['data_folder'] = "/home/muncok/DL/dataset/SV_sets/reddots_r2015q4_v1/wav/"
# si_config['data_folder'] = "/home/muncok/DL/projects/sv_experiments/vad/reddots_vad_cut_1sec/"


# In[5]:


import torch
from torch.autograd import Variable
from tqdm import tqdm_notebook
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def lda_on_tensor(tensor, lda):
    return torch.from_numpy(lda.transform(tensor.numpy()).astype(np.float32))

def embeds_utterance(opt, val_dataloader, model, lda=None):
    val_iter = iter(val_dataloader)
    model.eval()
    splice_dim = opt['splice_frames']
    embeddings = []
    labels = []
    for batch in tqdm_notebook(val_iter, total=len(val_iter)):
        x, y = batch
        time_dim = x.size(2)
        split_points = range(0, time_dim-(splice_dim+12), splice_dim)
        model_outputs = []
        for point in split_points:
            x_in = Variable(x.narrow(2, point, splice_dim+12))
            if not opt['no_cuda']:
                x_in = x_in.cuda()
            model_outputs.append(model.embed(x_in).cpu().data)
        model_output = torch.stack(model_outputs, dim=0)
        model_output = model_output.mean(0)
        if lda is not None:
            model_output = torch.from_numpy(lda.transform(model_output.numpy()).astype(np.float32))
        embeddings.append(model_output)
        labels.append(y.numpy())
    embeddings = torch.cat(embeddings)
    labels = np.hstack(labels)
    return embeddings, labels


# In[6]:


def embeds_one(opt, val_dataloader, model, lda=None):
    val_iter = iter(val_dataloader)
    model.eval()
    model
    embeddings = []
    labels = []
    for batch in tqdm_notebook(val_iter, total=len(val_iter)):
        x, y = batch
        if not opt['no_cuda']:
            x = x.cuda()
        model_output = model(x)
        embeddings.append(model_output.cpu().detach())
        if lda is not None:
            model_output = torch.from_numpy(lda.transform(model_output.numpy()).astype(np.float32))
        labels.append(y.numpy())
    embeddings = torch.cat(embeddings)
    labels = np.hstack(labels)
    return embeddings, labels


# ## SI_Model

# ### TDNN

# In[26]:


from sv_system.model.TDNN import TdnnModel
model = TdnnModel(si_config, 70, embed_mode=True)
# model.load("../models/compare_train_methods/voxc/si_voxc_TdnnModel_3s_0.1s_fbank_full_frame.pt")
model.load("../models/compare_train_methods/reddots/si_reddots_TdnnModel_3s_0.1s_fbank_full_frame.pt")
model.cuda()
# lda = pickle.load(open("models/lda/si_reddots_0.2s_random_2_lda.pkl", "rb"))
lda = None


# ### SimpleCNN

# In[ ]:


from sv_system.model.AuxModels import SimpleCNN
model = SimpleCNN(si_config, 70)
model.load("../models/compare_train_methods/reddots/si_reddots_TdnnModel_3s_0.1s_fbank_full_frame.pt")
model.cuda()
# lda = pickle.load(open("models/lda/si_reddots_0.2s_random_2_lda.pkl", "rb"))
lda = None


# ### SpeechModel

# In[129]:


from sv_system.model.SpeechModel import SpeechResModel, SpeechModel
model = SpeechResModel("res15", 1260)
model.load("../models/voxc/si_train/full_train/si_voxc_res15_0.1s_full_fbank.pt")
model.cuda()
# lda = pickle.load(open("models/lda/si_reddots_0.2s_random_2_lda.pkl", "rb"))
lda = None


# ##  Reddots Trial

# In[9]:


# ndx = pd.read_pickle("../dataset/dataframes/reddots/m_part1/m_part1_ndx.pkl")
# trn = pd.read_pickle("../dataset/dataframes/reddots/m_part1/m_part1_trn.pkl")
# cord = pickle.load(open("../dataset/dataframes/reddots/m_part1/ndx_idxs.pkl", "rb"))

ndx = pd.read_pickle("../dataset/dataframes/reddots//m_part4_tp/m_part4_tp_ndx.pkl")
trn = pd.read_pickle("../dataset/dataframes/reddots//m_part4_tp/m_part4_tp_trn.pkl")
cord = pickle.load(open("../dataset/dataframes/reddots/m_part4_tp//ndx_idxs.pkl", "rb"))


# In[10]:


# x_cord = []
# y_cord = []
# ndx_file =pd.DataFrame(ndx.file.unique().tolist(), columns=['file'])
# all_trials = trn.id.unique().tolist()
# for trial_id in tqdm(all_trials):
#     trial_ndx = ndx[(ndx.id == trial_id)].reset_index()
#     trial_embed_idx = np.nonzero(ndx_file.file.isin(trial_ndx.file))[0].tolist()
#     x_cord += [all_trials.index(trial_id)] * len(trial_embed_idx)
#     y_cord += trial_embed_idx

# cord = [x_cord, y_cord]
# pickle.dump(cord, open("../dataset/dataframes/reddots/m_part1/ndx_idxs.pkl", "wb"))


# In[11]:


err_type = {0:'TC', 1:'TW', 2:'IC', 3:'IW'}
si_config['batch_size'] = 64
si_config['num_workers'] = 32


# ###  Enrollment (trn)

# In[12]:


si_config['data_folder'] = "/home/muncok/DL/dataset/SV_sets/reddots_r2015q4_v1/wav/"
trn_dataset = SpeechDataset.read_df(si_config, trn, "test")

val_dataloader = init_default_loader(si_config, trn_dataset, shuffle=False)
trn_embeddings, _ = embeds_utterance(si_config, val_dataloader, model, lda)
embed_dim = trn_embeddings.shape[-1]
trn_id = list(trn.id.unique())
spk_model_dict = {}
for id in trn_id:
    index = np.nonzero(trn.id == id)
    spk_model_dict[id] = trn_embeddings[index].mean(0, True)

spk_models = torch.cat([emb for emb in spk_model_dict.values()])


# ###  SV Scoring (ndx)

# In[14]:


ndx_file = pd.DataFrame(ndx.file.unique().tolist(), columns=['file'])


# In[91]:


ndx_dataset = SpeechDataset.read_df(si_config, ndx_file, "test")
val_dataloader = init_default_loader(si_config, ndx_dataset, shuffle=False) 
ndx_embeddings, _ = embeds_utterance(si_config, val_dataloader, model, lda)


# In[92]:


sim_matrix = F.cosine_similarity(spk_models.unsqueeze(1), ndx_embeddings.unsqueeze(0), dim=2)
sims = sim_matrix[cord]

scores = dict()
for t in range(4):
    trial_type_idx = ndx[ndx.trial_type == t].index
    scores[t] = sims[trial_type_idx]

for t in range(4):
     print("{} mean:{:.2f}, std:{:.3f}".format(err_type[t], scores[t].mean(), scores[t].std()))


# TD EERs

# In[93]:


from sklearn.metrics import roc_curve


for t in range(1,4):
    score_vector = np.concatenate((scores[0], scores[t]))
    label_vector = np.concatenate((np.ones(len(scores[0])), 
                               np.zeros(len(scores[t]))))
    fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    print("[{}] eer: {:.2f}, thres: {:.5f}".format(err_type[t], eer, thres))


# TI EERs

# In[94]:


from sklearn.metrics import roc_curve

score_vector = np.concatenate((scores[0], scores[1],
                              scores[2], scores[3]))
label_vector = np.concatenate((np.ones(len(scores[0]) + len(scores[1])), 
                           np.zeros(len(scores[2]) + len(scores[3]))))
fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
print("[TI] eer: {:.2f}".format(eer))


# ## LDA

# ### Reddots LDA

# In[27]:


reddots_files = pd.read_pickle("../dataset/dataframes/reddots/Reddots_Dataframe.pkl").file
used_files = pd.concat([trn.file, ndx_file.file])
unused_files = reddots_files[~reddots_files.isin(used_files)]

lda_file = pd.DataFrame(unused_files, columns=['file'])
lda_dataset = SpeechDataset.read_df(si_config, lda_file, "test")
val_dataloader = init_default_loader(si_config, lda_dataset, shuffle=False) 
lda_embeddings, _ = embeds_utterance(si_config, val_dataloader, model, lda)


# In[ ]:


n_test = 100 # for test samples
embeddings = lda_embeddings


# In[29]:


spks =unused_files.apply(lambda x: x.split('_')[1])  


# In[30]:


all_spks = spks.unique().tolist()


# In[31]:


labels = np.array([all_spks.index(label) for label in spks])


# In[32]:


n_test = 100 # for test samples
embeddings = lda_embeddings


# In[33]:


n_samples = embeddings.shape[0]
clf = LDA()
random_idx = np.random.permutation(np.arange(0,n_samples))
train_X, train_y = embeddings[random_idx[:n_samples-n_test]], labels[random_idx[:n_samples-n_test]]
test_X, test_y = embeddings[random_idx[-n_test:]], labels[random_idx[-n_test:]]
clf.fit(train_X, train_y)


# In[34]:


score = clf.score(test_X, test_y)
print(score) # test_score


# In[35]:


lda = clf


# In[36]:


pickle.dump(clf, open("../models/compare_train_methods/reddots/si_reddots_TdnnModel_3s_0.1s_fbank_full_frame.lda",
                      "wb"))


# ### Voxc LDA

# In[58]:


df = pd.read_pickle("../dataset/dataframes/Voxc_Dataframe.pkl")


# In[59]:


train_voxc = df[df.set == 'train']


# In[76]:


lda_dataset = train_voxc.groupby(['spk']).apply(lambda x: x.sample(n=20))


# In[77]:


unused_files = lda_dataset.file
labels = lda_dataset.label
si_config['data_folder'] = "/home/muncok/DL/dataset/SV_sets/voxceleb/"

lda_file = pd.DataFrame(unused_files, columns=['file'])
lda_dataset = SpeechDataset.read_df(si_config, lda_file, "test")
val_dataloader = init_default_loader(si_config, lda_dataset, shuffle=False) 
lda_embeddings, _ = embeds_utterance(si_config, val_dataloader, model, lda)


# In[78]:


n_test = 100 # for test samples
embeddings = lda_embeddings


# In[79]:


n_samples = embeddings.shape[0]
clf = LDA()
random_idx = np.random.permutation(np.arange(0,n_samples))
train_X, train_y = embeddings[random_idx[:n_samples-n_test]], labels[random_idx[:n_samples-n_test]]
test_X, test_y = embeddings[random_idx[-n_test:]], labels[random_idx[-n_test:]]
clf.fit(train_X, train_y)


# In[80]:


score = clf.score(test_X, test_y)
print(score) # test_score


# In[81]:


lda = clf


# In[117]:


lda_out = "models/lda/{}_splice_lda.pkl".format(options.input.split('/')[-1][:-3])
pickle.dump(clf, open(lda_out, "wb"))

