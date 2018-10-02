import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# import torch
import pickle
from sv_system.data.dataloader import init_default_loader
from sv_system.data.data_utils import find_dataset
from sv_system.utils.parser import score_parser, set_score_config
from sv_system.eval.score_utils import embeds_utterance
from sv_system.train.train_utils import load_checkpoint, get_dir_path

#########################################
# Parser
#########################################
parser = score_parser()
args = parser.parse_args()
config = set_score_config(args)
if config['output_folder'] is None:
    output_folder = get_dir_path(config['input_file'])
else:
    output_folder = config["output_folder"]

#########################################
# Model Initialization
#########################################
# set data_folder, input_dim, n_labels, and dset
dfs, datasets = find_dataset(config, split=True)
model, _ = load_checkpoint(config)
lda = None
# lda = pickle.load(open("lda_model.pkl", "rb"))

#########################################
# Compute Embeddings
#########################################
si_train_dset = datasets[0]
labels = np.array(dfs[0].label.tolist())
val_dataloader = init_default_loader(config, si_train_dset, shuffle=False)
embeddings, _ = embeds_utterance(config, val_dataloader, model, lda)

n_samples = embeddings.shape[0]
print(embeddings.shape)
n_test = 500 # for test samples
random_idx = np.random.permutation(np.arange(0,n_samples))
train_X, train_y = embeddings[random_idx[:n_samples-n_test]], \
    labels[random_idx[:n_samples-n_test]]
test_X, test_y = embeddings[random_idx[-n_test:]], \
    labels[random_idx[-n_test:]]

lda_model = LDA()
lda_model.fit(train_X, train_y)
score = lda_model.score(test_X, test_y)
print(score) # test_score

pickle.dump(lda_model, open(os.path.join(output_folder,"lda_model.pkl"), "wb"))

# class LDAModel():
    # def __init__(self):
        # self.lda = LDA()

    # def fit(self, embeddings, labels):
        # n_samples = embeddings.shape[0]
        # n_test = 500 # for test samples
        # random_idx = np.random.permutation(np.arange(0,n_samples))
        # train_X, train_y = embeddings[random_idx[:n_samples-n_test]], \
            # labels[random_idx[:n_samples-n_test]]
        # test_X, test_y = embeddings[random_idx[-n_test:]], \
            # labels[random_idx[-n_test:]]
        # self.lda.fit(train_X, train_y)
        # score = self.lda.score(test_X, test_y)
        # print(score) # test_score


# LDA
# Reddots LDA
# reddots_files = pd.read_pickle("../dataset/dataframes/reddots/Reddots_Dataframe.pkl").file
# used_files = pd.concat([trn.file, ndx_file.file])
# unused_files = reddots_files[~reddots_files.isin(used_files)]

# lda_file = pd.DataFrame(unused_files, columns=['file'])
# lda_dataset = SpeechDataset.read_df(si_config, lda_file, "test")
# val_dataloader = init_default_loader(si_config, lda_dataset, shuffle=False)
# lda_embeddings, _ = embeds_utterance(si_config, val_dataloader, model, lda)

