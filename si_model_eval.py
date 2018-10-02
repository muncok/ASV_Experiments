# coding: utf-8
from sv_system.train.train_utils import find_criterion, load_checkpoint
from sv_system.data.data_utils import find_dataset
from sv_system.utils.parser import score_parser, set_score_config
from sv_system.data.dataloader import init_default_loader
from sv_system.train.si_train import val


#########################################
# Parser
#########################################
parser = score_parser()
args = parser.parse_args()
config = set_score_config(args)

#########################################
# Model Initialization
#########################################
_, datasets = find_dataset(config)
model, _ = load_checkpoint(config)
criterion = find_criterion(config, model)

#########################################
# Model Training
val_set = datasets[1]
val_loader = init_default_loader(config, val_set, shuffle=False)
val_loss, val_acc = val(config, val_loader, model, criterion)

test_set = datasets[2]
test_loader = init_default_loader(config, test_set, shuffle=False)
test_loss, test_acc = val(config, test_loader, model, criterion)

print("val acc: {}, test acc: {}".format(val_acc, test_acc))
