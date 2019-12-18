import torch

# Datasets and model's saved files
DATA_PATH = "Datasets/Dialogues"
train_dataset = "Datasets/train.txt"
knowledge_exists = True
knowledge_file = 'Knowledge/checkpoint_10000.tar'

# Thresholds - Limits #
MAX_LEN = 10  # Maximum sentence length to consider
TRIM_THRES = 2  # Trimming threshold

# Basic Tokens #
PAD = 0  # padding token
SOS = 1  # start of sentence token
EOS = 2  # start of sentence token

# Train Variables
iter_num = 10000
checkpoint_iter = 10000
tfr = 1.0
lr = 0.00005
decoder_lr = 5.0
clip = 50.0
print_every = 1
save_every = 1

# Model Variables
model_name = 'Chatbot'
attn_model = 'dot'
batch_size = 256
encoder_layers = 5
decoder_layers = 5
h_size = 500
dropout = 0.5

# Device options
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")