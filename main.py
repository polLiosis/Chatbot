import os
import Data
import config
import model
import torch
import torch.nn as nn
from torch import optim



# Load and Preprocess Data
questions, answers = Data.loadDataset(config.train_dataset)

save_dir = os.path.join("Knowledge")
voc, pairs = Data.startPreproccesing(config.train_dataset)

# Load Knowledge if model is already trained
if config.knowledge_exists:
    checkpoint = torch.load(config.knowledge_file)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building Chatbot model.....')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, config.h_size)
if config.knowledge_exists:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder and decoder models
encoder = model.Encoder(config.h_size, embedding, config.encoder_layers, config.dropout)
decoder = model.Decoder(config.attn_model, embedding, config.h_size, voc.num_words, config.decoder_layers, config.dropout)
if config.knowledge_exists:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

encoder = encoder.to(config.device)
decoder = decoder.to(config.device)

encoder.train()
decoder.train()

# Initialize optimizers
encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.lr)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.lr * config.decoder_lr)
if config.knowledge_exists:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

print('Chatbot model was built with success!')


# Train Chatbot model
print("Starting Training!")
model.train(voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, config.encoder_layers, config.decoder_layers, save_dir, config.iter_num, config.batch_size, config.print_every, config.save_every, config.clip, config.knowledge_file)

encoder.eval()
decoder.eval()

search_method = model.GreedySearch(encoder, decoder)
model.activateChatbot(voc, search_method)