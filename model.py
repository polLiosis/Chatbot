import random
import os
import itertools
import Data
import config
import torch
import torch.nn as nn
import torch.nn.functional as F

def sentenceToIndexes(voc, sentence):
    """
    Covert sentence to indexes

    For example:
    your welcome !  - > [136, 211, 46, 2]

    :param voc: the vocabulary
    :param sentence: sentence to convert
    :return: indexes for the specific sentence
    """
    return [voc.word2index[word] for word in sentence.split(' ')] + [2]


def zeroPadding(bsize, fillvalue=config.PAD):
    """
    Create zero-padded tensor

    :param bsize: batch size
    :param fillvalue: Value to fill (Always 0)
    :return: A zero-padded tensor
    """
    return list(itertools.zip_longest(*bsize, fillvalue=fillvalue))

def binaryMatrix(tensors, value=config.PAD):
    """
    Create a binary matrix which is based on input tensors.
    In each cell/position of the the binary matrix the value is:
    - 1 if it is not a PAD token
    - 0 if it is a PAD token

    :param tensors: tensors
    :param value: The PAD value (Always 0)
    :return: A binary matrix
    """
    matrix = []
    for i, tensor in enumerate(tensors):
        matrix.append([])
        for value in tensor:
            if value == config.PAD:
                matrix[i].append(0)
            else:
                matrix[i].append(1)
    return matrix

def convertBatch(input_batch, voc):
    """
    Convert sentences to tensors
    :param batch: batch to convert
    :param voc: Vocabulary
    :return: padded_batch (new converted batch) and lengths (lengths for each sentence in the input batch)
    """
    batch = [sentenceToIndexes(voc, sentence) for sentence in input_batch]
    # Tensor containing the lengths for each sentence in input batch
    lengths = torch.tensor([len(sentence) for sentence in batch])
    x = zeroPadding(batch)
    padded_batch = torch.LongTensor(x)
    return padded_batch, lengths



def getMaskAndLen(input_batch, voc):
    batch = [sentenceToIndexes(voc, sentence) for sentence in input_batch]
    max_len = max([len(indexes) for indexes in batch])
    x = zeroPadding(batch)
    mask = binaryMatrix(x)
    mask = torch.BoolTensor(mask)
    lt = torch.LongTensor(x)
    return lt, mask, max_len

def generateTrainData(voc, qa_pairs):
    """
    Generate train data to use in our model from QA_pairs

    :param voc: vocabulary
    :param qa_pairs: List with QA pairs (text form)
    :return: train data
    """
    qa_pairs.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    q_batch, a_batch = [], []
    for pair in qa_pairs:
        q_batch.append(pair[0])
        a_batch.append(pair[1])
    q_batch, lengths = convertBatch(q_batch, voc)
    a_batch, mask, max_len = getMaskAndLen(a_batch, voc)
    return q_batch, lengths, a_batch, mask, max_len




class Encoder(nn.Module):
    def __init__(self, h_size, embedding, n_layers=1, dropout=0):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.h_size = h_size
        self.embedding = embedding
        self.gru = nn.GRU(h_size, h_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, batch, lengths, hidden=None):
        """
        :param batch: batch of sentences
        :param lengths: list of sentence lengths for each sentence in the input batch
        :param hidden: hidden state
        :return: features from the last hidden layer and final hidden state
        """
        embedded = self.embedding(batch)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        outputs, final_hidden_state = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        sum_outputs = outputs[:, :, :self.h_size] + outputs[:, : ,self.h_size:]
        return sum_outputs, final_hidden_state

class AttentionMechanism(nn.Module):
    def __init__(self, method, h_size):
        super(AttentionMechanism, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError("Wrong appropriate attention method.")
        self.h_size = h_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, h_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, h_size)
            self.v = nn.Parameter(torch.FloatTensor(h_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        x = self.attn(encoder_output)
        return torch.sum(hidden * x, dim=2)

    def concat_score(self, hidden, encoder_output):
        attn_weight = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * attn_weight, dim=2)

    def forward(self, hidden, encoder_outputs):
        global attn_weights
        if self.method == 'general':
            attn_weights = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_weights = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_weights = self.dot_score(hidden, encoder_outputs)
        f_attn_weights = attn_weights.t()

        return F.softmax(f_attn_weights, dim=1).unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, att_model, embedding, h_size, output_size, n_layers=1, dropout=0.1):
        super(Decoder, self).__init__()

        self.att_model = att_model
        self.h_size = h_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(h_size, h_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(h_size * 2, h_size)
        self.out = nn.Linear(h_size, output_size)

        self.attn = AttentionMechanism(att_model, h_size)

    def forward(self, input_step, last_hidden_state, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        bigru_output, final_hidden_state = self.gru(embedded, last_hidden_state)
        attn_weights = self.attn(bigru_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        bigru_output = bigru_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((bigru_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, final_hidden_state


class GreedySearch(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearch, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=config.device, dtype=torch.long) * config.SOS
        all_tokens = torch.zeros([0], device=config.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=config.device)
        # DPass to decoder one token a time
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Get token and softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_tokens, all_scores

def maskNLLLoss(input, target, mask):
    """
    Calculate the average negative log likelihood of 1s in the mask
    """
    s = mask.sum()
    crossEntropy = -torch.log(torch.gather(input, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(config.device)
    return loss, s.item()

def single_train_iteration(input_batch, lengths, output_batch, mask, max_target_len, encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, embedding, max_len=config.MAX_LEN ):
    """
    The whole process of a single iteration during the training of the model

    :param input_batch: the input batch
    :param lengths: the lengths of the sentences in the specific batch
    :param output_batch:
    :param mask:
    :param max_target_len:
    :param encoder: the encoder
    :param decoder: the decoder
    :param encoder_optimizer: the optimizer of the encoder
    :param decoder_optimizer: the optimizer of the decoder
    :param batch_size: the size of the batch
    :param clip:
    :param embedding: embeddings
    :param max_len: the maximum length of ta sentence
    :return:
    """
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_batch = input_batch.to(config.device)
    lengths = lengths.to(config.device)
    target_variable = output_batch.to(config.device)
    mask = mask.to(config.device)

    loss = 0
    losses = []
    n_totals = 0

    # Pass to Encoder
    encoder_outputs, encoder_final_hidden_state = encoder(input_batch, lengths)

    #Decoder input (always start with SOS)
    decoder_input = torch.LongTensor([[1 for _ in range(batch_size)]])
    decoder_input = decoder_input.to(config.device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_final_hidden_state[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < config.tfr else False
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_variable[t].view(1, -1)
            mask_loss, n_total = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            losses.append(mask_loss.item() * n_total)
            n_totals += n_total
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(config.device)
            mask_loss, n_total = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            losses.append(mask_loss.item() * n_total)
            n_totals += n_total

    loss.backward()
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(losses) / n_totals

def train(voc, QA_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, iterations, batch_size, print_every, save_every, clip, knowledge_file):
    """
    Function we use for the training of the model

    :param model_name: model name
    :param voc: vocabulary
    :param QA_pairs: QA_pairs
    :param encoder: the encoder
    :param decoder: the decoder
    :param encoder_optimizer: the optimizer of the encoder
    :param decoder_optimizer: the ptimizer of the decoder
    :param embedding: embeddings
    :param encoder_n_layers: neural layers of encoder
    :param decoder_n_layers: neural layers of decoder
    :param save_dir: the directory where we will save the aquired knowledge during the training process
    :param iterations: the maximum number of iterations
    :param batch_size: the size of the batch
    :param print_every: maximum step for printing
    :param save_every: maximum number of steps in order to create a new checkpoint
    :param clip: clip
    :param corpus_name: the name
    :param knowledge_file: file to load info from training (Knowledge base)
    """
    # Load batches for each iteration
    training_batches = [generateTrainData(voc, [random.choice(QA_pairs) for _ in range(batch_size)]) for _ in range(iterations)]

    start_iteration = 1
    print_loss = 0
    if knowledge_file:
        start_iteration = config.checkpoint['iteration'] + 1

    # Training
    for iteration in range(start_iteration, iterations + 1):
        training_batch = training_batches[iteration - 1]
        # Retrieve data from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = single_train_iteration(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, embedding)
        print_loss += loss

        # Print
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / iterations * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format('checkpoint', iteration)))


def replyToSentence(voc, sentence, search_method):
    """
    Process to evaluate the best possible reply for a given sentence/query

    :param search_method: search method
    :param voc: vocabulary
    :param sentence: input sentence from user
    :return: Bot's reply
    """
    indexes_batch = [sentenceToIndexes(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

    input_batch = input_batch.to(config.device)
    lengths = lengths.to(config.device)

    tokens, scores = search_method(input_batch, lengths, config.MAX_LEN)
    reply = [voc.index2word[token.item()] for token in tokens]

    return reply


def activateChatbot(voc, search_method):
    """
    Activate Chatbot's dialog system.
    User enters a sentence, the model will evaluate it and then it returns a possible reply.
    If the user enters 'exit' then the chatbot system will be terminated.
    In case the user enters a sentence containing an unknown vocabulary word, then the system will encounter a error!

    :param search_method: Search method
    :param voc: vocabulary
    """
    user_input = ''
    while(True):
        try:
            user_input = input('>>> ')
            if user_input == 'exit': break
            user_input = Data.preprocessString(user_input)
            answer = replyToSentence(voc, user_input, search_method)
            answer[:] = [token for token in answer if not (token == 'EOS' or token == 'PAD')]
            print('Chatbot:', ' '.join(answer))
        except KeyError:
            print("Error!!!...Unknown vocabulary word...")

