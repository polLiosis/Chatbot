# Chatbot
A chatbot which is based on the respective PyTorch Chatbot tutorial (Available at: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)

# Dataset
MetaLWOz (Available at: https://www.microsoft.com/en-us/research/project/metalwoz/). MetaLWOz dataset contains 37,884 goal-oriented dialogs, covering 227 tasks in 47 domains.

# Directories
* Datasets: Contains the files used in order to extract the train set and the train set itself (train.txt)
* Knowledge: Contains checkpoints of the knowledge that the model acquired during training.

# Run Chatbot
* In order to create the train set, we run the parse.py file
* In order to train/test the Chatbot, we execute the main.py file


# Configuration (config.py)
* ATA_PATH = "Datasets/Dialogues"
* train_dataset = "Datasets/train.txt"
* knowledge_exists = True
* knowledge_file = 'Knowledge/checkpoint_10000.tar'

In case we want to train our model, we change the knowledge_exists and knowledge_file to False and None respectively.
We should also uncomment (or comment in case we want to test it) the following lines (57, 58) in main.py file:

``` 
print("Starting Training!")
model.train(voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, 
            config.encoder_layers, config.decoder_layers, save_dir, config.iter_num, config.batch_size, 
            config.print_every, config.save_every, config.clip, config.knowledge_file)```