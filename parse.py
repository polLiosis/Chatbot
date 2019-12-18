import Data

# Extract QA pairs as lists
questions, replies = Data.extractQAPairs()
# Create train set
Data.createTrainSet(questions, replies)