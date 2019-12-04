from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torchtext import data, vocab
from generate_embeddings import apply_preprocessing

NUM_CLASSES = 2
BATCH_SIZE = 32

HIDDEN_DIM = 128

LEARNING_RATE = 0.001
NUM_EPOCHS = 100

PRINT_EVERY = 100
SAVE_EPOCHS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, hidden_dim, emb_weights):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding.from_pretrained(emb_weights)
        self.LSTM = nn.LSTM(emb_weights.shape[1], hidden_dim)
        self.fc = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.LSTM(embeds.view(len(sentence), 1, -1))
        outputs = self.fc(lstm_out.view(len(sentence), -1))
        output_probs = functional.log_softmax(outputs, dim=1)
        return output_probs


if __name__ == "__main__":
    run_name = ""
    embedding_file_path = ''
    data_file_path = '../haruka_pathology_reports_111618.csv'

    # Prepare data
    text_field = data.Field(
        tokenize=apply_preprocessing,
        lower=True
    )
    label_field = data.Field(
        sequential=False,
        use_vocab=False,
        is_target=True
    )

    trainds, valds = data.TabularDataset(path=data_file_path,
                                         format='csv',
                                         csv_reader_params={'delimiter': '|'},
                                         fields={'text': text_field,
                                                 'label': label_field},
                                         skip_header=True).split(split_ratio=0.9)

    # Load/prepare pre-trained embedding vectors (FastText)
    vectors = vocab.Vectors(name=embedding_file_path)
    text_field.build_vocab(trainds, valds, vectors=vectors)
    label_field.build_vocab(trainds)

    # Prepare iterator
    traindl, valdl = data.BucketIterator.splits(datasets=(trainds, valds),
                                                batch_sizes=(BATCH_SIZE, BATCH_SIZE),
                                                sort_key=lambda x: len(x.text),
                                                device=device,
                                                repeat=False)

    # Build model
    model = LSTMModel(HIDDEN_DIM, emb_weights=text_field.vocab.vectors)

    # Train model
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.params(), lr=LEARNING_RATE)

    num_train_batches = len(traindl)
    num_train_examples = num_train_batches * BATCH_SIZE

    num_val_batches = len(valdl)
    num_val_examples = num_val_batches * BATCH_SIZE

    for epoch in range(NUM_EPOCHS):
        train_total_correct = 0
        running_loss = 0.0
        print("Starting Epoch {}/{}...".format(epoch + 1, NUM_EPOCHS))
        for i, batch in enumerate(traindl):
            report_batch = batch.text
            label_batch = batch.label

            # Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            predicted_probs = model(report_batch)

            train_loss = loss_function(predicted_probs, label_batch)
            train_loss.backward()
            optimizer.step()

            # print loss every PRINT_EVERY batches
            running_loss += train_loss.item()
            predicted_labels = torch.max(predicted_probs.data, 1)
            train_total_correct += (predicted_labels == label_batch).sum().item()

            if i % PRINT_EVERY == PRINT_EVERY - 1:
                print('Batch {}/{} ----- Loss per batch (running): {}'.format(epoch + 1, i + 1,
                                                                                       num_train_batches,
                                                                                       running_loss / PRINT_EVERY))
                running_loss = 0.0

        # Compute validation stats
        print("Computing validation statistics...")
        with torch.no_grad():
            val_total_loss = 0.0
            val_total_correct = 0
            for i, batch in enumerate(valdl):
                report_batch = batch.text
                label_batch = batch.label
                predicted_probs = model(report_batch)
                val_loss = loss_function(predicted_probs, label_batch)
                val_total_loss += val_loss.item()
                predicted_labels = torch.max(predicted_probs.data, 1)
                val_total_correct += (predicted_labels == label_batch).sum().item()

        # Print end-of-epoch statistics
        print("Finished Epoch {}/{}, Train Loss: {:.3f}, Train Accuracy: {:.3f}, Validation Loss: {:.3f}, Validation Accuracy: {:.3f}".format(epoch + 1, NUM_EPOCHS,
                                                                                              train_loss.item(),
                                                                                              train_total_correct / num_train_examples,
                                                                                              val_total_loss / num_val_batches,
                                                                                              val_total_correct / num_val_examples,
                                                                                            ))

        # Save checkpoint
        if epoch + 1 % SAVE_EPOCHS == 0:
            PATH = './{}_epoch{}.tar'.format(run_name, epoch + 1)
            print('Saving checkpoint to path: {}'.format(PATH))

            torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'last_training_loss': train_loss.item(),
            'last_training_accuracy': train_total_correct / num_train_examples,
            'last_val_loss': val_total_loss / num_val_batches,
            'last_val_accuracy': val_total_correct / num_val_examples,
            'embedding_path': embedding_file_path 
            }, PATH)
        
        
