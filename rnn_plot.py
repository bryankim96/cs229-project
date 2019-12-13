from datetime import datetime
import csv
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torchtext import data, vocab
from generate_embeddings import apply_preprocessing
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

NUM_CLASSES = 2
BATCH_SIZE = 128 # used to be 32

HIDDEN_DIM = 128

LEARNING_RATE = 0.01 # formerly 0.001
NUM_EPOCHS = 100

PRINT_EVERY = 1
SAVE_EPOCHS = 5

SPLIT_RATIO = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, hidden_dim, emb_weights):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding.from_pretrained(emb_weights)
        self.LSTM = nn.GRU(emb_weights.shape[1], hidden_dim)
        self.fc = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, sentence_batch):
        embeds = self.word_embeddings(sentence_batch)
        lstm_out, _ = self.LSTM(embeds)
        outputs = self.fc(lstm_out[-1, :, :])
        output_probs = functional.log_softmax(outputs, dim=1)
        return output_probs


if __name__ == "__main__":
    run_name = "fulldataset"
    embedding_file_path = "./embedding_vecs_wordseg_08122019_103814.w2vec"
    data_file_path = "../time_labeled_reports_full_preprocessed.csv" # new_labeled_path_reports_preprocessed

    print("Starting Run [{}]\n\n".format(run_name))
    print("Using data file at: {}\n".format(data_file_path))

    # Prepare data
    text_field = data.Field(
        #tokenize=apply_preprocessing,
        lower=True
    )
    label_field = data.Field(
        sequential=False,
        use_vocab=False,
        is_target=True
    )

    print("Creating TabularDatasets for training ({}) and validation ({})...".format(SPLIT_RATIO, 1.0 - SPLIT_RATIO))

    trainds, valds = data.TabularDataset(path=data_file_path,
                                         format='csv',
                                         csv_reader_params={'delimiter': '|'},
                                         fields=[('', None),
                                                 ('Unnamed: 0', None),
                                                 ('anon_id', None),
                                                 ('text', text_field),
                                                 ('label', label_field)],
                                         skip_header=True).split(split_ratio=SPLIT_RATIO)

    print("Loading vocab from embedding file: {}".format(embedding_file_path))

    # Load/prepare pre-trained embedding vectors (FastText)
    vectors = vocab.Vectors(name=embedding_file_path)
    text_field.build_vocab(trainds, valds, vectors=vectors)
    
    print("Vocab size: {}".format(len(text_field.vocab)))

    # Prepare iterator
    print("Preparing batch iterators w/ batch size {}...\n".format(BATCH_SIZE))
    traindl, valdl = data.BucketIterator.splits(datasets=(trainds, valds),
                                                batch_size=BATCH_SIZE,
                                                sort_key=lambda x: len(x.text),
                                                device=device,
                                                repeat=False
                                                )

    # Build model
    print("Building LSTM model w/ hidden dim {}...\n".format(HIDDEN_DIM))
    model = LSTMModel(HIDDEN_DIM, emb_weights=text_field.vocab.vectors)
    if torch.cuda.is_available():
        model = model.cuda()

    # Train model
    loss_function = nn.NLLLoss(weight=torch.Tensor([1, 3]).cuda())
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    num_train_batches = len(traindl)
    num_train_examples = num_train_batches * BATCH_SIZE

    num_val_batches = len(valdl)
    num_val_examples = num_val_batches * BATCH_SIZE

    print("Num train examples: {} ({} batches)".format(num_train_examples, num_train_batches))
    print("Num validation examples: {} ({} batches)".format(num_val_examples, num_val_batches))

    print("\nStarting training for {} epochs...\n".format(NUM_EPOCHS))
    
    train_losses = []
    val_losses = []
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
            _, predicted_labels = torch.max(predicted_probs.data, 1)
            train_total_correct += (predicted_labels == label_batch).sum().item()

            # if i % PRINT_EVERY == PRINT_EVERY - 1:
                # print('Batch {}/{} ----- Loss per batch (running): {}'.format(epoch + 1, i + 1,
                #                                                                       num_train_batches,
                #                                                                       running_loss / PRINT_EVERY))
                # running_loss = 0.0

        # Compute validation stats
        print("Computing validation statistics...")
        with torch.no_grad():
            val_total_loss = 0.0
            val_total_correct = 0
            
            avg_precision_0 = 0
            avg_recall_0 = 0
            avg_precision_1 = 0
            avg_recall_1 = 0
            num_batches = 0
            tp_total = 0
            fp_total = 0
            tn_total = 0
            fn_total = 0
               
            for i, batch in enumerate(valdl):
                report_batch = batch.text
                label_batch = batch.label
                predicted_probs = model(report_batch)
                val_loss = loss_function(predicted_probs, label_batch)
                val_total_loss += val_loss.item()
                _, predicted_labels = torch.max(predicted_probs.data, 1)
                val_total_correct += (predicted_labels == label_batch).sum().item()
                
                avg_precision_0 += precision_score(label_batch.cpu(), predicted_labels.cpu(), pos_label=0)
                avg_recall_0 += recall_score(label_batch.cpu(), predicted_labels.cpu(), pos_label=0)
                avg_precision_1 += precision_score(label_batch.cpu(), predicted_labels.cpu(), pos_label=1)
                avg_recall_1 += recall_score(label_batch.cpu(), predicted_labels.cpu(), pos_label=1)
                num_batches += 1

                x = confusion_matrix(label_batch.cpu(), predicted_labels.cpu()).ravel()
                tn = x[0]
                fp = x[1]
                fn = x[2]
                tp = x[3]
                # tn, fp, fn, tp
                tn_total += tn
                fp_total += fp
                fn_total += fn
                tp_total += tp
             
            avg_precision_0 /= num_batches
            avg_recall_0 /= num_batches
            avg_precision_1 /= num_batches
            avg_recall_1 /= num_batches

        train_losses.append(train_loss.item())
        val_losses.append(val_total_loss / num_val_batches)
                

        # Print end-of-epoch statistics
        print("Finished Epoch {}/{}, Train Loss: {:.3f}, Train Accuracy: {:.3f}, Validation Loss: {:.3f}, Validation Accuracy: {:.3f}".format(epoch + 1, NUM_EPOCHS,
                                                                                              train_loss.item(),
                                                                                              train_total_correct / num_train_examples,
                                                                                              val_total_loss / num_val_batches,
                                                                                              val_total_correct / num_val_examples,
                                                                                            ))
        print("Finished Epoch {}/{}, Class 0: Validation Precision: {:.3f}, Validation Recall: {:.3f}".format(epoch + 1, NUM_EPOCHS,
                                                                                              avg_precision_0,
                                                                                              avg_recall_0
                                                                                            ))

        print("Finished Epoch {}/{}, Class 1: Validation Precision: {:.3f}, Validation Recall: {:.3f}".format(epoch + 1, NUM_EPOCHS,
                                                                                              avg_precision_1,
                                                                                              avg_recall_1
                                                                                            ))
        
        print("Finished Epoch {}/{}, True positive: {}, True negative: {}, False positive: {}, False negative: {}".format(epoch + 1, NUM_EPOCHS,
                                                                                              tp_total,
                                                                                              tn_total,
                                                                                              fp_total,
                                                                                              fn_total
                                                                                            ))
        
        # Save checkpoint
        if (epoch + 1) % SAVE_EPOCHS == 0:
            PATH = './new_checkpoints/{}_epoch{}.tar'.format(run_name, epoch + 1)
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


        
    # matplotlib code    
    plt.title("Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1,NUM_EPOCHS+1),train_losses,label="Train Loss")
    plt.plot(range(1,NUM_EPOCHS+1),val_losses,label="Validation Loss")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, NUM_EPOCHS+1, 1.0))
    plt.legend()
    plt.savefig("loss_plot_100_epoch.png")
