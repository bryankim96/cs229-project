from datetime import datetime
import csv
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torchtext import data, vocab
from generate_embeddings import apply_preprocessing
from ignite.engine import Events, create_supervised_evaluator, _prepare_batch
from ignite.contrib.engines import create_supervised_tbptt_trainer
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor

NUM_CLASSES = 2
BATCH_SIZE = 200

HIDDEN_DIM = 128

LEARNING_RATE = 0.001
NUM_EPOCHS = 50

PRINT_EVERY = 1
SAVE_EPOCHS = 5

SPLIT_RATIO = 0.9

TBTT_STEP = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, hidden_dim, emb_weights):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding.from_pretrained(emb_weights)
        self.LSTM = nn.LSTM(emb_weights.shape[1], hidden_dim)
        self.fc = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, sentence_batch):
        embeds = self.word_embeddings(sentence_batch)
        lstm_out, hidden = self.LSTM(embeds)
        outputs = self.fc(lstm_out[-1, :, :])
        output_probs = functional.log_softmax(outputs, dim=1)
        return (output_probs, hidden)
        # return output_probs

def create_supervised_tbptt_evaluator(model, metrics=None,
                                device=None, non_blocking=False,
                                prepare_batch=_prepare_batch,
                                output_transform=lambda x, y, y_pred: (y_pred, y,)):
    """
    Modified version of factory function (default in ignite) for creating an evaluator for supervised models.
    Made it compatible with tbptt trainer since model is expected to return hidden state as well.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred, hidden = model(x)
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y


if __name__ == "__main__":
    run_name = "firstfull"
    embedding_file_path = "./embedding_vecs_wordseg300_12122019_124551.w2vec"# "./embedding_vecs_wordseg_08122019_103814.w2vec"
    data_file_path = "../new_labeled_reports_full_preprocessed.csv"# "../time_labeled_reports_full_preprocessed.csv" # new_labeled_reports_full_preprocessed.csv" # new_labeled_path_reports_preprocessed

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
                                                 # ('Unnamed: 0', None),
                                                 ('anon_id', None),
                                                 ('text', text_field),
                                                 ('label', label_field)],
                                         skip_header=True).split(split_ratio=SPLIT_RATIO)

    print("Loading vocab from embedding file: {}".format(embedding_file_path))

    # Load/prepare pre-trained embedding vectors (FastText)
    vectors = vocab.Vectors(name=embedding_file_path)
    text_field.build_vocab(trainds, valds, vectors=vectors)

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
    loss_function = nn.NLLLoss(weight=torch.Tensor([3, 1]).cuda())
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    num_train_batches = len(traindl)
    num_train_examples = num_train_batches * BATCH_SIZE

    num_val_batches = len(valdl)
    num_val_examples = num_val_batches * BATCH_SIZE

    print("Num train examples: {} ({} batches)".format(num_train_examples, num_train_batches))
    print("Num validation examples: {} ({} batches)".format(num_val_examples, num_val_batches))

    print("\nStarting training for {} epochs...\n".format(NUM_EPOCHS))

    print(type(traindl))

    # create ignite trainer
    trainer = create_supervised_tbptt_trainer(model, optimizer, loss_function, tbtt_step=TBTT_STEP)

    evaluator = create_supervised_tbptt_evaluator(model, metrics={'accuracy': Accuracy(), 
                                                                  'nll': Loss(loss_function), 
                                                                  'precision':Precision(output_transform=thresholded_output_transform), 
                                                                  'recall':Recall(output_transform=thresholded_output_transform)})
    # evaluator = create_supervised_evaluator(model, metrics=['accuracy'])

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(traindl)
        metrics = evaluator.state.metrics
        # precision_0 = metrics['precision'][0]
        # precision_1 = metrics['precision'][1]
        # recall_0 = metrics['recall'][0]
        # recall_1 = metrics['recall'][1]
        print(metrics['precision'])
        # print(type(precision_0))
        print(metrics['recall'])
        # string_precision = "$2.3f" % metrics['precision']
        # string_recall = "$2.2f" % metrics['recall']
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(valdl)
        metrics = evaluator.state.metrics
        print(metrics['precision'])
        # print(type(precision_0))
        print(metrics['recall'])
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

    trainer.run(traindl, max_epochs=100)
