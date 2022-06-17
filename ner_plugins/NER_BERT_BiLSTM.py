from transformers import BertModel, BertTokenizer, AdamW
import torch
import os
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import trange
import matplotlib.pyplot as plt

from seqeval.metrics import accuracy_score #used by transformer people https://github.com/huggingface/transformers/blob/a69e185074fff529ed60d936c6afe05580aee8ac/examples/legacy/token-classification/run_ner.py#L216
from sklearn.metrics import f1_score, classification_report
import numpy as np

from torchcrf import *
from nltk.tokenize import sent_tokenize

#import torchgeometry

class BERT_BiLSTM(torch.nn.Module):
  def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super(BERT_BiLSTM, self).__init__()
        # hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 256, 10

        self.embedding = BertModel.from_pretrained(
            "bert-base-cased",
            num_labels=10,
            output_attentions = False,
            output_hidden_states = True
        )

        # In this model we only train BiLSTM embedding and linear classifier, BERT embedding stands frozen
        for param in self.embedding.parameters():
          param.requires_grad = False

        self.lstm = torch.nn.LSTM(input_size=D_in, hidden_size=H, batch_first=True, bidirectional=True)

        self.linear = torch.nn.Linear(in_features= 2*H, out_features=D_out)


  def forward(self, input_ids, attention_mask=None, labels=None):
      outputs = self.embedding(input_ids=input_ids,attention_mask=attention_mask)
      sequence_output = outputs[0]
      emission, _ = self.lstm(sequence_output)
      print(f"DIMENSION OF EMISSIONS AFTER BILSTM IS {emission.size()}")
      logits = self.linear(emission)
      print(f"DIMENSION OF EMISSIONS AFTER LINEAR IS {logits.size()}")

      #emission = self.softmax(emission)
      # print(f"LEN OF SEQUENCE_OUTPUT IS {sequence_output.size()}") # torch.Size([256, 400, 100])
      # print(sequence_output.select(2, -1).size())
      # print(sequence_output[:,:,-1].size())
      #linear_output = self.classifier['linear'](sequence_output)# torch.Size([256, 400, 1]) sequence_output[:, -1]
      # sequence_output.select(2, -1)

      loss = None
      if labels is not None:
          loss_fct = torch.nn.CrossEntropyLoss()
          # Only keep active parts of the loss
          if attention_mask is not None:
              active_loss = attention_mask.view(-1) == 1
              active_logits = logits.view(-1, 10)
              active_labels = torch.where(
                  active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
              )
              loss = loss_fct(active_logits, active_labels)
          else:
              loss = loss_fct(logits.view(-1, 10), labels.view(-1))

      #print(f"LOSS IS {loss}")
      #print(f"LOGITS ARE {logits}")

      return (loss, logits)

class NER_BERT_BiLSTM():
  #torch.set_printoptions(profile="full") # TODO: REMOVE THIS SHIT BEFORE PUSH

  #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  device = torch.device("cpu")
  tag2idx = {'O':0, 'ID':1, 'PHI':2, 'NAME':3, 'CONTACT':4, 'DATE':5, 'AGE':6, 'PROFESSION':7, 'LOCATION':8, 'PAD': 9}
  tag_values = ["O","ID", "PHI", "NAME", "CONTACT", "DATE", "AGE", "PROFESSION", "LOCATION", "PAD"]
  MAX_LEN = 400
  bs = 64 # like in hugginface dev tutorial

  def __init__(self):
    self.model = BERT_BiLSTM()

    # final model is Models/NER_BERT_BiLSTM.pt
    if os.path.exists("gModels/NER_BERT_BiLSTM.pt"):
      print("Loading model")
      self.model.load_state_dict(torch.load("gModels/NER_BERT_BiLSTM.pt", map_location=torch.device('cpu')))
      print("Loaded model")

    # ref_model = BertModel.from_pretrained(
    #         "bert-base-cased",
    #         num_labels=10,
    #         output_attentions = False,
    #         output_hidden_states = True
    #     )

      # for p, p_ref in zip(self.model.embedding.parameters(), ref_model.parameters()):
      #   print(torch.eq(p.data, p_ref.data))

    self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', num_labels=len(NER_BERT_BiLSTM.tag2idx), do_lower_case=False)

  # Needed for transform_sequences
  def tokenize_and_preserve_labels(self, sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        if len(sentence) == 0:
          continue
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = self.tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)
        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)
    return tokenized_sentence, labels

  def transform_sequences(self,tokens_labels):
    """method that transforms sequences of (token,label) into feature sequences. Returns two sequence lists for X and Y"""
    print("I am in transform seq")
    # result - one document, result[i] is sentence in document, result [i][i] is word in sentence
    tokenized_sentences = []
    labels = []
    for index, sentence in enumerate(tokens_labels):
        if len(sentence) == 0:
          continue
        text_labels = []
        sentence_to_feed = []
        for word_label in sentence:
            text_labels.append(word_label[1])
            sentence_to_feed.append(word_label[0])
        a, b = self.tokenize_and_preserve_labels(sentence_to_feed, text_labels)
        tokenized_sentences.append(a)
        labels.append(b)

    # Now need to split long tokenized sequences into subsequences of length less than 512 tokens
    # not to loose valuable information in NER, basically not to cut sentences
    # i2b2 docs are very ugly and sentences in them are usually way too long as doctors forgot to put full stops...
    # tokenized_sentences AND labels are the same strucutre of 2d arrays

    # I need to take care of the issue if I am going to split beginning of the word and its end, like
    # Arina is tokenized as "Ari" and "##na", thus I cannot separate the two, otherwise it will not make sense

    distributed_tokenized_sentences, distributed_labels = [], []
    for sent, label in zip(tokenized_sentences, labels):
        if len(sent) > NER_BERT_BiLSTM.MAX_LEN:
            while len(sent) > NER_BERT_BiLSTM.MAX_LEN:
                #print("I am in while loop")
                index = NER_BERT_BiLSTM.MAX_LEN - 2
                for i in range(NER_BERT_BiLSTM.MAX_LEN - 2, 0, -1):
                    if sent[i][:2] == "##":
                        index = index - 1
                    else:
                        break

                new_sent = sent[:index] # 511 because we want to append [SEP] token in the end
                new_label = label[:index]

                sent = sent[index:]  # update given sent
                label = label[index:]

                distributed_tokenized_sentences.append(new_sent)
                distributed_labels.append(new_label)

            distributed_tokenized_sentences.append(sent)
            distributed_labels.append(label)
            #print(sent)

        else:
            distributed_tokenized_sentences.append(sent)
            distributed_labels.append(label)

    input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in distributed_tokenized_sentences],
                            maxlen=NER_BERT_BiLSTM.MAX_LEN, dtype="long", value=0.0,
                            truncating="post", padding="post")

    tags = pad_sequences([[NER_BERT_BiLSTM.tag2idx.get(l) for l in lab] for lab in distributed_labels],
                        maxlen=NER_BERT_BiLSTM.MAX_LEN, value=NER_BERT_BiLSTM.tag2idx["PAD"], padding="post",
                        dtype="long", truncating="post")

    # Result is pair X (array of sentences, where each sentence is an array of words) and Y (array of labels)
    return input_ids, tags


  def perform_NER(self,text):
      """Implementation of the method that should perform named entity recognition"""
      # tokenizer to divide data into sentences (thanks, nltk)

      print("THE MODEL!!!!!!!!!!!!!!!!!")
      print(self.model)

      list_of_sents = sent_tokenize(text)

      list_of_tuples_by_sent = []

      self.model.eval()

      for sent in list_of_sents:
          tokenized_sentence = self.tokenizer.encode(sent, truncation=True)

          input_ids = torch.tensor([tokenized_sentence])

          with torch.no_grad():
              # Run inference/classification
              output = self.model(input_ids)

          logits = output[1].detach().cpu().numpy()

          # Now get the max index position for each array, where each array is a token/word.
          #print(len(logits[0])) # 39 tokens in a sentence, 39 arrays with logits for 10 classes, choose the max among them for each token

          label_indices = []
          for p in logits:
            label_indices = np.argmax(p, axis=1)

          tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])

          new_tokens, new_labels = [], []
          for token, label_idx in zip(tokens, label_indices):
              if token.startswith("##"):
                  new_tokens[-1] = new_tokens[-1] + token[2:]
              else:
                  new_labels.append(self.tag_values[label_idx])
                  new_tokens.append(token)

          list_of_tuples = []
          for token, label in zip(new_tokens, new_labels):
              list_of_tuples.append((token, label))
              #print("{}\t{}".format(label, token))
          list_of_tuples_by_sent.append(list_of_tuples)

      # remove [CLS] and [SEP] tokens to comply wth xml structure
      for i in range(len(list_of_tuples_by_sent)):
          for tag in self.tag_values:
              if ('[CLS]', tag) in list_of_tuples_by_sent[i]:
                  list_of_tuples_by_sent[i].remove(('[CLS]', tag))

              if ('[SEP]', tag) in list_of_tuples_by_sent[i]:
                  list_of_tuples_by_sent[i].remove(('[SEP]', tag))

              if ('[UNK]', tag) in list_of_tuples_by_sent[i]:
                  list_of_tuples_by_sent[i].remove(('[UNK]', tag))

      return list_of_tuples_by_sent

  def learn(self, X_train,Y_train, epochs=1):
    """Function that actually train the algorithm"""
    if torch.cuda.is_available():
        self.model.cuda()

    tr_masks = [[float(i != 0.0) for i in ii] for ii in X_train]

    print("READY TO CREATE SOME TENZORS!!!!!!!!!!!!!!!!!!!!!!!!!!")
    tr_inputs = torch.tensor(X_train).type(torch.long)
    tr_tags = torch.tensor(Y_train).type(torch.long)
    tr_masks = torch.tensor(tr_masks).type(torch.long)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=NER_BERT_BiLSTM.bs)

    print("READY TO PREPARE OPTIMIZER!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # #bilstm = nn.LSTM() # try to change the classification head (last layer) to BiLSTM?

    # #model.classifier =

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(self.model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    #optimizer = torch.optim.Adam(params=optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    # optimizer is fine
    optimizer = AdamW(
      optimizer_grouped_parameters,
      lr=0.01,
      eps=1e-8
    )

    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    print("START TRAINING!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []

    epoch_num = 0

    for _ in trange(epochs, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================

        epoch_num += 1
        # Put the model into training mode.
        self.model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(b.to(NER_BERT_BiLSTM.device) for b in batch)
            b_input_ids, b_input_mask, b_labels = batch
            print(f"SIZE OF B LABELS is {b_labels.size()}")
            # Always clear any previously calculated gradients before performing a backward pass.
            self.model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            print("OUTPUTS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # TODO: Issue with the model itself, in that case with baseline bert


            # b_input_ids = torch.tensor([[101, 7592, 102]]).type(torch.long)
            # b_input_ids.to(NER_BERT_BiLSTM.device)
            # b_input_mask = torch.tensor([[1,1,1]]).type(torch.long)
            # b_input_mask.to(NER_BERT_BiLSTM.device)


            # outputs = self.model.bert(b_input_ids,
            #                  attention_mask=b_input_mask, return_dict=True)


            # print("IDS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(b_input_ids)
            # print("MASKS")
            # print(b_input_mask)
            # print("LABELS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(b_labels)

            #Previous outputs of disilbert + linear classification
            outputs = self.model(b_input_ids,
                            attention_mask=b_input_mask, labels=b_labels)

            #print(outputs)

            #get the loss
            loss = outputs[0]

            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        # Save intermediate weights of the model, i.e. if computer goes crazy and drops the training or you
        # want to test the performance from different epochs
        torch.save(self.model.state_dict(), os.path.join("Models_intermediate/", 'BERT_BiLSTM_epoch-4.pt'.format(epoch_num)))

    # Plot the post-epoch-training learning curve.
    plt.figure()
    plt.plot(loss_values, 'b-o', label="training loss")
    # Label the plot.
    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

  def evaluate(self, X_test,Y_test):
    if torch.cuda.is_available():
        self.model.cuda()
    """Function to evaluate algorithm"""
    val_masks = [[float(i != 0.0) for i in ii] for ii in X_test]
    val_inputs = torch.tensor(X_test).type(torch.long)
    val_tags = torch.tensor(Y_test).type(torch.long)
    val_masks = torch.tensor(val_masks).type(torch.long)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=NER_BERT_BiLSTM.bs)

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode to set dropout and batch normalization layers to evaluation mode to have consistent results
    self.model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(self.device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = self.model(b_input_ids,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [NER_BERT_BiLSTM.tag_values[p_i] for p, l in zip(predictions, true_labels)
                                for p_i, l_i in zip(p, l) if NER_BERT_BiLSTM.tag_values[l_i] != "PAD"]

    ###############################################################################
    # reconstruct given text for purposes of algorithms' performance comparison
    # our X_test is again a list of sentences, i.e. 2d array
    tokens = [self.tokenizer.convert_ids_to_tokens(sent) for sent in X_test]
    # Unpack tokens into 1d array to be able to go through it with labels
    # [PAD] and not just PAD because that is what BERT actually puts
    tokens_flat = [item for sublist in tokens for item in sublist if item != "[PAD]"]

    #for sentence in tokens:
    new_tokens, new_labels = [], []
    for token, pred in zip(tokens_flat, pred_tags):
        #print("{}\t{}".format(token, pred))
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(pred)
            new_tokens.append(token)
    ###############################################################################

    valid_tags = [NER_BERT_BiLSTM.tag_values[l_i] for l in true_labels
                                for l_i in l if NER_BERT_BiLSTM.tag_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    print("Validation F1-Score: {}".format(f1_score(valid_tags, pred_tags, average='weighted')))
    labels = ["ID", "PHI", "NAME", "CONTACT", "DATE", "AGE",
              "PROFESSION", "LOCATION"]
    print(classification_report(valid_tags, pred_tags, digits=4, labels=labels))
    print()

    # to evaluate union/intersection of algorithms
    return new_labels


  def save(self, model_path):
    """
    Function to save model. Models are saved as h5 files in Models directory. Name is passed as argument
    :param model_path: Name of the model file
    :return: Doesn't return anything
    """
    torch.save(self.model.state_dict(), "Models/"+model_path+".pt")
    print("Saved model to disk")
