---
layout: post
title: Creating a Basic Neural Machine Translation System
---

This article was inspired by [this article][a-star-nmt] from Raven Hon, and [this article][nmt-from-scratch] from Jason Brownlee. Rather than simply following and applying articles online, I decided to write my own tutorial following the philosophy of Albert Einstein's quote:

> If you can't explain it simply, you don't understand it well enough.

## Contents

* Table of contents (do not remove: this line is not displayed).
{:toc}

## Introduction

As suggested by their names, Neural Machine Translation (NMT) systems are neural networks used to translate text from one language to another. Today, we are going to see how to create a basic version of an NMT to translate French to English.

NMTs basically consist of:
* an encoding part turning the input text in the source language to translate from into a vector of features
* a decoding part turning the vector of features into an output text

These encoders and decoders are a set of one or more Recurrent Neural Networks (RNNs). In our case, we are going to use Long Short Term Memory (LSTM) cells rather than "vanilla" RNNs, which tend to perform better, but are not harder to implement.

NMT systems can be enhanced by, in the most common cases, adding layers to the encoder/decoder (stacking LSTMs for instance), and adding an attention layer in between to help the system focus on the meaningful elements of the sentence. This is, however, out of the scope of this article.

## Resources

List of all the resources used in this article:
* [French/English dataset][mt.org-down] from [ManyThings.org][mt.org]
* Python 3.8.3
* Keras 2.3.1
* TensorFlow 2.2.0
* Numpy 1.18.4
* Pandas 1.0.4
* [SentencePiece][sp] 0.1.91

Specific resources such as the French/English sentence pairs dataset and the SentencePiece tokenizer will be explained in their following dedicated sections.

You can also find the Python notebook I used to develop the code [on my GitHub][notebook].

## Taking a look at the dataset

The dataset we are going to use today is a set of sentence pairs in French and English originally created by the [Tatoeba project][tato] to be used as [Anki][anki] flashcards for language learning, and available on [ManyThings.org][mt.org]. You can download it [here directly][mt.org-down] or browse the list of datasets on their website.

The dataset is made out of fairly short and simple sentences, which is perfect for the "toy" example we are going to create. I can tell you from first hand experience that our model would likely learn much worse on datasets made out of longer and more complex sentences.

Download the dataset, then extract it with the following command:

```bash
unzip ./fra-eng.zip
```

The English and French sentences are saved together in a CSV style file called fra.txt. Here is what your repertory should look like:

```
├── _about.txt	# info file about the dataset
├── fra.txt		# dataset csv file
├── fra-eng.zip	# dataset zip archive
└── nmt.ipynb	# my python notebook for coding
```

The only file we need is `fra.txt`, containing the English sentences, French sentences, and some copyright info in 3 columns separated by tabs. Let's import it in Python just to check the contents:

```python
# Source text database
data_filename = "./fra.txt" 

# Opening the text dataset file
file = open(
    data_filename,
    mode = 'rt',
    encoding = 'utf-8')
# Getting the text content
raw_text = file.read()
# Closing the file handle
file.close()
```

Let's check what the text looks like:

```python
# Checking the beginning of the text
print(raw_text[:256])
```

Output:

```
Go.	Va !	CC-BY 2.0 (France) Attribution: tatoeba.org #2877272 (CM) & #1158250 (Wittydev)
Hi.	Salut !	CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #509819 (Aiji)
Hi.	Salut.	CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #4320462 (g
```

## Data preprocessing 1: cleaning and reducing the dataset

The dataset contains a total of 175,623 sentence pairs. In order to reduce processing/training time and allow us to play around more easily while developing our network, let's reduce our dataset to the first 10,000 sentence pairs.

Each sentence pair is on a single row of our dataset, in the first and second columns (index 0 and 1), respectively for English and French. We will import those in a Pandas data frame by reading the file as a csv:

```python
import pandas as pd
subset_size = 10000

# Importing the dataset in a Pandas data frame
train_df = pd.read_csv(
    data_filename, # path to our dataset file
    sep='\t', # tab delimiter between columns in the csv
    usecols=[0, 1], # import only columns 0 and 1
    nrows=subset_size, # read only the first subset_size rows
    names=["en","fr"]) # label them 'en' and 'fr'
```

Remember to always check what you are doing. The output of `print(train_df)` should look like this:

```
		    en                                fr
0                  Go.                              Va !
1                  Hi.                           Salut !
2                  Hi.                            Salut.
3                 Run!                           Cours !
4                 Run!                          Courez !
...                ...                               ...
9995  Be more precise.                 Soit plus précis.
9996  Be quiet, girls.  Restez tranquilles, les filles !
9997  Be very careful.               Sois très prudent !
9998  Be very careful.              Soyez très prudent !
9999  Be very careful.             Soyez très prudente !

[10000 rows x 2 columns]
```

Now would be the moment to do any "best practice" data cleaning like lowercasing the whole text, getting rid of punctuation, removing potential unprintable characters, etc. I will do none of that and train our network directly on the real sentences.

Let's then save our reduced (and if you do it, cleaned) dataset:

```python
clean_train_filename = "clean_en_fr.txt" # both languages

# Saving our cleaned and reduced dataset
train_df.to_csv(
    clean_train_filename,
    sep='\t', # using tab separators
    index=False) # don't print the row index in the csv
```

I will also be saving the English sentences and French sentences separately for reasons that will become apparent in the next section:

```python
clean_en_filename = "clean_en.txt" # file to save cleaned English text data
clean_fr_filename = "clean_fr.txt" # French

# Saving the English part separately for SentencePiece
train_df.to_csv(
    clean_en_filename,
    columns=['en'], # print only the column 'en'
    index=False)

# And the French one
train_df.to_csv(
    clean_fr_filename,
    columns=['fr'], # print only the column 'fr'
    index=False)
```

Our reduced and cleaned up dataset is now ready to go!

## Tokenizer: SentencePiece

Before we can do anything with a neural network, we have to make sure our data can be read by a machine. However, machines don't work well with words and sentences, but rather exclusively with numbers. This means that we have to turn our sentences into numbers. This is done by the tokenizer.

Tokenizers convert characters or words into tokens, numbers representing said characters or words. There are many different tokenizers available. The most common ones split sentences into words and convert each word into a unique number representing this word in a vocabulary list. This works great for languages where words are well delimited by spaces, but cannot be used for languages where word delimitation is fuzzier (like Chinese and Japanese for instance).

Enter the tokenizer we are going to use here: [SentencePiece][sp]. It has been designed to be used as a language independent, unsupervised tokenizer. This means we are going to let the tokenizer learn how to split our sentences by itself, in each language, by training on a text corpus.

You can install SentencePiece directly as a Python module using pip:

```bash
pip3 install --user sentencepiece
```

Let's import the module and train the tokenizer for English and French using out dataset. SentencePiece trains directly on the text files we give it as argument, without any intervention required. This is where our separate English and French sentence files come into play. SentencePiece will be learning vocabulary in each language using the text in the given file.

All we have to specify is a vocabulary size. This is a fixed number of subwords we want SentencePiece to learn. All the encoding and decoding between subwords and tokens will happen in that vocabulary list. Any subword that has not been learned by SentencePiece will be marked as 'unknown'.

```python
import sentencepiece as sp

# Let's define vocabulary size for out tokenizer
vocab_size = 2000
# We'll use the same size for both languages to simplify
en_vocab_size = vocab_size
fr_vocab_size = vocab_size

# Training the model for English
sp.SentencePieceTrainer.train(
    input = clean_en_filename,
    model_prefix = 'en',
    vocab_size = en_vocab_size,
)
# Training the model for French
sp.SentencePieceTrainer.train(
    input = clean_fr_filename,
    model_prefix = 'fr',
    vocab_size = fr_vocab_size,
)
```

We can then check that the tokenizer has been trained by creating an instance of the tokenizer for each language, loading its trained weights for both languages, and checking the tokenization on sample sentences.

```python
# Creating a tokenizer object for English
en_sp = sp.SentencePieceProcessor()
# Loading the English model
en_sp.Load("en.model")
# Creating a tokenizer object for French
fr_sp = sp.SentencePieceProcessor()
# Loading the French model
fr_sp.Load("fr.model")

# Testing the English tokenizer
en_test_sentence = "I like green apples."
# Encoding pieces
print(en_sp.EncodeAsPieces(en_test_sentence))
# Encoding pieces as IDs
print(en_sp.EncodeAsIds(en_test_sentence))
# Decoding encoded IDs
print(en_sp.DecodeIds(en_sp.EncodeAsIds(en_test_sentence)))

# Testing the French tokenizer
fr_test_sentence = "J'aime les pommes vertes."
# Encoding pieces
print(fr_sp.EncodeAsPieces(fr_test_sentence))
# Encoding pieces as IDs
print(fr_sp.EncodeAsIds(fr_test_sentence))
# Decoding encoded IDs
print(fr_sp.DecodeIds(fr_sp.EncodeAsIds(fr_test_sentence)))
```

Output:

```
['▁I', '▁like', '▁green', '▁a', 'p', 'ple', 's', '.']
[13, 66, 3745, 10, 315, 6333, 15, 5]
I like green apples.
['▁J', "'", 'aime', '▁les', '▁p', 'omme', 's', '▁verte', 's', '.']
[141, 3, 7161, 14, 563, 5964, 11, 5926, 11, 6]
J'aime les pommes vertes.
```

We can see that the tokenizer does not necessarily keep whole words, but can cut them into what it sees as components of words. A very interesting example here is the fact that the plural 's' in both the English 'apples' and the French 'pommes' is seen as a separate component. This is likely because the tokenizer has identified that nouns can exist both with and without an ending 's', depending on their number.

## Data preprocessing 2: tokenizing and formatting

Now that the tokenizer is ready, we can load our cleaned and reduced dataset, load the trained tokenizer models (if you have not done so previously when checking the tokenizer's training results), and start encoding our sentences into tokens.

We will load our sentences into a Pandas data frame in their respective columns `'en'` and `'fr'`.

```python
# Load the cleaned up dataset
train_df = pd.read_csv(
    clean_train_filename,
    sep='\t')
```

We can check how the data frame looks like by printing it:

```
		    en                                fr
0                  Go.                              Va !
1                  Hi.                           Salut !
2                  Hi.                            Salut.
3                 Run!                           Cours !
4                 Run!                          Courez !
...                ...                               ...
9995  Be more precise.                 Soit plus précis.
9996  Be quiet, girls.  Restez tranquilles, les filles !
9997  Be very careful.               Sois très prudent !
9998  Be very careful.              Soyez très prudent !
9999  Be very careful.             Soyez très prudente !

[10000 rows x 2 columns]
```

Again, creating instances of the tokenizer and loaded the trained models for encoding/decoding French and English:

```python
# Load trained tokenizer for English and French
# Creating a tokenizer object for English
en_sp = sp.SentencePieceProcessor()
# Loading the English model
en_sp.Load("en.model")
# Creating a tokenizer object for French
fr_sp = sp.SentencePieceProcessor()
# Loading the French model
fr_sp.Load("fr.model")
```

Now comes the important part: encoding all of our sentences. For this, we will define a small function that goes through the given data frame column using its label, and encode each sentence one by one, before adding them all to a new column `'en_ids'` or `'fr_ids'` (one per language).

```python
# Function to tokenize our text (list of sentences) and
# add it to our data frame in the column 'label'
def tokenize_text(df, spm, txt_label, id_label):
    ids = []
    for line in df[txt_label].tolist():
	id_line = spm.EncodeAsIds(line)
	ids.append(id_line)
    df[id_label] = ids

# Let's run this function on the English text
tokenize_text(train_df, en_sp, 'en', 'en_ids')
# And on the French text
tokenize_text(train_df, fr_sp, 'fr', 'fr_ids')
```

We can (and should) of course check what our data frame looks like by printing it:

```
		    en                                fr  \
0                  Go.                              Va !   
1                  Hi.                           Salut !   
2                  Hi.                            Salut.   
3                 Run!                           Cours !   
4                 Run!                          Courez !   
...                ...                               ...   
9995  Be more precise.                 Soit plus précis.   
9996  Be quiet, girls.  Restez tranquilles, les filles !   
9997  Be very careful.               Sois très prudent !   
9998  Be very careful.              Soyez très prudent !   
9999  Be very careful.             Soyez très prudente !   

		      en_ids                               fr_ids  
0                    [81, 3]                             [199, 9]  
1                  [1004, 3]                             [992, 9]  
2                  [1004, 3]                             [992, 3]  
3                  [472, 18]                      [18, 812, 5, 9]  
4                  [472, 18]                     [18, 812, 49, 9]  
...                      ...                                  ...  
9995  [42, 320, 282, 919, 3]              [118, 22, 203, 1143, 3]  
9996   [42, 271, 89, 753, 3]  [6, 221, 265, 5, 66, 20, 5, 765, 9]  
9997   [42, 17, 151, 177, 3]            [118, 5, 58, 169, 361, 9]  
9998   [42, 17, 151, 177, 3]           [108, 49, 58, 169, 361, 9]  
9999   [42, 17, 151, 177, 3]           [108, 49, 58, 169, 375, 9]  

[10000 rows x 4 columns]
```

Our data frame has all 4 columns we should have, and we can see that the sentences have all been turned into a list of integer tokens.

Since networks have trouble with variable length sentences, we need to pad our encoded sentences so that they are equal in length. We will check the maximum sentence length for each language, and pad all sentences to match them.

```python
# Check tokenized English sentence length
en_max_len = max(len(line) for line in train_df['en_ids'].tolist())
# Check tokenized French sentence length
fr_max_len = max(len(line) for line in train_df['fr_ids'].tolist())
```

In my case (this is depending on the result of the SentencePiece training), the respective maximum sentence length for English and French, `en_max_len` and `fr_max_len`, are 10 and 22. Let us pad all the English sentences to `en_max_len`, and all the French sentences to `fr_max_len`.

```python
from keras.preprocessing.sequence import pad_sequences

# Pad English tokens
padded_en_ids = pad_sequences(
    train_df['en_ids'].tolist(),
    maxlen = en_max_len,
    padding = 'post')
# Add them to our training data frame
train_df['pad_en_ids'] = padded_en_ids.tolist()

# Pad French tokens
padded_fr_ids = pad_sequences(
    train_df['fr_ids'].tolist(),
    maxlen = fr_max_len,
    padding = 'post')
# Add them to our training data frame
train_df['pad_fr_ids'] = padded_fr_ids.tolist()
```

Again, we can check the result by, for instance, running `print(train_df['pad_fr_ids'])`:

```
0       [199, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...
1       [992, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...
2       [992, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...
3       [18, 812, 5, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
4       [18, 812, 49, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
			      ...                        
9995    [118, 22, 203, 1143, 3, 0, 0, 0, 0, 0, 0, 0, 0...
9996    [6, 221, 265, 5, 66, 20, 5, 765, 9, 0, 0, 0, 0...
9997    [118, 5, 58, 169, 361, 9, 0, 0, 0, 0, 0, 0, 0,...
9998    [108, 49, 58, 169, 361, 9, 0, 0, 0, 0, 0, 0, 0...
9999    [108, 49, 58, 169, 375, 9, 0, 0, 0, 0, 0, 0, 0...
Name: pad_fr_ids, Length: 10000, dtype: object
```

As expected, all sentences have been padded appropriately.

Before creating and training our NMT model, we have a bit of formatting to do with our data. We need to split the dataset into a training set and a test set, and turn both into the expected input of our network: numpy arrays.

First, let's give our dataset a little shuffle to mix the sentences around:

```python
# Shuffling our data frame around
train_df = train_df.sample(frac=1).reset_index(drop=True)
```

Then we can split our dataset. We will do a 9,000/1,000 split between the training and test sets. The training input data `trainX` is going to be 9,000 random encoded (tokenized) French sentences, and the expected outputs for this training set is going to be the 9,000 corresponding encoded English sentences of `trainY`.

Similarly, our validation data will be the 1,000 leftover encoded French sentences of `testX`, and their expected encoded English outputs from `testY`.

```python
# Create our training input and target output numpy array to feed to our NMT model
# We'll take the first train_size lines for training (after random shuffle)
trainX = np.asarray(train_df['pad_fr_ids'][0:train_size].tolist())
trainY = np.asarray(train_df['pad_en_ids'][0:train_size].tolist())

# The test dataset for checking on the last test_size lines (after random shuffle)
testX = np.asarray(train_df['pad_fr_ids'][train_size:].tolist())
testY = np.asarray(train_df['pad_en_ids'][train_size:].tolist())
```

The last preprocessing step is to take care of the dimensionality mismatch between `trainY` and `testY`, and the output of our neural network. The network outputs a 3D numpy array, while our `trainY` and `testY` are 2D arrays.

We can check the dimension by running `print(trainY.shape)` which returns (in my case) `(9000, 10)`, because we have 9,000 sentences padded to the maximum length of 10. This is indeed a 2D array.

To take care of this, we simply need to reshape our array into a 3D where the 3rd dimension is 1. This does not change our data at all, it is simply a matter of format. Here is how to do it:

```python
# Reshape the output to match expected dimensionality
trainY = trainY.reshape(trainY.shape[0], trainY.shape[1], 1)
testY = testY.reshape(testY.shape[0], testY.shape[1], 1)
```

Running `print(trainY.shape)` should now output `(9000, 10, 1)`.

When this is done, our training and test data are ready to be fed to our network, which we will now build.

## Creating the NMT model

We are now ready to start building our model. We will be using Keras with a TensorFlow back-end for this.

The model itself is very simple. At the core are the encoder and decoder we have talked about in the introduction. Their roles are respectively to turn the input text (or more precisely its tokenized version) into a single vector representing the features of the input sentence (French in our case), and to turn this feature vector back into a tokenized sentence in the output target language (English in our case). To keep the model simple, the encoder and decoder will be a single LSTM layer each.

To get the machinery running, we will need a few extra components: an embedding layer as the first layer, a repeat vector layer between our encoder and decoder, and a dense layer as the last layer.

First, this model is going to be a sequential model from Keras. Let's import all the layers we are using, and create our sequential model.

```python
# Neural network components from Keras
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import Dense

model = Sequential()
```

The embedding layer transforms our "one hot" encoding of subwords in the vocabulary list of dimension `fr_vocab_size` into a vector mapping our subwords in a space of `nb_cells` dimensions. This means that our encoding of subwords is more efficient, saving us precious computing power. Moreover, the embedding layer will try to learn similarities between subwords and arrange them closer together in the output space. The `mask_zero = True` option allows us to consider `0` as an empty or unknown subword for padding for instance, and not as a real word to take into consideration. We need it as this is how both SentencePiece and our padding work.

```python
model.add(Embedding(
    fr_vocab_size,
    nb_cells,
    input_length = fr_max_len,
    mask_zero = True))
```

After this first embedding layer is the encoder LSTM which returns a vector of features. We do not want to return the sequences as our goal here is to get a single final vector of features.

```python
model.add(LSTM(
    units = nb_cells,
    return_sequences = False))
```

The feature vector we have just obtained is a 2D object, but our decoder LSTM expects a 3D object. To solve this problem, we use a repeat vector layer. This layer will simply repeat the 2D vector multiple times to match the expected input of our decoder LSTM.

```python
model.add(RepeatVector(en_max_len))
```

Then comes the decoder layer, just like the encoder.

```python
model.add(LSTM(
    units = nb_cells,
    return_sequences = True))
```

Finally, after our decoder is a dense layer. Because of the embedding, our subwords are represented by vectors in a space of dimension `nb_cells`. We need to convert them back to a "one hot" encoding in our vocabulary space. This operation is a classification problem, usually handled by using a softmax.

```python
model.add((Dense(
    en_vocab_size,
    activation = 'softmax')))
```

Our model is ready, it's time to compile it! We will be using the `adam` optimizer and `sparse_categorical_entropy` as our loss function, but just like other hyper-parameters in this article, this is up to you. Feel free to play around and optimize as you wish.

```python
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy')
```

Here is the complete code of the model:

```python
# Neural network components from Keras
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import Dense

# Defining all the parameters of our network
nb_cells = 512 # LSTM cells in encoder/decoder

# Creating a Keras Sequential object for our NMT model
model = Sequential()

# Embedding layer to map our one-hot encoding to a small word space
model.add(Embedding(
    fr_vocab_size,
    nb_cells,
    input_length = fr_max_len,
    mask_zero = True))
# Adding an LSTM layer to act as the encoder
model.add(LSTM(
    units = nb_cells,
    return_sequences = False))
# Since we are not returning a sequence but just a vector, we need
# to repeat this vector multiple times to input it to our decoder LSTM
model.add(RepeatVector(en_max_len))
# Adding an LSTM layer to act as the decoder
model.add(LSTM(
    units = nb_cells,
    return_sequences = True))
# Adding a softmax
model.add((Dense(
    en_vocab_size,
    activation = 'softmax')))

# Compiling the model
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy')
```

The command `print(model.summary())` can also be used to get a visual representation of our model.

```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, 22, 256)           512000    
_________________________________________________________________
lstm_3 (LSTM)                (None, 256)               525312    
_________________________________________________________________
repeat_vector_2 (RepeatVecto (None, 10, 256)           0         
_________________________________________________________________
lstm_4 (LSTM)                (None, 10, 256)           525312    
_________________________________________________________________
dense_2 (Dense)              (None, 10, 2000)          514000    
=================================================================
Total params: 2,076,624
Trainable params: 2,076,624
Non-trainable params: 0
_________________________________________________________________
None
```

Everything is ready for the training.

```python
# Training parameters
nb_epochs = 15
batch_size = 64
model_filename = 'model.h5'
checkpoint = ModelCheckpoint(
    model_filename,
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    mode = 'min')

# Training the model
model.fit(
    trainX,
    trainY,
    epochs = nb_epochs,
    batch_size = batch_size,
    callbacks = [checkpoint],
    validation_data = (testX, testY))
```

This can take about 30 minutes. Go make some tea, and see you in a moment.

## Testing the model on examples

Now that our model has been trained, we can load the best set of weights (our fittest version of the model!) that has been saved during the training, and check how good it has become at translating French to English on a few sentences of our test set.

```python
# Let's load the trained model
model = load_model(model_filename)

predictions = model.predict_classes(testX)

# Check the translation on a few sentences
decoded_predictions = []
for index in range(10):
    print("Original:")
    print(fr_sp.DecodeIds(testX[index, :].tolist()))
    print("Expected:")
    print(en_sp.DecodeIds(testY[index, :, 0].tolist()))
    print("Predicted:")
    print(en_sp.DecodeIds(predictions[index, :].tolist()))
    print("")
```

I am checking them on the first 10 sentences of the test set, but feel free to take more, or even test the model on our own sentences.

The output should look something like that:

```
Original:
Je suis presque mort. ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇ 
Expected:
I almost died. ⁇  ⁇  ⁇  ⁇  ⁇ 
Predicted:
I'm dead. ⁇  ⁇  ⁇  ⁇  ⁇ 
```

For the sake of simplicity, I've cleaned up the "unknown" or padding characters and compiled the test results into a table.

French | Expected English | Predicted English
-|-|-
Je suis presque mort.|I almost died.|I'm dead.
Es-tu sûre ?|Are you sure?|Are you sure?
Soyez de nouveau le bienvenu !|Welcome back.|Go him.
Allez le chercher !|Go get it.|Go take it it
C'est à moi.|That is mine.|It's mine.
Arrête de pleurer !|Stop crying.|Stop crying.
Est-ce que Tom va bien ?|Is Tom well?|Is Tom OK?
Permettez-moi d'essayer.|Let me try.|Allow me me me.
Je me dépêcherai.|I'll hurry.|I'll manage.
Veuillez venir.|Please come.|Please come.

We can see that the results are far from perfect, even on such short sentences. However a good handful of translations are correct or close to the intended meaning. This very simple model we have created today clearly illustrates the potential of neural machine translation systems.

## Conclusion

I hope this tutorial has been useful to you in taking a first look at NMT systems. Our example was very much a toy given its basic design and simple sentences of our dataset, but I believe it is a nice illustration of the core architecture of NMT systems and what they can accomplish.

There are many possibilities to improve the NMT system we have created today, such as adding extra LSTM layers to our encoder and decoder, adding an [attention network][attention], or using a more realistic dataset like the [Europarl][euro] parallel corpus, in order to translate longer and more complex text with higher accuracy.


[notebook]: https://github.com/lkuenemann/mllab/blob/master/nmt.ipynb
[nmt-from-scratch]: https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/
[a-star-nmt]: https://www.codeastar.com/nmt-make-an-easy-neural-machine-translator/
[sp]: https://github.com/google/sentencepiece
[euro]: http://www.statmt.org/europarl/
[mt.org]: http://www.manythings.org/anki/
[mt.org-down]: http://www.manythings.org/anki/fra-eng.zip
[tato]: https://tatoeba.org/eng
[anki]: https://apps.ankiweb.net/
[attention]: https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
