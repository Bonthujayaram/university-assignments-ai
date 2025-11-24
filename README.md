# Hindi Transliteration using Seq2Seq RNN (Aksharantar – Hindi)

This repository contains my solution for the Deep Learning assignment based on the **Aksharantar** dataset released by AI4Bharat. The task is to build a **character-level sequence-to-sequence (seq2seq)** model that transliterates **romanized Hindi** (Latin script) into **Devanagari** script, and to derive the **total computations** and **total number of parameters** of the model.

---

## 1. Assignment Overview

### Problem
Given a dataset of word pairs of the form:

- `src` → romanized Hindi (e.g., `ghar`)
- `tgt` → Devanagari Hindi (e.g., `घर`)

the goal is to train a model:

\[
f_\theta: \text{Latin char sequence} \rightarrow \text{Devanagari char sequence}
\]

This is a **character-level transliteration** problem and can be seen as a simplified version of machine translation.

---

## 2. Repository Contents

- `aksharantar_hin_seq2seq.ipynb`  
  Main Google Colab notebook with:
  - Data loading from Google Drive
  - Character vocabulary creation
  - Encoder–Decoder (Seq2Seq) model using GRU
  - Training loop
  - Inference (transliteration examples)
  - Simple character-level accuracy evaluation

- `Transliteration_Report.pdf`  
  Short report summarizing problem, model, results, and formulas.

(Note: The actual Aksharantar dataset is **not included** in this repo. It is mounted from Google Drive in Colab.)

---

## 3. Question 1 – Seq2Seq Model Implementation

### 3.1 Dataset Used

- **Dataset**: Aksharantar Hindi subset (`hin_train.csv`)
- **Location in Drive**: `/content/drive/MyDrive/aksharantar_sampled/hin/hin_train.csv`
- The original file has **no header row**, so I manually assign column names:
  - `src` – romanized Hindi word (Latin script)
  - `tgt` – Hindi word in Devanagari script

Example rows:

| src          | tgt         |
|-------------|------------|
| shastragaar | शस्त्रागार |
| bindhya     | बिन्द्या    |
| karanje     | करंजे      |

I use **6000** examples for training.

---

### 3.2 Character-Level Vocabulary

I work at **character level**:

- Build separate vocabularies for:
  - Source characters (Latin)
  - Target characters (Devanagari)
- Add 4 special tokens:
  - `<pad>` – padding
  - `<sos>` – start of sequence
  - `<eos>` – end of sequence
  - `<unk>` – unknown character

Each character is mapped to an integer index using `stoi` (string-to-index) and `itos` (index-to-string) dictionaries.

---

### 3.3 Model Architecture

The model follows the **Encoder–Decoder (Seq2Seq)** design.

#### 3.3.1 Input Layer: Character Embeddings

- Each character (src or tgt) is represented by a **learned embedding vector**.
- Embedding dimension used in experiments:  
  \[
  d = 128
  \]

- Two separate embedding layers are used:
  - Source embedding: `SRC_VOCAB_SIZE × d`
  - Target embedding: `TGT_VOCAB_SIZE × d`

---

#### 3.3.2 Encoder RNN

The encoder reads the **romanized input word** character by character and converts it into a fixed-length hidden representation.

- RNN cell: **GRU** (Gated Recurrent Unit)
- Hidden size:
  \[
  h = 256
  \]
- Number of layers: `1`
- Direction: unidirectional

Processing:

1. Input characters → source embeddings
2. Embeddings → GRU
3. Final encoder hidden state = summary of the entire input sequence
4. This hidden state is passed to the decoder as its initial hidden state

The encoder uses `pack_padded_sequence` to handle batches with different sequence lengths.

---

#### 3.3.3 Decoder RNN

The decoder generates the **Devanagari output word**, one character at a time.

- RNN cell: GRU (same hidden size `h = 256`)
- Input: previous target character (embedding) + previous hidden state
- Output: logits over target vocabulary

Steps:

1. Decoder starts with the `<sos>` token.
2. At time step \( t \), the decoder:
   - Takes input: previous character (`ground truth` or `prediction`)
   - Updates hidden state via GRU
   - Passes the hidden state through a linear layer to get logits over `TGT_VOCAB_SIZE`
   - Applies softmax to get a probability distribution over the next character.

---

#### 3.3.4 Teacher Forcing

- During training, **teacher forcing** is used with probability 0.5:
  - With probability 0.5, the next input to the decoder is the **true** previous character.
  - With probability 0.5, the next input is the **predicted** previous character.

This helps the model converge faster and improves stability.

---

#### 3.3.5 Training Setup

- Loss function: **Cross-Entropy Loss**
  - Ignores `<pad>` tokens using `ignore_index`.
- Optimizer: **Adam**
  - Learning rate: `0.001`
- Batch size: `64`
- Epochs: `20`
- Gradient clipping: `1.0` (to avoid exploding gradients)

---

### 3.4 Results for Question 1

#### Training Loss

- Initial loss: **3.0970**
- Final loss after 20 epochs: **0.3817**

The steady decrease in loss shows that the model is learning the mapping well.

#### Sample Transliteration Outputs

Some qualitative examples from the trained model:

- `santoshani` → `संतोषनी`
- `rangari` → `रंगारी`
- `sudharanyasathiche` → `सुधारण्यासाठीचे`
- `sugauli` → `सुगुली`
- `harmohinder` → `हरमोहिन्दर`
- `styling` → `स्टायलिंग`

Most outputs are correct or very close to the expected Devanagari words, demonstrating that the seq2seq model has successfully learned the transliteration task.

#### Evaluation: Character-Level Accuracy

I compute a simple character-level accuracy on the first 500 samples of the training data by comparing predicted and true characters:

- **Character-level accuracy:** **88.59%**

---

## 4. Question 2 – Total Number of Computations

Here I derive the **total number of computations** (dominant multiplications) done by the network in a forward pass.

### Assumptions

- Embedding size: \( d \)
- Hidden size (encoder and decoder): \( h \)
- Sequence length (input and output): \( T \)
- Vocabulary size (source and target): \( V \)
- 1 encoder layer, 1 decoder layer
- Simple RNN cell for analysis (GRU/LSTM just add constant factors)
- Separate source and target embeddings

---

### 4.1 Encoder Computations

At each time step \( t \), the encoder computes:

\[
h_t = \phi(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
\]

- Multiplications for \( W_{xh} x_t \): \( d \times h \)
- Multiplications for \( W_{hh} h_{t-1} \): \( h \times h \)

So, per time step:

\[
\text{Encoder ops per step} = d h + h^2
\]

Over \( T \) time steps:

\[
\text{Encoder ops} \approx T (d h + h^2)
\]

---

### 4.2 Decoder Computations

The decoder has the same kind of RNN cell, so its recurrent part also costs:

\[
\text{Decoder RNN ops} \approx T (d h + h^2)
\]

Additionally, at each decoder time step, we have the **output layer**:

- Logits:
  \[
  o_t = W_{ho} s_t + b_o
  \]
- Multiplications: \( h \times V \)

So the output layer costs:

\[
\text{Decoder output ops} \approx T (hV)
\]

---

### 4.3 Total Computations

Adding encoder, decoder RNN, and decoder output layer:

\[
\text{Total ops} \approx T (d h + h^2) + T (d h + h^2) + T (hV)
\]

\[
\boxed{
\text{Total ops} \approx T \big( 2(dh + h^2) + hV \big)
}
\]

In big-O notation:

\[
O\big(T (dh + h^2 + hV)\big)
\]

---

## 5. Question 3 – Total Number of Parameters

Now I compute the **total number of trainable parameters** in the network.

### Assumptions (same as above)

- Embedding size: \( d \)
- Hidden size: \( h \)
- Vocabulary size: \( V \)
- 1 encoder layer, 1 decoder layer
- Different embeddings for source and target
- Simple RNN cell for parameter counting

---

### 5.1 Embedding Layer Parameters

- Source embeddings: \( V \times d \)
- Target embeddings: \( V \times d \)

Total embedding parameters:

\[
2 V d
\]

---

### 5.2 Encoder RNN Parameters

For a vanilla RNN cell:

- \( W_{xh} \): \( d \times h \) → \( d h \)
- \( W_{hh} \): \( h \times h \) → \( h^2 \)
- Bias \( b_h \): size \( h \)

Total encoder params:

\[
d h + h^2 + h
\]

---

### 5.3 Decoder RNN Parameters

Decoder has the same structure:

\[
d h + h^2 + h
\]

---

### 5.4 Output Layer Parameters

- \( W_{ho} \): \( h \times V \) → \( h V \)
- \( b_o \): \( V \) parameters

Total output params:

\[
hV + V
\]

---

### 5.5 Total Parameters

Adding everything together:

\[
\text{Total params} =
2Vd
+ (d h + h^2 + h)
+ (d h + h^2 + h)
+ (hV + V)
\]

\[
\boxed{
\text{Total params}
= 2Vd + 2dh + 2h^2 + 2h + hV + V
}
\]

This can also be factored:

\[
\boxed{
\text{Total params} = V(2d + h + 1) + 2h^2 + 2dh + 2h
}
\]

For GRU or LSTM, each recurrent layer’s parameters are multiplied by a constant factor (due to gates), but the overall form of the expression remains the same.

---

## 6. How to Run (Colab)

1. Open the notebook: `aksharantar_hin_seq2seq.ipynb` in Google Colab.
2. Make sure the Aksharantar Hindi dataset is available in your Drive at:  
   `/content/drive/MyDrive/aksharantar_sampled/hin/hin_train.csv`
3. Run all cells in order:
   - Mount Drive
   - Load dataset
   - Build vocabularies
   - Create dataset and dataloader
   - Define encoder, decoder, seq2seq model
   - Train for 20 epochs
   - Run inference and evaluation

---

## 7. Future Work

- Add **attention mechanism** (Bahdanau or Luong) to improve transliteration quality.
- Use **beam search** during decoding instead of greedy argmax.
- Train on a larger subset or all of Aksharantar (multiple languages).
- Experiment with **deeper models** (more encoder/decoder layers).
- Evaluate with **BLEU score** or sequence-level accuracy.

---

## 8. Acknowledgements

- Aksharantar dataset by **AI4Bharat**.
- Implemented and trained in **Google Colab** using **PyTorch**.
