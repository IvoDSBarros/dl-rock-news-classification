'''
Rock News NLP
Multi-Label Classification Leveraging Deep Learning
CNN-BiLSTM with GloVe 300D and Attention with Optimized Threshold Tuning
Created on Mon Jun 16 15:32:27 2025
@author: IvoBarros
'''
import pandas as pd
import numpy as np
import os
import sys
from time import time
import pickle
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Input, LSTM, Dropout, Bidirectional, BatchNormalization, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
import rock_news_nlp_multi_label_utilities as utils_multi_label

def micro_f1_tf(y_true, y_pred):
    """
    To compute the micro-averaged F1-score using TensorFlow.
       
    Args:
        y_true : True binary labels (TensorFlow/Keras Tensor)
        y_pred : Predicted probabilities from the model (TensorFlow/Keras Tensor)

    Returns:
        float : The micro-averaged F1-score 
    """
    y_pred_thresh = K.cast(K.greater(y_pred, 0.30), 'float32')    
    tp = K.sum(y_true * y_pred_thresh)
    fp = K.sum((1 - y_true) * y_pred_thresh)
    fn = K.sum(y_true * (1 - y_pred_thresh))
    
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    return f1

def macro_f1_tf(y_true, y_pred):
    """
    To compute the macro-averaged F1-score using TensorFlow.

    Args:
        y_true : True binary labels (TensorFlow/Keras Tensor)
        y_pred : Predicted probabilities from the model (TensorFlow/Keras Tensor)
    
    Returns:
        float : The macro-averaged F1-score
    """    
    y_pred_thresh = K.cast(K.greater(y_pred, 0.30), 'float32')
    f1s = []

    num_labels_int = K.int_shape(y_true)[1]
    if num_labels_int is None:
        num_labels_int = tf.shape(y_true)[1]

    for i in range(num_labels_int):
        tp = K.sum(y_true[:, i] * y_pred_thresh[:, i])
        fp = K.sum((1 - y_true[:, i]) * y_pred_thresh[:, i])
        fn = K.sum(y_true[:, i] * (1 - y_pred_thresh[:, i]))
        
        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())
        f1 = 2 * p * r / (p + r + K.epsilon())
        f1s.append(f1)
        
    return K.mean(K.stack(f1s))

def exact_match_accuracy(y_true, y_pred):
    """
    To compute the exact match ratio (subset accuracy) for multi-label classification.

    Args:
        y_true : True binary labels (TensorFlow/Keras Tensor)
        y_pred : Predicted probabilities from the model (TensorFlow/Keras Tensor)

    Returns:
        float: The exact match accuracy.
    """
    y_pred_thresh = K.cast(K.greater(y_pred, 0.30), 'float32')
    return K.mean(K.cast(K.all(K.equal(y_true, y_pred_thresh), axis=-1), 'float32'))


## MAIN SCRIPT EXECUTION
def main():
    """
    Main function to train, evaluate and save the CNN-BiLSTM multi-label text
    classification model.
    """
    print("The training script is running...")
    t_start = time()

    ## SET UP PATHS AND DIRECTORIES
    current_working_dir = os.getcwd()
    path_data = os.path.join(current_working_dir, 'data')
    path_model_assets = os.path.join(current_working_dir, 'model_assets')

    os.makedirs(path_model_assets, exist_ok=True)
    print(f"Model assets will be saved in: {path_model_assets}")

    ## LOAD DATA
    print("Loading data...")
    try:
        df_multi_label = pd.read_csv(os.path.join(path_data, 'rock_news_multi_label_dataset.csv'), sep=';')
    except FileNotFoundError:
        print(f"Error: Dataset not found at {os.path.join(path_data, 'rock_news_multi_label_dataset.csv')}. Please check the 'data' folder.")
        sys.exit(1) # Exit with an error code

    ## DATA PREPARATION
    print("Performing multi-label data stratification...")
    feature = 'title_clean'
    feature_ref = 'full_pk'
    labels = [col for col in df_multi_label.columns if col not in ['title', feature, feature_ref]]
    test_size = 0.2
    
    X_train, y_train, X_test, y_test, _, _, _ = utils_multi_label.custom_multilabel_data_strat(
        df_multi_label, feature, feature_ref, labels, test_size)
    
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    print(f"Train/Test split complete. Number of labels: {len(labels)}")

    ## SAVE LABELS LIST
    labels_file_path = os.path.join(path_model_assets, 'labels.pkl')
    with open(labels_file_path, 'wb') as f:
        pickle.dump(labels, f)
    print(f"Labels list saved to {labels_file_path}")

    ## MODEL HYPERPARAMETERS
    num_words = 10000        
    max_len = 250
    embedding_dim = 300
    num_epochs = 30
    batch_size = 64

    ## TOKENIZATION AND PADDING
    print("Initializing tokenizer and preparing sequences...")
    tokenizer = Tokenizer(num_words=num_words, oov_token="<unk>")
    tokenizer.fit_on_texts(X_train)

    vocab_size = len(tokenizer.word_index) + 1

    ## SAVE TOKENIZER
    tokenizer_save_path = os.path.join(path_model_assets, 'tokenizer.pkl')
    with open(tokenizer_save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to {tokenizer_save_path}")

    ## GENERATE NUMERICAL SEQUENCES
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    ## GENERATE PAD SEQUENCES
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    print(f'X_train shape after padding: {X_train.shape}')
    print(f'X_test shape after padding: {X_test.shape}')

    ## LOAD PRE-TRAINED WORD EMBEDDINGS
    print("Loading GloVe embeddings...")
    embedding_index = {}
    glove_path = os.path.join(path_data, 'glove.6B.300d.txt')

    if not os.path.exists(glove_path):
        print(f"Error: GloVe file not found at {glove_path}. Please download it and place it in the 'data' folder.")
        sys.exit(1)

    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    ## CREATE EMBEDDING MATRIX
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("GloVe embeddings loaded and embedding matrix created.")

    ## DEFINE THE MODEL WITH ATTENTION
    print("Defining the model architecture...")
    input_layer = Input(shape=(max_len,))
    
    ## EMBEDDING LAYER USING GLOVE WEIGHTS
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                weights=[embedding_matrix], trainable=True)(input_layer)

    ## CNN LAYER (FEATURE EXTRACTION)
    conv_layer = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(embedding_layer)
    norm_layer = BatchNormalization()(conv_layer)

    ## BILSTM LAYER (SEQUENTIOAL CONTEXT)
    lstm_output = Bidirectional(LSTM(128, return_sequences=True))(norm_layer)

    ## ATTENTION LAYER
    attention_output = Attention()([lstm_output, lstm_output])

    ## POOLING AND DENSE LAYERS
    global_max_pooling = GlobalMaxPooling1D()(attention_output)
    dense_1 = Dense(64, activation='relu')(global_max_pooling)
    dropout_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(64, activation='relu')(dropout_1)
    dropout_2 = Dropout(0.5)(dense_2)
    
    ## OUTPUT LAYER WITH SIGMOID ACTIVATION
    output_layer = Dense(len(labels), activation='sigmoid')(dropout_2)

    model = Model(inputs=input_layer, outputs=output_layer)

    ## MODEL COMPILATION
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[
                      micro_f1_tf,           
                      macro_f1_tf,           
                      BinaryAccuracy(threshold=0.30, name='binary_accuracy'),
                      Precision(thresholds=0.30, name='precision'),
                      Recall(thresholds=0.30, name='recall'),
                      exact_match_accuracy
                  ])
    print("Model compiled successfully.")

    model.summary()

    ## CALLBACKS FOR TRAINING
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model_checkpoint = ModelCheckpoint(
        os.path.join(path_model_assets, 'best_model_with_attention_functional.h5'),
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    print("Training callbacks initialized.")

    ## FIT THE MODEL
    print("Starting model training...")
    history = model.fit(X_train, y_train,
                        epochs=num_epochs,
                        validation_data=(X_test, y_test),
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=[early_stopping, model_checkpoint])
    print("Model training finished.")

    ## TRAINING HISTORY
    metrics_df = pd.DataFrame(history.history)
    metrics_df['epoch'] = metrics_df.index + 1

    output_csv_filename = 'training_history_metrics.csv'
    metrics_df.to_csv(os.path.join(path_data, output_csv_filename),
                      header=True,
                      index=False,
                      encoding='utf-8',
                      sep=';')
    print(f"Training history saved to {os.path.join(path_data, output_csv_filename)}")

    ## MODEL EVALUATION: TEST SET
    print("\nEvaluating model on the test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.30).astype(int)

    print("\nTest Set Evaluation Results with Optimal Threshold (0.30):")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=labels, zero_division=0))
    print("Micro-average F1 Score:", f1_score(y_test, y_pred, average='micro'))
    print("Macro-average F1 Score:", f1_score(y_test, y_pred, average='macro'))
    print("Micro-average Precision:", precision_score(y_test, y_pred, average='micro', zero_division=0))
    print("Micro-average Recall:", recall_score(y_test, y_pred, average='micro', zero_division=0))
    print("Macro-average Precision:", precision_score(y_test, y_pred, average='macro', zero_division=0))
    print("Macro-average Recall:", recall_score(y_test, y_pred, average='macro', zero_division=0))

    exact_match_test = np.all(y_test == y_pred, axis=1).mean()
    print("Test Set Exact Match Accuracy:", exact_match_test)

    total_time = time() - t_start
    print(f"\n...Script successfully executed in {total_time:.1f} seconds.")

if __name__ == "__main__":
    main()