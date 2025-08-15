'''
Rock News NLP
Multi-Label Classification Leveraging Deep Learning
Fine-tuning Pre-trained Transformers: Distilbert
Created on Mon Jun 17 17:04:18 2025
@author: IvoBarros
'''
import pandas as pd
import numpy as np
import os
from time import time
import sys
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
import rock_news_nlp_multi_label_utilities as utils_multi_label

def tokenize_data(texts, tokenizer_obj, max_len):
    """
    To tokenize a list of texts for a Transformer model.

    Args:
        texts : list of str
        tokenizer_obj : AutoTokenizer
        max_len : int

    Returns:
        tuple : input_ids_tensor, attention_masks_tensor
    """
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer_obj.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf', # Return TensorFlow tensors
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return tf.concat(input_ids, axis=0), tf.concat(attention_masks, axis=0)

def micro_f1_tf(y_true, y_pred):
    """
    To compute the micro-averaged F1-score using TensorFlow.
       
    Args:
        y_true : True binary labels (TensorFlow/Keras Tensor)
        y_pred : Predicted probabilities from the model (TensorFlow/Keras Tensor)

    Returns:
        float : The micro-averaged F1-score 
    """
    y_pred_thresh = K.cast(K.greater(y_pred, 0.50), 'float32')    
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
    y_pred_thresh = K.cast(K.greater(y_pred, 0.50), 'float32')
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
    y_pred_thresh = K.cast(K.greater(y_pred, 0.50), 'float32')
    return K.mean(K.cast(K.all(K.equal(y_true, y_pred_thresh), axis=-1), 'float32'))

print("Starting training script...")
t_start = time()

## SET UP PATHS AND DIRECTORIES
BASE_PATH = os.getcwd()
DATA_PATH = os.path.join(BASE_PATH, 'data')
MODEL_ASSETS_PATH = os.path.join(BASE_PATH, 'model_assets')
os.makedirs(MODEL_ASSETS_PATH, exist_ok=True)
data_file_path = os.path.join(DATA_PATH, 'rock_news_multi_label_dataset.csv')
df_multi_label = pd.read_csv(data_file_path, sep=';')

## TRANSFORMER MODEL CONFIGURATION
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10
N_SPLITS = 5

## DATA PREPARATION
feature = 'title_clean'
feature_ref = 'full_pk'
labels = [col for col in df_multi_label.columns if col not in ['title', feature, feature_ref]]
test_size = 0.2

X_train_raw, y_train, X_test_raw, y_test, _, _, _ = \
    utils_multi_label.custom_multilabel_data_strat(df_multi_label, feature, feature_ref, labels, test_size)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

all_text_data = X_train_raw + X_test_raw
all_labels = np.vstack((y_train, y_test))

print('Train/Test split complete.')
print('\n--- Data Diagnostics ---')
print('X_train_text (samples):', len(X_train_raw))
print('X_test_text (samples):', len(X_test_raw))
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print('Number of labels identified:', len(labels))

## INITIALIZE TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
all_input_ids, all_attention_masks = tokenize_data(all_text_data, tokenizer, MAX_LEN)

## K-FOLD CROSS-VALIDATION FOR TRANSFORMER MODEL
print(f"\nStarting K-Fold Cross-Validation with {N_SPLITS} folds for {MODEL_NAME}...")
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

all_histories = []
fold_macro_f1_scores = []
fold_micro_f1_scores = []
fold_classification_reports = []

num_labels = len(labels)
tokenizer_saved = False

for fold, (train_index, test_index) in enumerate(kf.split(all_input_ids, all_labels)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

    ## DATA PREPARATION FOR CURRENT FOLD
    train_input_ids = tf.constant(all_input_ids.numpy()[train_index])
    train_attention_masks = tf.constant(all_attention_masks.numpy()[train_index])
    train_labels = all_labels[train_index]

    val_input_ids = tf.constant(all_input_ids.numpy()[test_index])
    val_attention_masks = tf.constant(all_attention_masks.numpy()[test_index])
    val_labels = all_labels[test_index]

    ## LOAD A FRESH MODEL
    print(f"Loading fresh model {MODEL_NAME} for Fold {fold+1}...")
    model = TFAutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    print("Model loaded successfully.")

    ## SET OPTIMIZER AND LOSS FUNCTION
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True) # `from_logits=True` because TFAutoModel outputs logits

    ## MODEL COMPILATION
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            micro_f1_tf,
            macro_f1_tf,
            BinaryAccuracy(threshold=0.5, name='binary_accuracy'),
            Precision(thresholds=0.5, name='precision'),
            Recall(thresholds=0.5, name='recall'),
            exact_match_accuracy
        ]
    )

    ## SET EARLYSTOPPING CALLBACK
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3, 
        restore_best_weights=True,
        mode='min'
    )
    
    print(f"Training model for Fold {fold+1}...")
    history = model.fit(
        x=[train_input_ids, train_attention_masks],
        y=train_labels,
        validation_data=([val_input_ids, val_attention_masks], val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )
    print(f"Model training complete for Fold {fold+1}.")

    ## SAVE FOLD BEST MODEL
    fold_model_save_path = os.path.join(MODEL_ASSETS_PATH, f'multilabel_fold_{fold+1}_distilbert')
    print(f"Saving best model for Fold {fold+1} to {fold_model_save_path}...")
    model.save_pretrained(fold_model_save_path)
    print("Model saved successfully in Hugging Face format.")

    ## SAVE TOKENIZER ONCE
    if not tokenizer_saved:
        tokenizer_save_path = os.path.join(MODEL_ASSETS_PATH, 'tokenizer_distilbert')
        print(f"Saving tokenizer to {tokenizer_save_path}...")
        tokenizer.save_pretrained(tokenizer_save_path)
        print("Tokenizer saved successfully.")
        tokenizer_saved = True

    ## TRAINING HISTORY RECORDING
    fold_history_df = pd.DataFrame(history.history)
    fold_history_df['fold'] = fold + 1
    fold_history_df['epoch'] = fold_history_df.index + 1
    all_histories.append(fold_history_df)

    ## MODEL EVALUATION ON VALIDATION SET
    print(f"Evaluating model for Fold {fold+1}...")
    val_preds_logits = model.predict([val_input_ids, val_attention_masks]).logits
    val_preds_proba = tf.sigmoid(val_preds_logits).numpy()
    val_preds_binary = (val_preds_proba > 0.5).astype(int)

    ## F1 SCORES CALCULATION
    macro_f1 = f1_score(val_labels, val_preds_binary, average='macro', zero_division=0)
    micro_f1 = f1_score(val_labels, val_preds_binary, average='micro', zero_division=0)

    fold_macro_f1_scores.append(macro_f1)
    fold_micro_f1_scores.append(micro_f1)

    print(f"Fold {fold+1} Validation Macro-F1: {macro_f1:.4f}")
    print(f"Fold {fold+1} Validation Micro-F1: {micro_f1:.4f}")

    current_report = classification_report(val_labels, val_preds_binary, target_names=labels, zero_division=0)
    print("Fold Classification Report:\n", current_report)
    fold_classification_reports.append(current_report)

## AGGREGATING FINAL RESULTS
if all_histories:
    combined_metrics_df = pd.concat(all_histories, ignore_index=True)
    print("\nCombined Training Metrics across all Folds (DistilBERT):\n", combined_metrics_df)

    output_csv_filename = 'training_history_metrics_distilbert.csv'
    combined_metrics_df.to_csv(os.path.join(MODEL_ASSETS_PATH, output_csv_filename),
                               header=True,
                               index=False,
                               encoding='utf-8',
                               sep=';')
    print(f"Combined training history saved to {os.path.join(MODEL_ASSETS_PATH, output_csv_filename)}")
else:
    print("No training history collected. Ensure K-Fold loop executed.")

## SAVING LABELS LIST FOR INFERENCE
labels_save_path = os.path.join(MODEL_ASSETS_PATH, 'labels_distilbert.pkl')
try:
    with open(labels_save_path, 'wb') as f:
        pickle.dump(labels, f)
    print(f"Labels list saved to {labels_save_path}")
except Exception as e:
    print(f"Error saving labels list: {e}")

print("\n--- K-Fold Cross-Validation Complete ---")
print(f"Average Macro-F1 across {N_SPLITS} folds: {np.mean(fold_macro_f1_scores):.4f} (+/- {np.std(fold_macro_f1_scores):.4f})")
print(f"Average Micro-F1 across {N_SPLITS} folds: {np.mean(fold_micro_f1_scores):.4f} (+/- {np.std(fold_micro_f1_scores):.4f})")

print("\nIndividual Fold Reports:")
for i, report in enumerate(fold_classification_reports):
    print(f"\n--- Fold {i+1} Report ---")
    print(report)

t_end = time()
print(f"\nTotal script execution time: {(t_end - t_start):.1f}s.")