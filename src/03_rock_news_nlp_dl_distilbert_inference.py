'''
Rock News NLP
Multi-Label Classification Leveraging Deep Learning
Using a fine-tuned DistilBERT model for inference
Created on Sat Aug 23 16:45:12 2025
@author: IvoBarros
'''
import tensorflow as tf
import pickle
import os
import pandas as pd
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

## SET UP PATHS AND DIRECTORIES
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'assets', 'distilbert')
MODEL_FOLDER_PATH = os.path.join(ASSETS_DIR, 'multilabel_fold_5_distilbert')
TOKENIZER_FOLDER_PATH = os.path.join(ASSETS_DIR, 'tokenizer_distilbert')
LABELS_PATH = os.path.join(ASSETS_DIR, 'labels_distilbert.pkl')

## LOAD DATA (MODEL, TOKENIZER AND LABELS)
try:
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FOLDER_PATH)
    
    with open(LABELS_PATH, 'rb') as f:
        labels = pickle.load(f)

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please check your file paths. The script could not find a required file or folder.")
    
else:
    ## UNSEEN ROCK NEWS HEADLINES
    new_headlines = [
        "AC/DC's new single rocks the charts with a classic sound.",
        "Led Zeppelin reunion rumors swirl after cryptic social media post.",
        "Metallica announces a massive world tour for 2026.",
        "Guns N' Roses release a remastered live album from their 1988 tour.",
        "Nirvana's 'Nevermind' gets a 30th-anniversary reissue.",
        "New interview with Dave Grohl reveals his plans for the next Foo Fighters album.",
        "Queen's 'Bohemian Rhapsody' voted greatest rock anthem of all time.",
        "The Rolling Stones celebrate their 65th anniversary with a surprise concert.",
        "Emerging punk band from London takes the independent scene by storm.",
        "Iconic rock venue in New York City slated for demolition.",
        "New documentary about the life and legacy of Jimi Hendrix is in the works.",
        "Radiohead's Thom Yorke announces a solo acoustic tour.",
        "Pearl Jam performs a benefit concert for climate change awareness.",
        "The Strokes release a new song teasing a forthcoming album.",
        "Heavy metal band unveils a new music video with a dystopian theme.",
        "Classic rock legend shares the secrets behind his signature guitar solo.",
        "Rock Hall of Fame announces its new class of inductees.",
        "Bruce Springsteen announces a new Broadway run of his one-man show.",
        "The Beatles' 'Sgt. Pepper's' album to be re-released with unreleased takes.",
        "Aerosmith postpones their final tour due to Steven Tyler's health issues."
    ]

    ## SET THE PREDICTION THESHOLD
    BEST_THRESHOLD = 0.25

    ## HEADLINE PROCESSING AND PREDICTION
    max_sequence_length = 128
    
    results_list = []

    for headline in new_headlines:
        text_sequence = tokenizer(
            headline,
            padding='max_length',
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="tf"
        )
        predictions = model(text_sequence)
        probabilities = tf.nn.sigmoid(predictions.logits)[0].numpy()

        ## HEADLINE RAW PROBABILITIES
        print(f"\nHeadline: '{headline}'")
        print(f"  Raw Probabilities: {probabilities}")
        print(f"  Labels in Order: {labels}\n")

        predicted_labels = []
        for i, prob in enumerate(probabilities):
            if prob >= BEST_THRESHOLD:
                predicted_labels.append(labels[i])
        
        results_list.append({
            'headline': headline,
            'predicted_labels': predicted_labels
        })

    ## COMPILING RESULTS
    df_results = pd.DataFrame(results_list)
    df_results['predicted_labels'] = df_results['predicted_labels'].apply(lambda x: ', '.join(x) if x else 'No Label')

    print("\n--- Final Predictions ---")
    print(df_results)