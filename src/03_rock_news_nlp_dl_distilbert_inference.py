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
from transformers import AutoTokenizer

## SET UP PATHS AND DIRECTORIES
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'assets', 'distilbert')
QUANTIZED_MODEL_PATH = os.path.join(ASSETS_DIR, 'multilabel_fold_5_distilbert', 'multilabel_fold_5_distilbert_quantized.tflite')
TOKENIZER_FOLDER_PATH = os.path.join(ASSETS_DIR, 'tokenizer_distilbert')
LABELS_PATH = os.path.join(ASSETS_DIR, 'labels_distilbert.pkl')

## LOAD DATA (MODEL, TOKENIZER AND LABELS)
try:
    interpreter = tf.lite.Interpreter(model_path=QUANTIZED_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FOLDER_PATH)
    
    with open(LABELS_PATH, 'rb') as f:
        labels = pickle.load(f)

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("\nIMPORTANT: The required files were not found.")
    print("Please ensure the TFLite model, tokenizer, and labels files are in the correct directories.")
    
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print("This might be due to an issue with the TFLite model or tokenizer files.")
    
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

    ## SET THE PREDICTION THRESHOLD
    BEST_THRESHOLD = 0.25

    ## HEADLINE PROCESSING AND PREDICTION
    max_sequence_length = 128
    
    results_list = []

    ## INPUT TENSOR INDICES
    input_ids_index = input_details[0]['index']
    attention_mask_index = input_details[1]['index']
    
    required_dtype = input_details[0]['dtype']

    for headline in new_headlines:
        text_sequence = tokenizer(
            headline,
            padding='max_length',
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="tf"
        )
        
        ## NUMPY ARRAYS CONVERSION
        input_ids = text_sequence.input_ids.numpy().astype(required_dtype)
        attention_mask = text_sequence.attention_mask.numpy().astype(required_dtype)

        ## SET THE TENSORS AMD INVOKE THE INTERPRETER 
        interpreter.set_tensor(input_ids_index, input_ids)
        interpreter.set_tensor(attention_mask_index, attention_mask)
        
        interpreter.invoke()

        ## GET THE OUTPUT
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        probabilities = tf.nn.sigmoid(output_data)[0].numpy()

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

    print("\n## Final Predictions ##")
    print(df_results)
