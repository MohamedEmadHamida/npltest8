#region Import and download data 

#           Import 
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from spellchecker import SpellChecker
from itertools import permutations
import random

#region check data files 

# Load the stopwords
#nltk.download('stopwords')
#nltk.download('punkt')

#endregion

#region Enable Debug
DEBUG = True  # Set to False to disable all print statements

def debug_print(message):
    if DEBUG:
        print('\n', message, '\n')

#endregion

#region features   
'''
Simple LNP Model For Tips Recommendation about illness and symptoms
 
feature: 
    1- Convert human text to useful words 
    2- Handle human error of writing useful words
    3- Matching the useful words with dataset
    4- Show tips if there is a good match from random tips
    5- Multi-tip return from single user input 
    libraries used: 
        pandas: A powerful data manipulation and analysis library providing data structures and handling structured data.
        nltk: A popular natural language processing (NLP) package for Python.
        fuzzywuzzy: A Python library for string matching and comparison.
        nltk.corpus.stopwords: A collection of common stop words in various languages, used to filter out non-essential words in text processing.
        nltk.tokenize.word_tokenize: A tokenizer that splits text into individual words and punctuation, facilitating text analysis.
                        
v2 
using spellchecker for better accuracy and support en 
mark multi-word tokens 
bigger dataset and fix duplicates
'''
#endregion

# Global variables 
ThresholdValue = 80  # threshold for a good match 

#region Load your symptoms and tips data
debug_print("Loading data...")
symptoms_df = pd.read_csv('symptoms.csv')  # Ensure this file exists with 'Symptom' and 'Symptom_ID' columns
tips_df = pd.read_csv('tips.csv')          # Ensure this file exists with 'Tip' and 'Symptom_ID' columns
debug_print("done Loading data...")

#endregion

#region Spell checker and custom words
# load custom words
def load_custom_words(spell, file_path):
    # Load words from the file into the spell checker
    with open(file_path, 'r') as file:
        custom_words = file.read().splitlines()
        spell.word_frequency.load_words(custom_words)
    return set(custom_words)  # Return the set of valid words for further filtering    

def correct_sentence(sentence):
    # Initialize the spell checker
    spell = SpellChecker()
    # Load custom words and get a set of allowed words
    allowed_words = load_custom_words(spell, "SpellCheakerTextHelper.txt")
    # Split the sentence into words and correct each misspelled word
    corrected_words = []
    for word in sentence.split():
        # Check if the word is misspelled and correct if needed
        if word in spell:
            corrected_word = word
        else:
            corrected_word = spell.correction(word)
        
        # Only keep the word if it's in the allowed list
        if corrected_word in allowed_words:
            corrected_words.append(corrected_word)

    # Return the array of corrected words
    return corrected_words
#endregion

# Take text and convert tokens
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    debug_print(f"Tokens: {tokens}")
    return tokens

#region symptoms matches
def get_closest_symptoms(user_inputs):
    symptoms_list = symptoms_df['Symptom'].tolist()
    all_matched_indices = []

    for user_input in user_inputs:
        # Check if the word exists in the symptoms list and get its index
        matched_indices = [
            symptoms_list.index(symptom) for symptom in symptoms_list if symptom == user_input
        ]
        
        # Add matches to the result list
        all_matched_indices.extend(matched_indices)

    if all_matched_indices:
        return all_matched_indices
    
    return []
#endregion

def preprocess_and_extract_symptoms(text, symptoms_list, threshold=80):
    # Tokenize and preprocess user input
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    
    # Join tokens to handle multi-word symptoms
    processed_text = ' '.join(tokens)
    
    debug_print(f"Processed text: {processed_text}")
    found_symptoms = []
    
    # Fuzzy match each symptom in the list to handle typos and multi-word symptoms
    for symptom in symptoms_list:
        # Use fuzzy matching for each symptom
        match_score = fuzz.partial_ratio(symptom, processed_text)
        if match_score >= threshold:
            found_symptoms.append(symptom)
    
    return found_symptoms

#region Get random tips based on matched indices
def get_Random_tips(matched_indices):
    # Get all corresponding Symptom_IDs from the matched indices
    symptom_ids = symptoms_df.loc[matched_indices, 'Symptom_ID']
    
    # Get tips for the list of Symptom_IDs
    tips = tips_df[tips_df['Symptom_ID'].isin(symptom_ids)][['Symptom_ID', 'Tip1', 'Tip2', 'Tip3']]

    # Melt the DataFrame to have a single column for tips
    tips_melted = tips.melt(id_vars='Symptom_ID', value_vars=['Tip1', 'Tip2', 'Tip3'], value_name='Tip')

    # Drop NaN values and remove duplicates
    tips_melted = tips_melted.dropna(subset=['Tip']).drop_duplicates(subset=['Symptom_ID', 'Tip'])

    # Randomly choose one tip for each symptom ID without including the grouping columns
    tips_for_symptoms = tips_melted.groupby('Symptom_ID')['Tip'].apply(lambda x: random.choice(x.tolist())).to_dict()

    return tips_for_symptoms
#endregion

#region Function to handle the closest symptom and tips
def GetTipsWithHandling(closest_symptom):
    if closest_symptom:
        debug_print(f"Closest match found: {closest_symptom}")
        tip = get_Random_tips(closest_symptom)
        if tip:
            print("\n\n\n", f"Tip: {tip}", "\n\n\n")
        else:
            print("\n\n\n", "No tips found for this symptom.", "\n\n\n")
    else:
        print("\n\n\n", "No matching symptom found.", "\n\n\n")
#endregion

#region Function to generate match scores
def generate_match_scores(tokens):
    if len(tokens) > 1:
        # Generate circular two-word combinations
        circular_combinations = [f"{tokens[i]} {tokens[(i+1) % len(tokens)]}" for i in range(len(tokens))]

        # Generate all unique pairs in any order (non-circular)
        unique_combinations = [f"{x} {y}" for x, y in permutations(tokens, 2)]

        # Combine both lists and remove duplicates
        match_scores = list(set(circular_combinations + unique_combinations))
        return match_scores
    else:
        return []
#endregion

# Main function to handle user input
# Main function to handle user input
def main():
    # Step 1: Capture user input
    user_input = "I have fever" 
    debug_print(f"User Input: {user_input}")

    # Step 2: Correct the input sentence for spelling
    corrected_input = correct_sentence(user_input)
    debug_print(f"User Input After Correction: {corrected_input}")

    # Step 3: Generate match scores for the corrected input
    user_input_matches = generate_match_scores(corrected_input)
    debug_print(f"User Input After Generating Match Scores: {user_input_matches}")

    # Step 4: Find the closest symptoms based on matches
    closest_symptoms_from_matches = get_closest_symptoms(user_input_matches)
    closest_symptoms_from_input = get_closest_symptoms(corrected_input)
    all_closest_symptoms = list(set(closest_symptoms_from_matches + closest_symptoms_from_input))
    debug_print(f"All Closest Symptoms: {all_closest_symptoms}")

    # Step 5: Get tips based on matched symptoms
    GetTipsWithHandling(all_closest_symptoms)


if __name__ == "__main__":
    main()
    
#region tasks 
'''
1- Tasks add more than one tip for each symptom  ✓
2- Make random selection from tips when there is a matching symptom ✓
3- More than one symptom in the same time ✓
4- Find dataset for this 
5- Make it matching multi symptoms at same time with ThresholdValue >= 60 ✓
6- Bigger dataset of Symptom 
7- Test results and make it more accurate
8- Test the performance of the code
9- Super loop with one time load data 

v2 tasks 
1- Use spellchecker for better accuracy and support en 
2- Txt file for spellchecker 
3- Test multi-word tokens
4- Test single-word tokens 
5- Test performance of the code
6- Split each function in a file
'''

#endregion
