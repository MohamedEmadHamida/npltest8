import csv
import pandas as pd 
def load_from_csv(csv_filename):
    words = set()  # Use a set to automatically remove duplicates
    with open(csv_filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        # Skip the header row if your CSV file has headers
        next(csvreader, None)
        for row in csvreader:
            # Split each phrase into individual words and add to the set
            for word in row[1].split():  # Assuming the word or phrase is in the second column
                words.add(word.lower())  # Convert to lowercase to ensure unique entries
                
    print (words)
    return words


def DeleteDuplicateWords(words):
    words = pd.drop_duplicates(words)
    
    return words

def save_to_txt(words, txt_filename):
    with open(txt_filename, 'w') as file:
        for word in sorted(words):  # Sort words for consistency (optional)
            file.write(word + "\n")

# Load data from CSV and save it to TXT in a spellchecker-friendly format
csv_filename = 'symptoms.csv'  # Replace with your actual CSV file name
txt_filename = 'SpellCheakerTextHelper.txt'

words = load_from_csv(csv_filename)
save_to_txt(words, txt_filename)
print(f"Data saved to {txt_filename}")
