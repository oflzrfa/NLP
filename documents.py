import requests
from bs4 import BeautifulSoup
import nltk
import re
import os
import shutil

# Download the web page and extract the raw text content
def extract_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content.decode('utf-8'), "html.parser")
    text = ''
    for paragraph in soup.find_all('p'):
        text += paragraph.text
    text = re.sub(r'\[\d+\]', ' ', text)
    text = text.replace('\n', '')
    return (text)

# Tokenize the text into sentences
def extract_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences[:50]
#extract 100 sentences from each

urls =[
    "https://en.wikipedia.org/wiki/History_of_medicine",
    "https://en.wikipedia.org/wiki/Medical_racism_in_the_United_States",
    "https://en.wikipedia.org/wiki/U.S._Immigration_and_Customs_Enforcement",
    "https://en.wikipedia.org/wiki/Controlled_Substances_Act",
    "https://en.wikipedia.org/wiki/Drug_Enforcement_Administration",
    "https://en.wikipedia.org/wiki/Greco-Roman_world",
    "https://en.wikipedia.org/wiki/Gender_identity",
    "https://en.wikipedia.org/wiki/World_War_I",
    "https://en.wikipedia.org/wiki/Picasso",
    "https://en.wikipedia.org/wiki/SpaceX",
    "https://en.wikipedia.org/wiki/Great_Depression",
    "https://en.wikipedia.org/wiki/Maya_civilization",
    "https://en.wikipedia.org/wiki/Batman",
    "https://en.wikipedia.org/wiki/Climate_change_denial",
    "https://en.wikipedia.org/wiki/Macbeth",
    "https://en.wikipedia.org/wiki/Meiji_Restoration",
    "https://en.wikipedia.org/wiki/Brave_New_World",
    "https://en.wikipedia.org/wiki/Marxism",
    "https://en.wikipedia.org/wiki/Industrial_Revolution",
    "https://en.wikipedia.org/wiki/The_Holocaust",
    "https://en.wikipedia.org/wiki/Quantum_mechanics",
    "https://en.wikipedia.org/wiki/Harry_Potter",
    "https://en.wikipedia.org/wiki/Ancient_Egypt",
    "https://en.wikipedia.org/wiki/Renaissance",
    "https://en.wikipedia.org/wiki/Atomic_bombings_of_Hiroshima_and_Nagasaki",
    "https://en.wikipedia.org/wiki/Barack_Obama",
    "https://en.wikipedia.org/wiki/World_War_II",
    "https://en.wikipedia.org/wiki/Freud",
    "https://en.wikipedia.org/wiki/Slavery_in_the_United_States",
    "https://en.wikipedia.org/wiki/Middle_Ages",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Cold_War",
    "https://en.wikipedia.org/wiki/French_Revolution",
    "https://en.wikipedia.org/wiki/Nelson_Mandela",
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "https://en.wikipedia.org/wiki/Black_Death",
    "https://en.wikipedia.org/wiki/Pablo_Picasso",
    "https://en.wikipedia.org/wiki/Inca_Empire",
    "https://en.wikipedia.org/wiki/Plato",
    "https://en.wikipedia.org/wiki/George_Washington",
    "https://en.wikipedia.org/wiki/Elizabeth_I",
    "https://en.wikipedia.org/wiki/Jane_Austen",
    "https://en.wikipedia.org/wiki/Salem_witch_trials",
    "https://en.wikipedia.org/wiki/Harriet_Tubman",
    "https://en.wikipedia.org/wiki/Steve_Jobs",
    "https://en.wikipedia.org/wiki/Leonardo_da_Vinci",
    "https://en.wikipedia.org/wiki/Prohibition_in_the_United_States",
    "https://en.wikipedia.org/wiki/The_Great_Gatsby",
    "https://en.wikipedia.org/wiki/Civil_Rights_Movement",
    "https://en.wikipedia.org/wiki/Apartheid",
    "https://en.wikipedia.org/wiki/French_and_Indian_War",
    "https://en.wikipedia.org/wiki/J.K._Rowling",
    "https://en.wikipedia.org/wiki/Romeo_and_Juliet",
    "https://en.wikipedia.org/wiki/Christopher_Columbus",
    "https://en.wikipedia.org/wiki/American_Civil_War",
    "https://en.wikipedia.org/wiki/Charles_Darwin",
    "https://en.wikipedia.org/wiki/Crusades",
    "https://en.wikipedia.org/wiki/Big_Bang",
    "https://en.wikipedia.org/wiki/The_Iliad",
    "https://en.wikipedia.org/wiki/Napoleon_Bonaparte"
    ]


# Folder that stores all retrieved documents
documents_folder = "documents"
if not os.path.exists(documents_folder):
    os.mkdir(documents_folder)

documentsfolder = []
for i, url in enumerate(urls):
    
    text = extract_text(url) 
    sentences = extract_sentences(text)
    print(f"Extracted {len(sentences)} sentences from {url}")
    print(extract_sentences(text))

    # Save the subdocuments to files
    filename = f"text_{i+1}.txt" 
    doc_path = os.path.join(documents_folder, filename)
    #with open(filename, "w", encoding="utf-8") as file:  
    with open(doc_path, "w", encoding="utf-8") as file:  
        file.write(text)
        documentsfolder.append(doc_path)

#export retrieved documents folder to local file  
local_folder = r"C:\Users\refij\OneDrive\Dokumenti\GitHub\NLP"
shutil.copytree(documents_folder, os.path.join(local_folder, documents_folder))
   
        


    


