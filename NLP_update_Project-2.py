#!/usr/bin/env python
# coding: utf-8



import torch




from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')





from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')





def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    
    
def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))
    
    ##
    tokens = tokenizer.convert_ids_to_tokens(input_ids)



    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text
    

    
    
    
    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

  
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

        
    import matplotlib.pyplot as plt
    import seaborn as sns

    
    sns.set(style='darkgrid')

    
    plt.rcParams["figure.figsize"] = (16,8)
    
    s_scores = start_scores.detach().numpy().flatten()
    e_scores = end_scores.detach().numpy().flatten()

    # We'll use the tokens as the x-axis labels. In order to do that, they all need
    # to be unique, so we'll add the token index to the end of each one.
    token_labels = []
    for (i, token) in enumerate(tokens):
        token_labels.append('{:} - {:>2}'.format(token, i))
        
        

    
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]
            
    ax = sns.barplot(x=token_labels, y=s_scores, ci=None)

    # Turn the xlabels vertical.
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
    
    # Turn on the vertical grid to help align words to scores.
    ax.grid(True)

    plt.title('Start Word Scores')

    plt.show()
    
    
    # Create a barplot showing the end word score for all of the tokens.
    ax = sns.barplot(x=token_labels, y=e_scores, ci=None)

    # Turn the xlabels vertical.
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

    # Turn on the vertical grid to help align words to scores.
    ax.grid(True)

    plt.title('End Word Scores')

    plt.show()
            
    import pandas as pd

    # Store the tokens and scores in a DataFrame. 
    # Each token will have two rows, one for its start score and one for its end
    # score. The "marker" column will differentiate them. A little wacky, I know.
    scores = []
    for (i, token_label) in enumerate(token_labels):

        # Add the token's start score as one row.
        scores.append({'token_label': token_label, 
                       'score': s_scores[i],
                       'marker': 'start'})
    
        # Add  the token's end score as another row.
        scores.append({'token_label': token_label, 
                       'score': e_scores[i],
                       'marker': 'end'})
    
    df = pd.DataFrame(scores)
    
    g = sns.catplot(x="token_label", y="score", hue="marker", data=df,
                kind="bar", height=6, aspect=4)

    # Turn the xlabels vertical.
    g.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha="center")
    
    #print("Both start and end word togeather")
    # Turn on the vertical grid to help align words to scores.
    g.ax.grid(True)
    
    import time
 
    # Wait for 5 seconds
    time.sleep(5)


    speak(answer)
    print('Answer: "' + answer + '"')
    


# In[5]:


import textwrap

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=80) 

bert_abstract = "In meteorology, precipitation is any product of the condensation of atmospheric water vapor that falls under gravity. The main forms of pre- cipitation include drizzle, rain, sleet, snow, grau- pel and hail... Precipitation forms as smaller droplets coalesce via collision with other rain drops or ice crystals within a cloud. Short, in- tense periods of rain in scattered locations are called “showers”."

print(wrapper.fill(bert_abstract))





import pyttsx3
# pip install pyttsx3
import datetime
#inbuild library 
engine  = pyttsx3.init()
voices = engine.getProperty('voices')       #getting details of current voice
#engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female
#print(voices[1].id) #printing voice id name of the recoder 


import textwrap
import speech_recognition as sr
def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening .....")
        audio = r.listen(source)
    try:
        print("Recognizing  ......")
        query= r.recognize_google(audio,language="en-in")
        print(f"user said : {query}\n")
    except Exception as e:
        print("Say that again please !" )
    
    return query

if __name__ == "__main__":
    
    # Wrap text to 80 characters.
    wrapper = textwrap.TextWrapper(width=80) 
    ans= input("Do you want new paragraph (Yes) or use the exciting ? (No)").lower()
    if(ans == "no"):
        bert_abstract = "In meteorology, precipitation is any product of the condensation of atmospheric water vapor that falls under gravity. The main forms of pre- cipitation include drizzle, rain, sleet, snow, grau- pel and hail... Precipitation forms as smaller droplets coalesce via collision with other rain drops or ice crystals within a cloud. Short, in- tense periods of rain in scattered locations are called “showers”."
        print(wrapper.fill(bert_abstract))
    elif(ans=="yes"):
        bert_abstract = input("Enter your Paragraph .. \n")
        print(wrapper.fill(bert_abstract))
    input_type = input("How would you like your input speech or text ? \n ").lower()
    
    
    if(input_type== "text"):
        futher="yes"
        while(futher != "no"):
            question = input("Enter the question to be asked ? \n")
            answer_question(question, bert_abstract)
            import time 
            # Wait for 2 seconds
            time.sleep(2)
            futher =input("Do you want to ask more question ? \n").lower()
   
    else:
        futher="yes"
        while(futher != "no"):
            question = takecommand()
            answer_question(question, bert_abstract)
            import time 
            # Wait for 3.2 seconds
            time.sleep(3.2)
            print("Do you want to ask more question ?")
            futher = takecommand()
        
    
        
    


    


