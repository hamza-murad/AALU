# AALU

AALU is a cloud-based intelligent virtual agent to solve the common customer issues automatically and freeing up staff to focus on complex stuff. Three main components are:

  - NLP
  - CV
  - Modelling

## NLP
NLP primarily comprises of natural language understanding (human to machine) and natural language generation (machine to human).
### IBM Cloud Serivces
   IBM Watson is used for semantic analysis, tone analyzation and reply generation. Its model was trained for domain-specific knowledge. Currently, the bot can answer queries that apropos a university setting including registering for a course, dropping a course, asking for help, etcetera.  
### Small Talk
Google Assistant (Actions) has been integrated with IBM Watson Assistant in order to provide a realistic and interactive bot in case the user ever starts a conversation outside of the service bots domain. This module handles everything from non-domain related questions to Smalltalk such as greetings, goodbyes, etc. 
### Sarcasm
A self-contained python module (Multinomial Naïve Bayes Classifier) has been set up that classifies the input as sarcastic or not. If the input is found to be sarcastic then a fitting sarcastic reply is generated in response.  

