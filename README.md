
# AALU


AALU is a cloud-based intelligent virtual agent to solve the common customer issues automatically and freeing up staff to focus on complex stuff. Four main components are:

  - NLP
  - CV
  - Modelling
  - Backend

## NLP
NLP primarily comprises of natural language understanding (human to machine) and natural language generation (machine to human).
### IBM Cloud Serivces
   `IBM Watson` is used for semantic analysis, tone analyzation and reply generation. Its model was trained for domain-specific knowledge. Currently, the bot can answer queries that apropos a university setting including registering for a course, dropping a course, asking for help, etcetera.  
### Small Talk
`Google Assistant` (Actions) has been integrated with IBM Watson Assistant in order to provide a realistic and interactive bot in case the user ever starts a conversation outside of the service bots domain. This module handles everything from non-domain related questions to Smalltalk such as greetings, goodbyes, etc. 
### Sarcasm
A self-contained python module (`Multinomial Naïve Bayes Classifier`) has been set up that classifies the input as sarcastic or not. If the input is found to be sarcastic then a fitting sarcastic reply is generated in response.  

## CV
Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.
### OCR
Document Scanner gets an image as input and applies image preprocessing techniques to enhance quality of image and adjust orientation. It produces a processed image from which OCR can finally extract required content.

### Person Detection
 The face detection is incorporated using opencv library. `OpenCV` uses machine learning algorithms to search for faces within a picture. `dlib` library is used for head posture detection. The library has a frontal face detector that is made by HOG feature, linear classifier, an image pyramid and sliding window detection scheme. It has fair accuracy in detecting when a person is facing the camera.

### Gesture Detection
This module is responsible for recognizing and classifying the body language, gestures and motions made by the user.  This would help aid the virtual agent in deciding what the appropriate response would be as well as adding onto the audio input stream.

## Modelling
This module is responsible for recognizing and classifying the body language, gestures and motions made by the user.  This would help aid the virtual agent in deciding what the appropriate response would be as well as adding onto the audio input stream.

## Backend
The purpose of this subsection was to provide an interface for Blender (the software we are using as a front end) for managing and running all the animations and also constructing all the real-time animations i.e. the lip-sync and the eye tracking.

# Contributions
### NLP Team
1. [Muhammad Wasiq]()
2. [Hadi Amjad](https://github.com/hadiamjad)
3. [Ather Fawaz]()
4. [Muhammad Adan]()
5. [Usman Khan]()
6. [Daniyal Tahir]()
7. [Arsylan Sheikh]()
8. [Hafsa Saleem]()
9. [Habiba Akram]()

### CV Team
1. [Hamza Murad](https://github.com/hamza-murad)
2. [Muhammad Bilal]()
3. [Abou Sufiyan]()
4. [Ali Zahid]()
5. [Muhammad Romman]()
6. [Amina Zeb]()
7. [Ammara Naseer]()
8. [Zarrish Nadeem]()
9. [Durraiz Waseem]()
10. [Taha Khan]()

## Modelling Team
1. [Mubariz Barkat]()
2. [Muhammad Mujtaba]()
3. [Feza Roheel]()
4. [Umair Ahmed]()
5. [Maryam Nizam](https://github.com/MaryamNizam)

## Backend Team
1. [Sharjeel Hassan]()
2. [Sana Basharat]()
3. [Hmaza Jawad]()

License
----
AALU-TEAM
