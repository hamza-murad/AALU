#!/usr/bin/env python
#
# Copyright 2016 IBM
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import argparse
import base64
import configparser
import json
import socket
import threading
import time
import wave
from threading import Thread, Event
import mmap
import pyaudio
import websocket
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from websocket import ABNF

import ibm_watson.Integration.Sarcasm.classifier as sarcasm_module
from ibm_watson import AssistantV2
from ibm_watson import TextToSpeechV1
from ibm_watson import ToneAnalyzerV3
from ibm_watson.Integration.textinput import google_search
import textgrid
import shutil
import subprocess
import os
import re
from ibm_watson.Networking.client import send_data, send_wav_data

REGION_MAP = {
    'us-east': 'gateway-wdc.watsonplatform.net',
    'us-south': 'stream.watsonplatform.net',
    'eu-gb': 'stream.watsonplatform.net',
    'eu-de': 'stream-fra.watsonplatform.net',
    'au-syd': 'gateway-syd.watsonplatform.net',
    'jp-tok': 'gateway-syd.watsonplatform.net',
}

dict = {}


# CYNA#####################################
def Gen_Phoneme(path):
    data = textgrid.TextGrid.fromFile(path)
    ls = data.getList('phones')

    D = {}
    for i in ls[0]:
        D[i.mark] = (i.minTime, i.maxTime)

    print(D)
    with open('D:\\BL8\\phonemesFile.txt', 'w') as f:
        for item in ls[0]:
            f.write("%s\n" % item)

        f.close()


def Gen_Textgrid(reply):
    # a=subprocess.call('D:\\montreal-forced-aligner\\bin\\mfa_align D:\\montreal-forced-aligner\\input D:\\montreal-forced-aligner\\english_dict.txt D:\\montreal-forced-aligner\\english.zip D:\\montreal-forced-aligner\\output')
    shutil.copy('D:\\BL8\\rcvd_file.wav', 'D:\\montreal-forced-aligner\\input')

    global dict
    ukn_word = False

    for word in reply.split():
        if word.upper() not in dict:
            ukn_word = True
            break

    if ukn_word:
        os.system(
            'echo N | D:\\montreal-forced-aligner\\bin\\mfa_align D:\\montreal-forced-aligner\\input D:\\montreal-forced-aligner\\english_dict.txt D:\\montreal-forced-aligner\\english.zip D:\\montreal-forced-aligner\\output')
    else:
        subprocess.call(
            'D:\\montreal-forced-aligner\\bin\\mfa_align D:\\montreal-forced-aligner\\input D:\\montreal-forced-aligner\\english_dict.txt D:\\montreal-forced-aligner\\english.zip D:\\montreal-forced-aligner\\output')

    Gen_Phoneme('D:\\montreal-forced-aligner\\output\\input\\rcvd_file.TextGrid')

    # print('curr in gen',current_state)


def init_NLP():
    # establishing connection
    SERVER = '127.0.0.1'#'127.0.0.1'  # '43.245.206.104'  # "196.194.235.248"
    PORT = 10005
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((SERVER, PORT))

    client.sendall(bytes('NLP', 'UTF-8'))
    status = client.recv(1024)
    print(status.decode())

    return client


def flag_check(event):
    state = 0
    while True:
        with open('D:\\BL8\\states.txt', 'r+b')as f:
            mm = mmap.mmap(f.fileno(), 0)
            # print(mm[0])
            # print(mm[1])
            if mm[1] == 1:
                event.set()
                mm.seek(1)
                mm.write(state.to_bytes(1, byteorder='big'))
                f.close()


CLIENT = init_NLP()
# GLOBALS
TTS_AUTH = None
TTS_SERVICE = None
WATSON_KEY = None
WATSON_AUTH = None
WATSON_ASSISTANT = None
HEADERS = None
USERPASS = None
URL = None
TONE_SERVICE = None
PYAUDIO_OBJ = pyaudio.PyAudio()
PYAUDIO_OBJ_INPUT = pyaudio.PyAudio()
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10  # Cyna
STT = []
TRANSCRIPT = None
WAV_PATH = "C:\\ibmwatson_python_sdk-networking\\ibm_watson\\Networking\\output.wav"
JSON_PATH = "C:\\ibmwatson_python_sdk-networking\\ibm_watson\\Networking\\data.json"
posture_input = None
form_input = None


# GLOBALS


# sets up the variables
def authentication_function():
    global TTS_AUTH
    global TTS_SERVICE
    global WATSON_KEY
    global WATSON_AUTH
    global WATSON_ASSISTANT
    global HEADERS
    global USERPASS
    global URL
    global TONE_SERVICE

    # TTS
    TTS_AUTH = IAMAuthenticator('msoasmplOnX24TCnYif7vaJK_bb9h2MuJG0yjp6kpPOR')
    TTS_SERVICE = TextToSpeechV1(authenticator=TTS_AUTH)
    TTS_SERVICE.set_service_url(
        'https://api.us-south.text-to-speech.watson.cloud.ibm.com/instances/74be1af9-eb16-43eb-ae58-3b3f5849e112')
    # END TTS

    # STT
    HEADERS = {}
    USERPASS = ":".join(get_auth())
    HEADERS["Authorization"] = "Basic " + base64.b64encode(
        USERPASS.encode()).decode()
    URL = get_url()
    # END STT

    # WATSON
    WATSON_AUTH = IAMAuthenticator(
        'AGesgrUJa4L4OVBHpbJgTKfOeCU6kVeVxo2qhIVFqIYS')  # put the general watson api key here
    WATSON_ASSISTANT = AssistantV2(
        version='2018-09-20',
        authenticator=WATSON_AUTH)
    WATSON_ASSISTANT.set_service_url(
        'https://api.us-south.assistant.watson.cloud.ibm.com/instances/28f6a127-f399-482b-9b66-5502ad5af6f5')
    session = WATSON_ASSISTANT.create_session(
        "9bf7bf36-235e-4089-bf1d-113791da5b43").get_result()  # put the specific assistant api key
    WATSON_KEY = session.get("session_id", "")
    # END WATSON

    # TONE ANALYZER
    TONE_AUTHENTICATOR = IAMAuthenticator(
        'b0DmzKxaFck7YymuFStEYpJPMmt_bbYLPu8fPO9aEend')
    TONE_SERVICE = ToneAnalyzerV3(
        version='2017-09-21',
        authenticator=TONE_AUTHENTICATOR)
    TONE_SERVICE.set_service_url(
        'https://api.us-south.tone-analyzer.watson.cloud.ibm.com/instances/4a4d15eb-5212-447b-8da9-dcad6434130a')
    # TONE ANALYZER


def play_wav(path):
    # output via pyaudio
    f = wave.open(path, "rb")

    stream = PYAUDIO_OBJ.open(format=PYAUDIO_OBJ.get_format_from_width(f.getsampwidth()),
                              channels=f.getnchannels(),
                              rate=f.getframerate(),
                              output=True)
    # read data
    data = f.readframes(CHUNK)
    # play stream
    while data:
        stream.write(data)
        data = f.readframes(CHUNK)

    # stop stream
    stream.stop_stream()
    stream.close()


# TTS
def generate_wav(reply):
    # authenticator = IAMAuthenticator('gV332Uci-w4LVq6fturapl2P88gE50SFtGBG9wjWelYq')
    # service = TextToSpeechV1(authenticator=authenticator)
    # service.set_service_url(
    #    'https://api.us-south.text-to-speech.watson.cloud.ibm.com/instances/2a38c46e-2eb1-4376-8a09-a7c1713865a4')

    # voices = service.list_voices().get_result()
    # print(json.dumps(voices, indent=2))
    # Cyna########################################################
    # watson tts api call

    ''' with open('D:\\BL8\\rcvd_file.wav',
               'wb') as audio_file:
         response = TTS_SERVICE.synthesize(
             reply, accept='audio/wav',
             voice="en-US_MichaelV3Voice").get_result()
         audio_file.write(response.content)
         audio_file.close()'''
    # watson api call
    with open("D:\\BL8\\rcvd_file.wav",
              'wb') as audio_file:
        response = TTS_SERVICE.synthesize(
            reply, accept='audio/wav',
            voice="en-US_MichaelV3Voice").get_result()
        audio_file.write(response.content)

    # play_wav(WAV_PATH)


# uses watson to get an answer for conversation
# Cyna############
seq = ['I want to drop a course', 'why is the sky blue', 'I would like to transfer my credits.']


def generate_reply(transcript):
    global posture_input
    global form_input

    if form_input is not None:
        transcript=form_input
        form_input=None
    elif posture_input is not None and transcript=="":
        transcript=posture_input
        posture_input=None

    message = json.dumps(WATSON_ASSISTANT.message(
        "9bf7bf36-235e-4089-bf1d-113791da5b43", WATSON_KEY,
        input={'text': transcript},
        context={
            'metadata': {
                'deployment': 'myDeployment'
            }
        }).get_result())

    parsed_message = json.loads(message)
    reply = parsed_message['output']['generic'][0]['text']
    intents = parsed_message['output']['intents']

    # check if watson returned a valid reply
    if len(intents) is not 0:
        print("Your question: ", transcript)
        print('Reply: ', reply)

    # if watson does not understand, send google search call
    else:
        reply = google_search('embeddedassistant.googleapis.com',
                              'C:\\Users\\intel\\AppData\\Roaming\\google-oauthlib-tool\\credentials.json',
                              'watson-73b2e-watsongoogle-famt7c', '5debc6de-60a0-11ea-be7f-186024c1a96d',
                              'en-US', False, False, 185, transcript)

    if reply is None or reply is ' ':
        sarcastic_remark = sarcasm_module.sarcasm_detector(transcript)
        if sarcastic_remark is not None:
            print("Your Input: ", transcript)
            print("Reply: ", sarcastic_remark)
            reply = sarcastic_remark
            tone = "Sarcastic"
    # divert program flow to IBM watson if sarcasm not detected
    if reply is None or reply is ' ':
        reply = "Sorry, I could not understand that. Could you try rephrasing your question?"
        print('Reply: ', reply)

    # determine the tone of the generated reply
    tone = TONE_SERVICE.tone(
        tone_input=reply, content_type="text/plain").get_result()
    if tone['document_tone']['tones']:
        arr = tone['document_tone']['tones']
        new_list = sorted(arr, key=lambda k: k['score'], reverse=True)
        # {k: v for k, v in sorted(arr.items(), key=lambda item:['score'])}
        tone = new_list[0]['tone_name']
    else:
        tone = "Neutral"

    print('Reply tone: ', tone)
    print(reply)
    reply = reply.replace('\n',' ')
    reply = re.sub(r'http\S+', '', reply)
    reply= reply.encode('ascii', 'ignore').decode('ascii')
    print(reply)
    return reply, tone


# STT
def read_audio(ws, timeout):
    global RATE

    # signal blender to switch to listening state
    listening_state = 1

    with open('D:\\BL8\\states.txt', 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)
        mm.write(int(listening_state).to_bytes(1, byteorder='big'))
        f.close()

    RATE = int(PYAUDIO_OBJ_INPUT.get_default_input_device_info()['defaultSampleRate'])
    stream = PYAUDIO_OBJ_INPUT.open(format=FORMAT,
                                    channels=CHANNELS,
                                    rate=RATE,
                                    input=True,
                                    frames_per_buffer=CHUNK)

    print("* Recording...")
    rec = RECORD_SECONDS or timeout

    for i in range(0, int(RATE / CHUNK * rec)):
        data = stream.read(CHUNK)

        ws.send(data, ABNF.OPCODE_BINARY)

    # Disconnect the audio stream
    stream.stop_stream()
    stream.close()
    print("* Recording ended.")

    # signal blender to switch to thinking state
    thinking_state = 2

    with open('D:\\BL8\\states.txt', 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)
        mm.write(int(thinking_state).to_bytes(1, byteorder='big'))
        f.close()

    # In order to get a final response from STT we send a stop, this
    # will force a final=True return message.
    data = {"action": "stop"}
    ws.send(json.dumps(data).encode('utf8'))
    # ... which we need to wait for before we shutdown the websocket
    time.sleep(1)
    ws.close()

    # ... and kill the audio device
    # p.terminate()


# SST
def on_message(self, msg):

    data = json.loads(msg)
    if "results" in data:
        if data["results"][0]["final"]:
            STT.clear()
            STT.append(data)
        # This prints out the current fragment that we are working on
        print(data['results'][0]['alternatives'][0]['transcript'])


# SST
def on_error(self, error):
    """Print any errors."""
    print(error)


# SST
def on_close(ws):
    """Upon close, print the complete and final transcript."""
    global TRANSCRIPT
    global STT
    """Upon close, print the complete and final transcript."""
    if STT:
        TRANSCRIPT = "".join([i['results'][0]['alternatives'][0]['transcript'] for i in STT])
    else:
        TRANSCRIPT = ""
    STT = []

# SST
def on_open(ws):
    """Triggered as soon a we have an active connection."""
    args = ws.args
    data = {
        "action": "start",
        # this means we get to send it straight raw sampling
        "content-type": "audio/l16;rate=%d" % RATE,
        "continuous": True,
        "interim_results": True,
        # "inactivity_timeout": 5, # in order to use this effectively
        # you need other tests to handle what happens if the socket is
        # closed by the server.
        "word_confidence": True,
        "timestamps": True,
        "max_alternatives": 3
    }

    # Send the initial control message which sets expectations for the
    # binary stream that follows:
    ws.send(json.dumps(data).encode('utf8'))
    # Spin off a dedicated thread where we are going to read and
    # stream out audio.
    threading.Thread(target=read_audio,
                     args=(ws, args.timeout)).start()


# SST
def get_url():
    config = configparser.RawConfigParser()
    config.read('speech.cfg')
    # See
    # https://console.bluemix.net/docs/services/speech-to-text/websockets.html#websockets
    # for details on which endpoints are for each region.
    region = config.get('auth', 'region')
    host = REGION_MAP[region]
    return ("wss://{}/speech-to-text/api/v1/recognize"
            "?model=en-US_BroadbandModel").format(host)


# SST
def get_auth():
    config = configparser.RawConfigParser()
    config.read('speech.cfg')
    apikey = config.get('auth', 'apikey')
    return "apikey", apikey


# SST
def parse_args():
    parser = argparse.ArgumentParser(
        description='Transcribe Watson text in real time')
    parser.add_argument('-t', '--timeout', type=int, default=5)
    # parser.add_argument('-d', '--device')
    # parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    return args


Cache = {}


def watson_loop(sst_socket, client_socket, event_signal):
    with open('C:\cache\config.txt', 'rt') as df:
        while True:
            s = df.readline()
            if s == '':
                break
            c = s.split('|')
            Cache[c[0]] = c[1]
        df.close()
        print(Cache)

    while True:

        event_signal.wait()

        # call the STT service that will generate a transcript

        sst_socket.run_forever()
        try:
            reply, tone = generate_reply(TRANSCRIPT)
        except KeyError:
            print('dict change error..................')
            reply='Sorry, I could not understand that. Could you try rephrasing your question?'
            tone='sadness'

        '''json_dict = {'reply': reply, 'tone': tone}
        with open(JSON_PATH, 'w') as outfile:
            json.dump(json_dict, outfile)'''
        # Cyna##########################################################################
        if reply not in Cache:
            try:
                shutil.rmtree('C:\\Users\\intel\\Documents\\MFA')
            except FileNotFoundError:
                print('it do be like that')
            with open('D:\\montreal-forced-aligner\\input\\rcvd_file.txt', 'w') as f:
                f.write(reply)
                f.close()

            with open('D:\\BL8\\tone.txt', 'w') as f:
                f.write(tone.lower()+'\n')
                f.write('D:\\BL8\\phonemesFile.txt'+'\n')
                f.write('D:\\BL8\\rcvd_file.wav')
                f.close()

            ########################################################################
            generate_wav(reply)

            Gen_Textgrid(reply)

            # send_wav_data(WAV_PATH, client_socket, "SEND_WAV")

            # send_data(JSON_PATH, client_socket, "SEND_JSON")
            # signal blender to switch to speaking state
            # CYNA#######################################################
        else:
            with open('D:\\BL8\\tone.txt', 'w') as f:
                f.write(tone.lower()+'\n')
                f.write('C:\\cache\\' + Cache[reply].rstrip('\n') + '.txt' +'\n')
                f.write('C:\\cache\\' + Cache[reply].rstrip('\n') + '.wav'+ '\n')
                f.close()

        if reply == 'Please submit the form.':
            speaking_state = 7
        else:
            speaking_state = 3

        with open('D:\\BL8\\states.txt', 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0)
            mm.write(int(speaking_state).to_bytes(1, byteorder='big'))
            f.close()

        event_signal.clear()


def main():
    # Cyna####################
    f = open("D:\\montreal-forced-aligner\\english_dict.txt", "rt")
    global dict
    dict = {}
    for line in f:
        dict[line.split()[0]] = line

    ###############################

    authentication_function()
    ws = websocket.WebSocketApp(URL, header=HEADERS, on_message=on_message, on_error=on_error, on_close=on_close,
                                on_open=on_open)
    ws.args = parse_args()

    anim_signal = Event()
    work = Thread(target=watson_loop, args=(ws, CLIENT, anim_signal))
    work.start()

    while True:

        cv_signal = Event()
        cv_input = Thread(target=flag_check, args=(anim_signal,))

        cv_input.start()

        while True:
            server_input = CLIENT.recv(1025)
            cou = 0
            server_input = server_input.decode()
            print(server_input)

            if 'CV_INPUT' in server_input.upper():
                global posture_input
                global form_input
                rcv = CLIENT.recv(1025)
                rcv = rcv.decode()
                print(rcv)
                CLIENT.sendall((1).to_bytes(1,byteorder='big'))
                ls=rcv.split(',')
                if ls[0]=='form':
                    form_input=ls[1]
                if ls[0]=='posture':
                    posture_input=ls[1]

                time_input = time
                cv_signal.set()


if __name__ == "__main__":
    main()
