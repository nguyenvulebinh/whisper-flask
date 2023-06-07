from flask import Flask, request
import torch
import numpy as np
import math
import sys
import json
import threading
import queue
import uuid
import traceback
import os
import whisper
from whisper.decoding import decode, DecodingOptions
from whisper.audio import log_mel_spectrogram, pad_or_trim, SAMPLE_RATE, N_FRAMES, HOP_LENGTH
os.environ["CUDA_VISIBLE_DEVICES"]=""

host = sys.argv[1]  # 192.168.0.72
port = sys.argv[2]  # 5051
# host = '0.0.0.0'
# port = 5000


app = Flask(__name__)

def create_unique_list(my_list):
    my_list = list(set(my_list))
    return my_list

def initialize_model():
    # filename = "/export/data1/workspaces/whisper/model-bin/tiny.en.pt"
    filename =  "tiny" # ["tiny.en","tiny","base.en","base","small.en","small","medium.en","medium","large-v1","large-v2","large"]

    model = whisper.load_model(filename)
    print("ASR initialized")

    max_batch_size = 8

    return model, max_batch_size

def pad_audios(list_audios):
    list_segments = []
    for audio in list_audios:
        mel = log_mel_spectrogram(audio.squeeze())
        segment = pad_or_trim(mel, N_FRAMES).unsqueeze(0)
        list_segments.append(segment)
    
    return torch.cat(list_segments, dim=0).to(model.device)



def use_model(reqs):

    if len(reqs) == 1:
        req = reqs[0]
        audio_tensor, prefix, input_language, output_language = req.get_data()
        if not (input_language == output_language or output_language == 'en'):
            result = {"hypo": "", "status":400, "message": 'Wrong option. Perform X->X "transcribe" or X->English "translate". Found {} -> {}'.format(input_language, output_language)}
            req.publish(result)
            return
        result = model.transcribe(audio_tensor.squeeze(), language=input_language, initial_prompt=prefix, 
                                  task="transcribe" if input_language == output_language else "translate",
                                  temperature=0)
        hypo = result['text']
        if not hypo.strip().startswith(prefix.strip()):
            hypo = prefix + hypo
        
        result = {"hypo": hypo.strip()}
        req.publish(result)

    else:

        audio_tensors = list()
        prefixes = ['']
        input_languages = list()
        output_languages = list()

        batch_runnable = False

        for req in reqs:
            audio_tensor, prefix, input_language, output_language = req.get_data()
            audio_tensors.append(audio_tensor)
            prefixes.append(prefix)
            input_languages.append(input_language)
            output_languages.append(output_language)

        unique_prefix_list = create_unique_list(prefixes)
        unique_input_languages = create_unique_list(input_languages)
        unique_output_languages = create_unique_list(output_languages)
        if len(unique_prefix_list) == 1 and len(unique_input_languages) == 1 and len(unique_output_languages) == 1:
            batch_runnable = True

        if batch_runnable:

            segments = pad_audios(audio_tensors)
            if unique_input_languages[0] == unique_output_languages[0]:
                hypos = decode(model, segments, DecodingOptions(**{'language': input_languages[0], 'fp16': False, 'prompt': []}))
            elif unique_output_languages[0] == 'en':
                hypos = decode(model, segments, DecodingOptions(**{'task': 'translate', 'language': input_languages[0], 'fp16': False, 'prompt': []}))
            else:
                for req in reqs:
                    result = {"hypo": "", "status":400, "message": 'Wrong option. Perform X->X "transcribe" or X->English "translate". Found {} -> {}'.format(unique_input_languages[0], unique_output_languages[0])}
                    req.publish(result)
                return

            for req, hypo in zip(reqs, hypos):
                result = {"hypo": hypo.text}
                req.publish(result)
        else:
            for req, audio_tensor, prefix, input_language, output_language \
                    in zip(reqs, audio_tensors, prefixes, input_languages, output_languages):
                if not (input_language == output_language or output_language == 'en'):
                    result = {"hypo": "", "status":400, "message": 'Wrong option. Perform X->X "transcribe" or X->English "translate". Found {} -> {}'.format(input_language, output_language)}
                    req.publish(result)
                else:
                    result = model.transcribe(audio_tensor.squeeze(), language=input_language, initial_prompt=prefix,
                                              task="transcribe" if input_language == output_language else "translate",
                                              temperature=0)
                    hypo = result['text']
                    if not hypo.strip().startswith(prefix.strip()):
                        hypo = prefix + hypo
                    result = {"hypo": hypo}
                    req.publish(result)

def run_decoding():
    while True:
        reqs = [queue_in.get()]
        while not queue_in.empty() and len(reqs) < max_batch_size:
            req = queue_in.get()
            reqs.append(req)
            if req.priority >= 1:
                break

        print("Batch size:",len(reqs),"Queue size:",queue_in.qsize())

        try:
            use_model(reqs)
        except Exception as e:
            print("An error occured during model inference")
            traceback.print_exc()
            for req in reqs:
                req.publish({"hypo":"", "status":400})

class Priority:
    next_index = 0

    def __init__(self, priority, id, condition, data):
        self.index = Priority.next_index

        Priority.next_index += 1

        self.priority = priority
        self.id = id
        self.condition = condition
        self.data = data

    def __lt__(self, other):
        return (-self.priority, self.index) < (-other.priority, other.index)

    def get_data(self):
        return self.data

    def publish(self, result):
        dict_out[self.id] = result
        try:
            with self.condition:
                self.condition.notify()
        except:
            print("ERROR: Count not publish result")

def pcm_s16le_to_tensor(pcm_s16le):
    audio_tensor = np.frombuffer(pcm_s16le, dtype=np.int16)
    audio_tensor = torch.from_numpy(audio_tensor)
    audio_tensor = audio_tensor.float() / math.pow(2, 15)
    audio_tensor = audio_tensor.unsqueeze(1)  # shape: frames x 1 (1 channel)
    return audio_tensor

# corresponds to an asr_server "http://$host:$port/asr/infer/en,en" in StreamASR.py
# use None when no input- or output language should be specified
@app.route("/asr/infer/<input_language>,<output_language>", methods=["POST"])
def inference(input_language, output_language):
    pcm_s16le: bytes = request.files.get("pcm_s16le").read()
    prefix = request.files.get("prefix") # can be None
    if prefix is not None:
        prefix: str = prefix.read().decode("utf-8")

    # calculate features corresponding to a torchaudio.load(filepath) call
    audio_tensor = pcm_s16le_to_tensor(pcm_s16le)

    priority = request.files.get("priority") # can be None
    try:
        priority = int(priority.read()) # used together with priority queue
    except:
        priority = 0

    condition = threading.Condition()
    with condition:
        id = str(uuid.uuid4())
        if input_language.lower() == 'none':
            input_language = None
        data = (audio_tensor,prefix,input_language,output_language)

        queue_in.put(Priority(priority,id,condition,data))

        condition.wait()

    result = dict_out.pop(id)
    status = 200
    if status in result:
        status = result.pop(status)

    # result has to contain a key "hypo" with a string as value (other optional keys are possible)
    return json.dumps(result), status

# called during automatic evaluation of the pipeline to store worker information
@app.route("/asr/version", methods=["POST"])
def version():
    # return dict or string (as first argument)
    return "Whisper", 200

model, max_batch_size = initialize_model()

queue_in = queue.PriorityQueue()
dict_out = {}

decoding = threading.Thread(target=run_decoding)
decoding.daemon = True
decoding.start()


if __name__ == "__main__":
    app.run(host=host, port=port)
