import requests
import threading
import random

def test_call():
    list_audio = [
        #  "./data/noise-free-sound-0541.wav",
        #  "./data/target_raw.wav",
         "./data/sample_de.wav",
    ]
    with open(random.choice(list_audio),"rb") as f:
            d = {'pcm_s16le': f.read(), 'prefix': ''}
    print(requests.post('http://0.0.0.0:5005/asr/infer/de,{}'.format(random.choice(['de', 'en'])), files=d).json())

if __name__ == "__main__":
    num_thread = 10
    threads = []
    for _ in range(num_thread):
        threads.append(threading.Thread(target=test_call))
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()