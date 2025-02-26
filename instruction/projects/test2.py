from transformers import VitsTokenizer, VitsModel, set_seed
import torch
import scipy

def getAudio(textInput):
    inputs = tokenizer_tts(text=textInput, return_tensors="pt")
    set_seed(456)
    with torch.no_grad():
        outputs = model_tts(**inputs)
    waveform = outputs.waveform[0]
    waveform_array = waveform.numpy()
    scipy.io.wavfile.write("instruction/projects/output.wav", rate=model_tts.config.sampling_rate, data=waveform_array)


ttsModel = "VIZINTZOR/MMS-TTS-THAI-MALEV2"

tokenizer_tts = VitsTokenizer.from_pretrained(ttsModel,cache_dir="./mms")
model_tts = VitsModel.from_pretrained(ttsModel,cache_dir="./mms")

getAudio("สวัสดีครับ. มีอะไรให้ช่วยมั้ยครับ?")