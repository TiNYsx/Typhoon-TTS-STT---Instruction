# Typhoon TTS-STT - Instruction

This repos is going to show you how to setup a Workspace and working with Speech-to-text and Text-to-speech AI using [llama3.2-typhoon2-3b-instruct](https://huggingface.co/scb10x/llama3.2-typhoon2-3b-instruct) developed by scb10x.

## Setting up workspace

### Downloading Python and VirtualEnvironment
+ You'll need to install [python](https://www.python.org/downloads/) on your device, for me, I'm using Python 3.12, which is the newest version for me rightnow.
+ Install [VirtualEnvironment](https://pypi.org/project/virtualenv/) for Python, or you can use `python -m pip install virtualenv` or for python3 `python3 -m pip install virtualenv`.
+ Create a virtual environment with `python -m venv <environment_name>` or you can clone this repository and continue to the next step.
+ Use virtual environment with `source <environment_name>/bin/activate` to activate the virtual environment, for this repos you can use `source instruction/bin/activate`.
+ And that's all for setting up the workspace!

### Requirement
+ dataclasses
+ transformers
+ torch
+ wave
+ pyaudio
+ scipy
+ numpy

> [!TIP]
> For more information on how to use venv in other os, you can checkout his [document](https://docs.python.org/3/library/venv.html).

## Downloading a LLM AI Model
+ You can find many models on [Huggingface](https://huggingface.co/) website, for this instruction, I'd suggest you to use [llama3.2-typhoon2-3b-instruct](https://huggingface.co/scb10x/llama3.2-typhoon2-3b-instruct) for text generating llm model.
+ For downloading, you'll need to create a directory to store all the data from now on, so create a folder as you desire, for me it is `instruction/projects`.
+ Set your current directory to the project's path directory by using `cd instruction/projects`
+ Repository cloning command can be found on Huggingface page, for Typhoon 2 - 3B - Instruct it is `git clone https://huggingface.co/scb10x/llama3.2-typhoon2-3b-instruct`.
+ After installed the model, you're good to go to test the model!
> [!NOTE]
> This step might take a while, depends on how fast your wifi/internet is, it can take up to 10-30 minutes.

## Testing LLM Model - [llama3.2-typhoon2-3b-instruct](https://huggingface.co/scb10x/llama3.2-typhoon2-3b-instruct)

<details>
<summary>Example - Python Code</summary>

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

def getAiResponse(userInput):
    messages = [
        {"role": "system", "content": "คุณคือผู้ช่วยที่จะตอบคำถามด้วยคำตอบที่ถูกต้อง สั้นแต่ได้ใจความ และเข้าใจง่าย."},
        {"role": "user", "content": userInput}
    ]

    input_ids = tokenizer_llm.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model_llm.device)

    terminators = [
        tokenizer_llm.eos_token_id,
        tokenizer_llm.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model_llm.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        temperature=0.7,
        top_p=0.95)
    response = tokenizer_llm.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response

llmModel = "instruction/projects/llama3.2-typhoon2-3b-instruct"

tokenizer_llm = AutoTokenizer.from_pretrained(llmModel)
model_llm = AutoModelForCausalLM.from_pretrained(llmModel)

print(getAiResponse("สวัสดี, คุณช่วยสอนฉันทำข้าวผัดหน่อยได้มั้ย?"))
```
</details>

<details>
<summary>Example - Output</summary>

แน่นอน! นี่คือวิธีทำข้าวผัด:

1. **เตรียมวัตถุดิบ**: ข้าวสวย, ไข่, เนื้อสัตว์ (เช่น หมู, ไก่), ผัก (เช่น แครอท, ถั่วงอก), ซอสถั่วเหลือง, น้ำมันพืช.
2. **ผัดข้าว**: ใส่น้ำมันพืชในกระทะ ตั้งไฟกลางแล้วใส่เนื้อสัตว์ผัดจนเหลือง.
3. **ใส่ข้าว**: ใส่ข้าวสวยลงไปผัดให้เข้ากัน.
4. **เติมไข่**: ใส่ไข่ลงไปผัดให้เข้ากับข้าว.
5. **ผัดผัก**: ใส่ผักลงไปผัดจนเข้ากัน.
6. **ปรุงรส**: เติมซอสถั่วเหลืองแล้วผัดให้เข้ากันอีกครั้ง.
7. **เสริฟ**: เสิร์ฟข้าวผัดร้อนๆ พร้อมกับน้ำจิ้มซีฟู้ดหรือซอสถั่วเหลือง.

สนุกกับการทำข้าวผัดนะ!
</details>


## Testing Text-to-speech Model - [VIZINTZOR/MMS-TTS-THAI-MALEV2](https://huggingface.co/VIZINTZOR/MMS-TTS-THAI-MALEV2)

> [!WARNING]
> This models only support Thai input language, if you're using any English character, the model will just ignore them.

<details>
<summary>Example - Python Code</summary>

```py
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
```
</details>
<details>
<summary>Example - Result</summary>

https://github.com/user-attachments/assets/c04e0a8a-8a2b-4f96-9024-25f55f978372

</details>

## Testing Speech-to-text Model - [biodatlab/distill-whisper-th-medium](https://huggingface.co/biodatlab/distill-whisper-th-medium)

<details>
<summary>Example - Python Code</summary>

```py
from dataclasses import dataclass, asdict
from transformers import pipeline
import wave
import pyaudio

@dataclass
class StreamParams:
    format: int = pyaudio.paInt16
    channels: int = 2
    rate: int = 44100   
    frames_per_buffer: int = 1024
    input: bool = True
    output: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

class Recorder:
    def __init__(self, stream_params: StreamParams) -> None:
        self.stream_params = stream_params
        self._pyaudio = None
        self._stream = None
        self._wav_file = None

    def record(self, duration: int, save_path: str) -> None:
        print("Start recording...")
        self._create_recording_resources(save_path)
        self._write_wav_file_reading_from_stream(duration)
        self._close_recording_resources()
        print("Stop recording")

    def _create_recording_resources(self, save_path: str) -> None:
        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(**self.stream_params.to_dict())
        self._create_wav_file(save_path)

    def _create_wav_file(self, save_path: str):
        self._wav_file = wave.open(save_path, "wb")
        self._wav_file.setnchannels(self.stream_params.channels)
        self._wav_file.setsampwidth(self._pyaudio.get_sample_size(self.stream_params.format))
        self._wav_file.setframerate(self.stream_params.rate)

    def _write_wav_file_reading_from_stream(self, duration: int) -> None:
        for _ in range(int(self.stream_params.rate * duration / self.stream_params.frames_per_buffer)):
            audio_data = self._stream.read(self.stream_params.frames_per_buffer)
            self._wav_file.writeframes(audio_data)

    def _close_recording_resources(self) -> None:
        self._wav_file.close()
        self._stream.close()
        self._pyaudio.terminate()

def getSTTModel():
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=sttModel,
        chunk_length_s=30,
        device=device)
    return pipe

def getRecorder(time):
    stream_params = StreamParams()
    recorder = Recorder(stream_params)
    recorder.record(time, "instruction/projects/audio.wav")

def getTextFromRecord(pipe):
    transcriptions = pipe(
        "instruction/projects/audio.wav",
        batch_size=16,
        return_timestamps=False,
        generate_kwargs={"language": "<|th|>", "task": "transcribe"}
    )["text"]
    return transcriptions

device = "cpu"
sttModel = "biodatlab/whisper-th-medium-combined"
pipe = getSTTModel()
getRecorder(5)
print(getTextFromRecord(pipe))
```
</details>
<details>
<summary>Example - Input</summary>

https://github.com/user-attachments/assets/7810b1e9-1410-4142-befa-84bb30fff3b6

</details>
<details>
<summary>Example - Result</summary>

สวัสดี ครับ คุณ ชื่อ อะไร หรอ ครับ

</details> 