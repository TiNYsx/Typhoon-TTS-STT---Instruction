from dataclasses import dataclass, asdict
from transformers import pipeline, VitsTokenizer, VitsModel, set_seed, AutoModelForCausalLM, AutoTokenizer
import torch
import wave
import pyaudio
import scipy
import numpy

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
    recorder.record(time, "Insurebot/audios/audio.wav")

def getTextFromRecord(pipe):
    transcriptions = pipe(
        "Insurebot/audios/audio.wav",
        batch_size=16,
        return_timestamps=False,
        generate_kwargs={"language": "<|th|>", "task": "transcribe"}
    )["text"]
    return transcriptions

def getAudio(textInput):
    inputs = tokenizer_tts(text=textInput, return_tensors="pt")
    set_seed(456)
    with torch.no_grad():
        outputs = model_tts(**inputs)
    waveform = outputs.waveform[0]
    waveform_array = waveform.numpy()
    waveform_array = waveform_array * 2.0
    waveform_array = numpy.clip(waveform_array, -1.0, 1.0)
    scipy.io.wavfile.write("Insurebot/audios/output.wav", rate=model_tts.config.sampling_rate, data=waveform_array)

def getAiResponse(userInput):
    messages = [
        {"role": "system", "content": "คุณคือ อินชัวร์บอท เป็นผู้ช่วยที่ตอบคำถามเกี่ยวกับสิทธิประโยชน์ของผู้ประกันตน หรือประกันสังคม ทั้งด้านกฎหมาย ความรู้ง่ายๆ หรือให้คำปรึกษา ซึ่งคุณจะต้องให้คำตอบที่ถูกต้อง ใช้ข้อความและคำพูดที่เป็นมิตรต่อผู้สูงอายุ และผู้ที่เข้าใจความหมายคำพูดได้ยาก. คุณถูกตั้งให้เป็นเพศหญิง และจะใช้คำว่า 'ค่ะ' แทนคำว่า 'ครับ'. โดยคำตอบที่คุณจะให้กับผู้ใช้งาน จะสั้น เข้าใจง่าย ได้ใจความ ไม่วกวนหรือเยอะจนเกินไป."},
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

device = "cpu"

sttModel = "biodatlab/whisper-th-medium-combined"
ttsModel = "VIZINTZOR/MMS-TTS-THAI-FEMALEV2"
llmModel = "Insurebot/llama3.2-typhoon2-3b-instruct"

pipe = getSTTModel()

tokenizer_tts = VitsTokenizer.from_pretrained(ttsModel,cache_dir="./mms")
model_tts = VitsModel.from_pretrained(ttsModel,cache_dir="./mms")

tokenizer_llm = AutoTokenizer.from_pretrained(llmModel)
model_llm = AutoModelForCausalLM.from_pretrained(llmModel)

quantized_model = torch.quantization.quantize_dynamic(model_llm, {torch.nn.Linear}, dtype=torch.qint8)

appExit = False

while appExit == False:
    print(
        "0.Exit the app\n" +
        "Input number to start recording."
    )
    recordTime = int(input("How long you want to record? (Use 0 to exit): "))
    if (recordTime != 0):
        getRecorder(recordTime)
        resultText = getTextFromRecord(pipe)
        getAudio(getAiResponse(resultText))
    else:
        appExit = True