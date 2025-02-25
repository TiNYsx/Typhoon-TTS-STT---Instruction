# Typhoon TTS-STT - Instruction

This repos is going to show you how to setup a Workspace and working with Speech-to-text and Text-to-speech AI using (LLAMA3.1-Typhoon2-Audio-8b-Instruct)[https://huggingface.co/scb10x/llama3.1-typhoon2-audio-8b-instruct] developed by scb10x.

## Setting up workspace
+ You'll need to install (python)[https://www.python.org/downloads/] on your device, for me, I'm using Python 3.12, which is the newest version for me rightnow.
+ Install (VirtualEnvironment)[https://pypi.org/project/virtualenv/] for Python, or you can use `python -m pip install virtualenv` or for python3 `python3 -m pip install virtualenv`.
+ Create a virtual environment with `python -m venv <environment_name>` or you can clone this repository and continue to the next step.
+ Use virtual environment with `source <environment_name>/bin/activate` to activate the virtual environment, for this repos you can use `source instruction/bin/activate`.
+ And that's all for setting up the workspace!

> [!TIP]
> For more information on how to use venv in other os, you can checkout his (document)[https://docs.python.org/3/library/venv.html].

## Downloading a Speech-to-text / Text-to-speech AI Model
+ You can find many models on (Huggingface)[https://huggingface.co/] website, for this instruction, I'd suggest you to use (Typhoon-Audio-8b)[https://huggingface.co/scb10x/llama3.1-typhoon2-audio-8b-instruct] for both Text-to-speech and Speech-to-text.
+ For downloading, you'll need to create a directory to store all the data from now on, so create a folder as you desire, for me it is `instruction/projects`.
+ Set your current directory to the project's path directory by using `cd instruction/projects`
+ Repository cloning command can be found on Huggingface page, for Typhoon-Audio it is `git clone https://huggingface.co/scb10x/llama3.1-typhoon2-audio-8b-instruct`.
> [!NOTE]
> This step might take a while, depends on how fast your wifi/internet is, it can take up to 20-60 minutes.
+ After installed the Typhoon-Audio model, you're good to go to test the mdoel!

## Testing Typhoon-Audio-8b Model
