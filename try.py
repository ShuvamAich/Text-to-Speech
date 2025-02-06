# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write
import gradio as gr

import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

import sys
sys.path.append('/home/st/st_us-053000/st_st190561/text2speech/gradtts/Speech-Backbones/Grad-TTS/hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

HIFIGAN_CONFIG = '/home/st/st_us-053000/st_st190561/text2speech/gradtts/Speech-Backbones/Grad-TTS/checkpts/config.json'
HIFIGAN_CHECKPT = '/home/st/st_us-053000/st_st190561/text2speech/gradtts/Speech-Backbones/Grad-TTS/checkpts/g_02500000'

# Initialize Grad-TTS and HiFi-GAN once
def initialize_models():
    print('Initializing Grad-TTS...')
    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    generator = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')

    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    vocoder = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    return generator, vocoder

# Synthesize speech
def generate_speech(text):
    cmu = cmudict.CMUDict('/home/st/st_us-053000/st_st190561/text2speech/gradtts/Speech-Backbones/Grad-TTS/resources/cmu_dictionary')

    generator, vocoder = initialize_models()

    with torch.no_grad():
        print(f'Synthesizing text: {text}')
        x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
        x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

        t = dt.datetime.now()
        y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=10, temperature=1.5, stoc=False, spk=None, length_scale=0.91)
        t = (dt.datetime.now() - t).total_seconds()
        print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

        audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
        output_path = '/home/st/st_us-053000/st_st190561/text2speech/gradtts/Speech-Backbones/Grad-TTS/out/sample_3.wav'
        write(output_path, 22050, audio)
    
    return output_path

# Gradio interface
def gradio_speech_synthesis(text):
    audio_path = generate_speech(text)
    return audio_path

demo = gr.Interface(
    fn=gradio_speech_synthesis,
    inputs=gr.Textbox(label="Enter Text to Synthesize"),
    outputs=gr.Audio(label="Generated Speech"),
    title="Text-to-Speech with Grad-TTS"
)

if __name__ == "__main__":
    demo.launch()
