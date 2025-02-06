from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

input_text = "Tell me about Rumi, the Persian poet and mystic."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_length=100, temperature=0.7, top_k=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))


import numpy as np
import gradio as gr

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def generate_speech(note, octave, duration=):
    sr = 48000
    a4_freq, tones_from_a4 = 440, 12 * (octave - 4) + (note - 9)
    frequency = a4_freq * 2 ** (tones_from_a4 / 12)
    duration = int(duration)
    audio = np.linspace(0, duration, duration * sr)
    audio = (20000 * np.sin(audio * (2 * np.pi * frequency))).astype(np.int16)
    return sr, audio

demo = gr.Interface(
    generate_speech,
    [
        gr.Textbox(label="Write the text"),
    ],
    "audio",
)
if __name__ == "__main__":
    demo.launch()
