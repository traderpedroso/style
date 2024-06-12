import os
import gradio as gr # type: ignore
import styletts2importable
# import ljspeechimportable
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from text_utils import normalizer
import torch

from txtsplit import txtsplit # type: ignore
import numpy as np
import soundfile as sf
import warnings
from pedalboard.io import AudioFile
from pedalboard import *
import noisereduce as nr
        
warnings.filterwarnings("ignore")

INTROTXT = """
# StyleTTS 2
"""

theme = gr.themes.Base(
    font=[
        gr.themes.GoogleFont("Libre Franklin"),
        gr.themes.GoogleFont("Public Sans"),
        "system-ui",
        "sans-serif",
    ],
)

voicelist = [
    "Francisco",
    "Antonio",
    "Paulo",
    "James",
    "Lucas",
    "Bia",
    "Pedro",
    "Emerson",
    "Luna",
    "Cristina",
    "Carlos",
    "Esther",
    "vs",
    
]
voices = {}
voices_path = {}




for v in voicelist:
    if v != "vs":
        voices[v] = styletts2importable.compute_style(f"voices/{v}.wav")
    voices_path[v] = f"voices/{v}.wav"

if not torch.cuda.is_available():
    INTROTXT += "\n\n### Você está em um sistema Text to speech "

ckpt_converter = 'Models/OpenVoice/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)


def openvoice(voice):
    base_speaker = f'{output_dir}/temp.wav'
    source_se, audio_name = se_extractor.get_se(base_speaker, tone_color_converter, vad=True)
    target_se, audio_name = se_extractor.get_se(voices_path[voice], tone_color_converter, vad=True)
    encode_message = "@MyShell"
    audio = tone_color_converter.convert(
    audio_src_path=base_speaker, 
    src_se=source_se, 
    tgt_se=target_se, 
    output_path=None,
    message=encode_message)
    return 24000, audio


def trim_audio(audio_np_array, sample_rate=24000, trim_ms=500):
    trim_samples = int(trim_ms * sample_rate / 1000)
    if len(audio_np_array) > 2 * trim_samples:
        trimmed_audio_np = audio_np_array[trim_samples:-trim_samples]
    else:
        trimmed_audio_np = audio_np_array
    return trimmed_audio_np


def synthesize(
    text, voice, multispeakersteps, alpha, beta, embscale, progress=gr.Progress()
):
    text = normalizer(text)
    if text.strip() == "":
        raise gr.Error("Você deve inserir algum texto")
    if len(text) > 50000:
        raise gr.Error("O texto deve ter menos de 50.000 caracteres")
    texts = txtsplit(text, desired_length=100, max_length=250)
    audios = []
    for t in progress.tqdm(texts):
        audio = styletts2importable.inference(
            t,
            voices[voice],
            alpha=alpha,
            beta=beta,
            diffusion_steps=multispeakersteps,
            embedding_scale=embscale,
        )
        # Considerando que `audio` já seja um array NumPy
        trimmed_audio = trim_audio(audio)
        audios.append(trimmed_audio)
    result = (24000, np.concatenate(audios)) 
    sample_rate, audio_data = result
    sf.write(f'{output_dir}/temp.wav',audio_data, samplerate=sample_rate)    
    return result




def process_audio(input_file):
    # Set sampling rate
    sr = 24000
    # Read audio file
    with AudioFile(input_file).resampled_to(sr) as f:
        audio = f.read(f.frames)

    # Reduce stationary noise
    reduced_noise = nr.reduce_noise(
        y=audio, sr=sr, stationary=True, prop_decrease=0.75)

    # Apply audio effects using pedalboard
    board = Pedalboard([
        NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
        Compressor(threshold_db=-16, ratio=2.5),
        LowShelfFilter(cutoff_frequency_hz=400, gain_db=1, q=1),
        Gain(gain_db=1)
    ])

    processed_audio = board(reduced_noise, sr)

    return processed_audio, sr

def clsynthesize(text, voice, vcsteps, embscale, alpha, beta, progress=gr.Progress()):
    text = normalizer(text)
    if text.strip() == "":
        raise gr.Error("Você deve inserir algum texto")
    if len(text) > 50000:
        raise gr.Error("O texto deve ter menos de 50.000 caracteres")
    if embscale > 1.3 and len(text) < 20:
        gr.Warning("AVISO: Você inseriu um texto curto, você pode obter estática!")
    texts = txtsplit(text, desired_length=100, max_length=400)
    audios = []
    vs = styletts2importable.compute_style(voice)
    for t in progress.tqdm(texts):
        audio = styletts2importable.inference(
            t,
            vs,
            alpha=alpha,
            beta=beta,
            diffusion_steps=vcsteps,
            embedding_scale=embscale,
        )
         # Considerando que `audio` já seja um array NumPy
        trimmed_audio = trim_audio(audio)
        audios.append(trimmed_audio)
        
    
    result = (24000, np.concatenate(audios)) 
    sample_rate, audio_data = result
    
    wave, sr = sf.read(voice)
    sf.write(f'voices/vs.wav',wave, samplerate=sr) 
    sf.write(f'{output_dir}/temp.wav',audio_data, samplerate=sample_rate)    
    return result


def clsyn(voice, speaker):
    wave, sr = sf.read(voice)
    sf.write(f'{output_dir}/temp.wav',wave, samplerate=sr)
    result = openvoice(speaker)    
    return result


# def ljsynthesize(text, steps, progress=gr.Progress()):
#     if text.strip() == "":
#         raise gr.Error("Você deve inserir algum texto")
#     if len(text) > 150000:
#         raise gr.Error("O texto deve ter menos de 150.000 caracteres")
#     noise = torch.randn(1, 1, 256).to("cuda" if torch.cuda.is_available() else "cpu")
#     texts = txtsplit(text, desired_length=250, max_length=400)
#     audios = []
#     for t in progress.tqdm(texts):
#         audio = ljspeechimportable.inference(
#             t, noise, diffusion_steps=steps, embedding_scale=1
#         )
#         trimmed_audio = trim_audio(audio)
#         audios.append(trimmed_audio)
#     return (24000, np.concatenate(audios))


with gr.Blocks() as vctk:
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(
                label="Texto",
                info="O que você gostaria que o StyleTTS 2 lesse? Funciona melhor com frases completas.",
                interactive=True,
            )
            voice = gr.Dropdown(
                voicelist,
                label="Voz",
                info="Selecione uma voz padrão.",
                value="Francisco",
                interactive=True,
            )
            multispeakersteps = gr.Slider(
                minimum=5,
                maximum=200,
                value=20,
                step=1,
                label="Passos de Difusão",
                info="Teoricamente, quanto maior, melhor a qualidade, mas mais lento. Experimente com passos menores primeiro - é mais rápido",
                interactive=True,
            )
            alpha = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.3,
                step=0.1,
                label="Alpha",
                info="Determina o timbre da fala, quanto maior, mais adequado o estilo ao texto do que à voz alvo.",
                interactive=True,
            )
            beta = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.7,
                step=0.1,
                label="Beta",
                info="Determina a entonação e um ritmo da fala, quanto maior, mais adequado o estilo ao texto do que à voz alvo.",
                interactive=True,
            )
            embscale = gr.Slider(
                minimum=1,
                maximum=3,
                value=1,
                step=0.1,
                label="Escala de Embedding",
                info="Escala maior significa que o estilo é mais condicional ao texto de entrada e, portanto, mais emocional.",
                interactive=True,
            )
        with gr.Column(scale=1):
            btn = gr.Button("Sintetizar", variant="primary")
            audio = gr.Audio(
                interactive=False,
                label="Áudio Sintetizado",
                waveform_options={"waveform_progress_color": "#3C82F6"},
            )
            
            btn.click(
                synthesize,
                inputs=[inp, voice, multispeakersteps, alpha, beta, embscale],
                outputs=[audio],
                concurrency_limit=4,
            )
            
            btn1 = gr.Button("Humanizar", variant="secondary")
            audio_tonecolor = gr.Audio(
                interactive=False,
                label="Áudio Humanizado",
                waveform_options={"waveform_progress_color": "#3C82F6"},
            )
            
            btn1.click(
                openvoice,
                inputs=[voice],
                outputs=[audio_tonecolor],
                concurrency_limit=4,
            )
            
            

with gr.Blocks() as clone:
    with gr.Row():
        with gr.Column(scale=1):
            clinp = gr.Textbox(
                label="Texto",
                info="O que você gostaria que o StyleTTS 2 lesse? Funciona melhor com frases completas.",
                interactive=True,
            )
            clvoice = gr.Audio(
                label="Voz",
                interactive=True,
                type="filepath",
                max_length=300,
                waveform_options={"waveform_progress_color": "#3C82F6"},
            )
            vcsteps = gr.Slider(
                minimum=3,
                maximum=200,
                value=20,
                step=1,
                label="Passos de Difusão",
                info="Teoricamente, quanto maior, melhor a qualidade, mas mais lento. Experimente com passos menores primeiro - é mais rápido",
                interactive=True,
            )
            
            alpha = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.3,
                step=0.1,
                label="Alpha",
                info="Determina o timbre da fala, quanto maior, mais adequado o estilo ao texto do que à voz alvo.",
                interactive=True,
            )
            beta = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.7,
                step=0.1,
                label="Beta",
                info="Determina a entonação e um ritmo da fala, quanto maior, mais adequado o estilo ao texto do que à voz alvo.",
                interactive=True,
            )
            embscale = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=0.1,
                label="Escala de Embedding",
                info="Escala maior significa que o estilo é mais condicional ao texto de entrada e, portanto, mais emocional.",
                interactive=True,
            )
        with gr.Column(scale=1):
            clbtn = gr.Button("Sintetizar", variant="primary")
            claudio = gr.Audio(
                interactive=False,
                label="Áudio Sintetizado",
                waveform_options={"waveform_progress_color": "#3C82F6"},
            )
            
            voice = gr.Textbox(
                label="Voz Humanizar base",
                value="vs",
                interactive=False,
            )
            clbtn.click(
                clsynthesize,
                inputs=[clinp, clvoice, vcsteps, embscale, alpha, beta],
                outputs=[claudio],
                concurrency_limit=4,
            )
            
            clbtn1 = gr.Button("Humanizar", variant="secondary")
            claudio_tonecolor = gr.Audio(
                interactive=False,
                label="Áudio Humanizado",
                waveform_options={"waveform_progress_color": "#3C82F6"},
            )
            
            clbtn1.click(
                openvoice,
                inputs=[voice],
                outputs=[claudio_tonecolor],
                concurrency_limit=4,
            )

with gr.Blocks() as cls:
    with gr.Row():
        with gr.Column(scale=1):
            
            speaker = gr.Dropdown(
                voicelist,
                label="Voz",
                info="Selecione uma voz padrão.",
                value="Francisco",
                interactive=True,
            )
            
            clsvoice = gr.Audio(
                label="Voz",
                interactive=True,
                type="filepath",
                max_length=300,
                waveform_options={"waveform_progress_color": "#3C82F6"},
            )
        with gr.Column(scale=1):
            clsbtn = gr.Button("Clone", variant="primary")
            
            clsaudio = gr.Audio(
                interactive=False,
                label="Áudio Sintetizado",
                waveform_options={"waveform_progress_color": "#3C82F6"},
            )
            clsbtn.click(
                clsyn,
                inputs=[clsvoice, speaker],
                outputs=[clsaudio],
                concurrency_limit=4,
            )

with gr.Blocks(
    title="StyleTTS 2", css="footer{display:none !important}", theme=theme
) as demo:
    gr.Markdown(INTROTXT)
    gr.TabbedInterface(
        [vctk, clone, cls],
        ["Multi-Voice", " TTS Clonagem de Voz","Clonagem de Voz"],
    )
    gr.Markdown(
        """
"""
    )

if __name__ == "__main__":
    demo.queue(api_open=False, max_size=15).launch(show_api=False,server_name="0.0.0.0")
