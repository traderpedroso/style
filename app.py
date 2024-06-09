import os
import gradio as gr # type: ignore
import styletts2importable
import ljspeechimportable
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import torch

from txtsplit import txtsplit # type: ignore
import numpy as np
import soundfile as sf
import warnings

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
    "francisco",
    "antonio",
    "paulo",
    "james",
    "lucas",
    "loucutor",
    "bia",
    "luna",
    "masculino-1",
    "masculino-2",
    "masculino-3",
    "masculino-4",
    "feminino-1",
    "feminino-2",
    "feminino-3",
    "feminino-4",
]
voices = {}




for v in voicelist:
    voices[v] = styletts2importable.compute_style(f"voices/{v}.wav")

if not torch.cuda.is_available():
    INTROTXT += "\n\n### Você está em um sistema Text to speech "

ckpt_converter = 'checkpoints/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)


def openvoice(wave, sr=24000, voice):
    sf.write(f'{output_dir}/temp.wav', wave, sr)
    base_speaker = f'{output_dir}/temp.wav'
    source_se, audio_name = se_extractor.get_se(base_speaker, tone_color_converter, vad=True)
    target_se, audio_name = se_extractor.get_se(voice, tone_color_converter, vad=True)
    save_path = f'{output_dir}/output.wav'
    encode_message = "@MyShell"
    tone_color_converter.convert(
    audio_src_path=voice, 
    src_se=source_se, 
    tgt_se=target_se, 
    output_path=save_path,
    message=encode_message)
    
    return save_path


def trim_audio(audio_np_array, sample_rate=24000, trim_ms=300):
    trim_samples = int(trim_ms * sample_rate / 1000)
    if len(audio_np_array) > 2 * trim_samples:
        trimmed_audio_np = audio_np_array[trim_samples:-trim_samples]
    else:
        trimmed_audio_np = audio_np_array
    return trimmed_audio_np


def synthesize(
    text, voice, multispeakersteps, alpha, beta, embscale, progress=gr.Progress()
):
    if text.strip() == "":
        raise gr.Error("Você deve inserir algum texto")
    if len(text) > 50000:
        raise gr.Error("O texto deve ter menos de 50.000 caracteres")
    texts = txtsplit(text, desired_length=250, max_length=400)
    v = voice.lower()
    audios = []
    for t in progress.tqdm(texts):
        audio = styletts2importable.inference(
            t,
            voices[v],
            alpha=alpha,
            beta=beta,
            diffusion_steps=multispeakersteps,
            embedding_scale=embscale,
        )
        # Considerando que `audio` já seja um array NumPy
        trimmed_audio = trim_audio(audio)
        audios.append(trimmed_audio)
    audio = (24000, np.concatenate(audios))

    return audio


def clsynthesize(text, voice, vcsteps, embscale, alpha, beta, progress=gr.Progress()):
    if text.strip() == "":
        raise gr.Error("Você deve inserir algum texto")
    if len(text) > 50000:
        raise gr.Error("O texto deve ter menos de 50.000 caracteres")
    if embscale > 1.3 and len(text) < 20:
        gr.Warning("AVISO: Você inseriu um texto curto, você pode obter estática!")
    texts = txtsplit(text, desired_length=250, max_length=400)
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
        trimmed_audio = trim_audio(audio)
        audios.append(trimmed_audio)
    return (24000, np.concatenate(audios))


def ljsynthesize(text, steps, progress=gr.Progress()):
    if text.strip() == "":
        raise gr.Error("Você deve inserir algum texto")
    if len(text) > 150000:
        raise gr.Error("O texto deve ter menos de 150.000 caracteres")
    noise = torch.randn(1, 1, 256).to("cuda" if torch.cuda.is_available() else "cpu")
    texts = txtsplit(text, desired_length=250, max_length=400)
    audios = []
    for t in progress.tqdm(texts):
        audio = ljspeechimportable.inference(
            t, noise, diffusion_steps=steps, embedding_scale=1
        )
        trimmed_audio = trim_audio(audio)
        audios.append(trimmed_audio)
    return (24000, np.concatenate(audios))


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
                value="loucutor",
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
            clbtn.click(
                clsynthesize,
                inputs=[clinp, clvoice, vcsteps, embscale, alpha, beta],
                outputs=[claudio],
                concurrency_limit=4,
            )

with gr.Blocks() as lj:
    with gr.Row():
        with gr.Column(scale=1):
            ljinp = gr.Textbox(
                label="Texto",
                info="O que você gostaria que o StyleTTS 2 lesse? Funciona melhor com textos maiores.",
                interactive=True,
            )
            ljsteps = gr.Slider(
                minimum=3,
                maximum=20,
                value=3,
                step=1,
                label="Passos de Difusão",
                info="Teoricamente, quanto maior, melhor a qualidade, mas mais lento. Experimente com passos menores primeiro - é mais rápido",
                interactive=True,
            )
        with gr.Column(scale=1):
            ljbtn = gr.Button("Sintetizar", variant="primary")
            ljaudio = gr.Audio(
                interactive=False,
                label="Áudio Sintetizado",
                waveform_options={"waveform_progress_color": "#3C82F6"},
            )
            ljbtn.click(
                ljsynthesize,
                inputs=[ljinp, ljsteps],
                outputs=[ljaudio],
                concurrency_limit=4,
            )

with gr.Blocks(
    title="StyleTTS 2", css="footer{display:none !important}", theme=theme
) as demo:
    gr.Markdown(INTROTXT)
    gr.TabbedInterface(
        [vctk, clone, lj],
        ["Multi-Voice", "Clonagem de Voz", "Ingles"],
    )
    gr.Markdown(
        """
"""
    )

if __name__ == "__main__":
    demo.queue(api_open=False, max_size=15).launch(show_api=False,server_name="0.0.0.0")
