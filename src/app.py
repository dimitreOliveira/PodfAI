import logging

import numpy as np
import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from whisperspeech.pipeline import Pipeline

from common import configs


def generate_transcript(content: str, prompt: str, model: GenerativeModel) -> str:
    logger.info("Generating transcript")
    file_input = Part.from_text(content)
    responses = model.generate_content(
        [file_input, prompt],
        generation_config={
            "max_output_tokens": configs["max_output_tokens"],
            "temperature": configs["temperature"],
            "top_p": configs["top_p"],
        },
    )

    response = responses.candidates[0].content.parts[0].text
    return response


def format_transcript(transcript: str) -> list[dict[str]]:
    logger.info("Formatting transcript")
    transcript_roles = []

    for sentence in [x for x in transcript.strip().split("\n") if x]:
        sentence = sentence.split()
        role = sentence[0]
        text = " ".join(sentence[1:])
        if "Host" in role:
            transcript_roles.append({"role": "Host", "text": text})
        elif "Guest" in role:
            transcript_roles.append({"role": "Guest", "text": text})

    return transcript_roles


def transcript_to_speech(
    transcript: list[dict[str]], tts_model: Pipeline
) -> list[np.ndarray]:
    logger.info("Generating speeches")
    podcast_audio = []

    for sentence in transcript:
        role = sentence["role"]
        text = sentence["text"]
        if role == "Host":
            speech = tts_model.generate(
                text,
                speaker=host_voice,
                lang="en",
                cps=configs["host_cps"],
            )
            podcast_audio.append(speech.cpu().numpy())
        elif role == "Guest":
            speech = tts_model.generate(
                text,
                speaker=guest_voice,
                lang="en",
                cps=configs["guest_cps"],
            )
            podcast_audio.append(speech.cpu().numpy())

    podcast_audio = np.concatenate(podcast_audio, axis=-1)
    return podcast_audio


@st.cache_resource()
def setup_vertex(project: str, location: str) -> None:
    """Setups the Vertex AI project.

    Args:
        project (str): Vertex AI project name
        location (str): Vertex AI project location
    """
    vertexai.init(project=project, location=location)
    logger.info("Vertex AI setup finished")


@st.cache_resource()
def setup_text_model(model_id: str) -> GenerativeModel:
    logger.info("Text model setup starting")
    model = GenerativeModel(model_id)
    logger.info("Text model setup finished")
    return model


@st.cache_resource()
def setup_tts_model() -> Pipeline:
    logger.info("TTS model setup starting")
    tts_model = Pipeline(
        t2s_ref="whisperspeech/whisperspeech:t2s-v1.95-small-8lang.model",
        s2a_ref="whisperspeech/whisperspeech:s2a-v1.95-medium-7lang.model",
    )
    logger.info("TTS model setup finished")
    return tts_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_prompt = """Create content in the format of a podcast from this document, the participants will be a host and a guest.
This podcast must be focused on the content of the file provided, covering the relevant topics.
Each phrase must start with "Host:" or "Guest:" to outline who is speaking.
Make this content very conversational, where the guest and the host are exchanging ideas and asking questions."""

host_voice = "./voice_samples/host_1.wav"
guest_voice = "./voice_samples/dimitre_olivera.wav"


st.set_page_config(
    page_title="PodifAI",
    page_icon="🎙",
)


if not st.session_state:
    setup_vertex(configs["project"], configs["location"])
    text_model = setup_text_model(configs["model_id"])
    tts_model = setup_tts_model()

st.title("🎙 PodifAI")

st.markdown("### Generate content in a stype of podcast based on any input.")

uploaded_file = st.file_uploader("Upload a file", type=("txt", "md"))

host_col, guest_col = st.columns([1, 1])

with host_col:
    st.markdown("Host voice")
    host_voice = st.file_uploader(
        "Provide a voice sample (optional)",
        type=(".wav", ".mp3", ".ogg"),
    )

with guest_col:
    st.markdown("Guest voice")
    guest_voice = st.file_uploader(
        "Provide a voice sample (optional) ",
        type=(".wav", ".mp3", ".ogg"),
    )

generate_btn = st.button("Generate podcast")

if generate_btn:
    if uploaded_file:
        file_content = uploaded_file.read().decode()
        transcript = generate_transcript(file_content, system_prompt, text_model)

        transcript_formatted = format_transcript(transcript)
        podcast_audio = transcript_to_speech(transcript_formatted, tts_model)

        st.write("### Audio content")
        st.audio(podcast_audio, sample_rate=24000)

        st.write("### Transcipt")
        st.write(transcript)
    else:
        st.error("Upload a file.")
