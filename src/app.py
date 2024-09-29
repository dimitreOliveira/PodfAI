import logging

import pandas as pd
import streamlit as st
import vertexai
from google.cloud import texttospeech
from vertexai.generative_models import GenerativeModel, Part

from common import configs

VOICE_DATA_PATH = "./assets/google_tts_voice_data.csv"

ACCEPTED_FILE_INPUTS = [
    "txt",
    "md",
    "pdf",
]


def process_file_input(uploaded_files: list):
    logger.info("Processing file input")
    files = []

    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension in ["txt", "md"]:
            files.append(Part.from_text(uploaded_file.read().decode()))
        elif file_extension in ["pdf"]:
            files.append(
                Part.from_data(uploaded_file.read(), mime_type="application/pdf")
            )

    return files


def generate_transcript(
    uploaded_files: list, prompt: str, model: GenerativeModel
) -> str:
    logger.info("Generating transcript")
    responses = model.generate_content(
        [*uploaded_files, prompt],
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


def tts_fn(prompt, client, voice, audio_config):
    synthesis_input = texttospeech.SynthesisInput(text=prompt)

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )
    return response.audio_content


def transcript_to_speech(
    transcript: list[dict[str]],
    tts_model: texttospeech.TextToSpeechClient,
    host_voice: texttospeech.VoiceSelectionParams,
    guest_voice: texttospeech.VoiceSelectionParams,
) -> bytes:
    logger.info("Generating speeches")
    podcast_audio = bytes()

    for sentence in transcript:
        role = sentence["role"]
        text = sentence["text"]
        if role == "Host":
            response = tts_fn(text, tts_model, host_voice, audio_config)
        elif role == "Guest":
            response = tts_fn(text, tts_model, guest_voice, audio_config)
        podcast_audio += response

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
def setup_tts_model() -> texttospeech.TextToSpeechClient:
    logger.info("TTS model setup starting")
    tts_model = texttospeech.TextToSpeechClient()
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    logger.info("TTS model setup finished")
    return tts_model, audio_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_prompt = """Create content in the format of a podcast from this document, the participants will be a host and a guest.
This podcast must be focused on the content of the file provided, covering the relevant topics.
Each phrase must start with "Host:" or "Guest:" to outline who is speaking.
Make this content very conversational, where the guest and the host are exchanging ideas and asking questions."""

voice_data = pd.read_csv(VOICE_DATA_PATH).query("language == 'en-US'")

st.set_page_config(
    page_title="PodfAI",
    page_icon="🎙",
)


if not st.session_state:
    setup_vertex(configs["project"], configs["location"])
    text_model = setup_text_model(configs["model_id"])
    tts_model, audio_config = setup_tts_model()

st.title("🎙 PodfAI")

st.markdown("### Generate content in podcast style based on any input.")

with st.expander("Customize voices"):
    host_col, guest_col = st.columns([1, 1])

    with host_col:
        host_voice_choice = st.selectbox(
            "Host voice",
            voice_data.description,
            index=27,
        )
        host_data = voice_data.loc[voice_data["description"] == host_voice_choice].iloc[
            0
        ]
        host_voice = texttospeech.VoiceSelectionParams(
            language_code=host_data.language,
            name=host_data.voice,
            ssml_gender=host_data.gender,
        )

    with guest_col:
        guest_voice_choice = st.selectbox(
            "Guest voice",
            voice_data.description,
            index=0,
        )
        guest_data = voice_data.loc[
            voice_data["description"] == guest_voice_choice
        ].iloc[0]
        guest_voice = texttospeech.VoiceSelectionParams(
            language_code=guest_data.language,
            name=guest_data.voice,
            ssml_gender=guest_data.gender,
        )

col_1, col_2 = st.columns([1, 1])

with col_1:
    uploaded_files = st.file_uploader(
        "Upload a file",
        accept_multiple_files=True,
        type=ACCEPTED_FILE_INPUTS,
    )

with col_2:
    generate_btn = st.button("Generate podcast")

if generate_btn:
    if uploaded_files:
        uploaded_files = process_file_input(uploaded_files)
        transcript = generate_transcript(uploaded_files, system_prompt, text_model)

        transcript_formatted = format_transcript(transcript)
        podcast_audio = transcript_to_speech(
            transcript_formatted, tts_model, host_voice, guest_voice
        )

        st.write("### Audio content")
        st.audio(podcast_audio)

        st.write("### Transcript")
        st.write(transcript)
    else:
        st.error("Upload a file.")
