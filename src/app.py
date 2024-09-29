import logging

import pandas as pd
import streamlit as st
from google.cloud import texttospeech

from common import ACCEPTED_FILE_INPUTS, parse_configs
from models import TTSModel, VertexTranscriptModel
from utils import setup_vertex

CONFIGS_PATH = "configs.yaml"
VOICE_DATA_PATH = "./assets/google_tts_voice_data.csv"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

voice_data = pd.read_csv(VOICE_DATA_PATH).query("language == 'en-US'")

st.set_page_config(
    page_title="PodfAI",
    page_icon="🎙",
)


# Run when the app is initialized
if not st.session_state:
    configs = parse_configs(CONFIGS_PATH)

    transcript_model = VertexTranscriptModel(configs=configs["transcript"])
    tts_model = TTSModel()

    setup_vertex(configs["vertex"]["project"], configs["vertex"]["location"])
    transcript_model.setup()
    tts_model.setup()

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
        transcript = transcript_model.generate_transcript(uploaded_files)

        transcript_formatted = transcript_model.format_transcript(transcript)
        podcast_audio = tts_model.transcript_to_speech(
            transcript_formatted, host_voice, guest_voice
        )

        st.write("### Audio content")
        st.audio(podcast_audio)

        st.write("### Transcript")
        st.write(transcript)
    else:
        st.error("Upload a file.")
