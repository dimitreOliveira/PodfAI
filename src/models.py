import abc
import logging
from typing import Any

import pydantic
from google.cloud import texttospeech
from vertexai.generative_models import GenerativeModel, Part

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRANSCRIPT_SYSTEM_PROMPT = """
Create content in the format of a podcast from given file and document(s).
The general podcast structure should be an introduction to the guest and topics covered followed by the content conversation and some closing ideas.
This podcast must be focused on the content of the file provided, covering the relevant topics.
Make this content very engaging resembling a natural conversation, where the guest and the host are exchanging ideas and asking questions expressing their emotions.
The participants will be a host and a guest so each phrase must start with "Host:" or "Guest:" to outline who is speaking.
"""


class BaseModel(pydantic.BaseModel, abc.ABC):
    model: Any = None
    configs: Any = None

    @abc.abstractmethod
    def setup(self):
        """Run the model setup."""
        pass


class VertexTranscriptModel(BaseModel):
    def setup(self):
        """Run the model setup."""
        logger.info("Initializing transcript model")
        self.model = GenerativeModel(self.configs["model_id"])

    def generate_transcript(self, uploaded_files: list) -> str:
        """Generate the podcast transcript based on one or more files.

        Args:
            uploaded_files (list): list of files used as base for the podcast.

        Returns:
            str: Podcast transcript.
        """
        logger.info("Generating transcript")
        prompt = f"{TRANSCRIPT_SYSTEM_PROMPT}\nThe podcast should have around {self.configs['transcript_len']} words."
        uploaded_files = self.process_file_for_vertex(uploaded_files)

        responses = self.model.generate_content(
            [*uploaded_files, prompt],
            generation_config={
                "max_output_tokens": self.configs["max_output_tokens"],
                "temperature": self.configs["temperature"],
                "top_p": self.configs["top_p"],
            },
        )

        response = responses.candidates[0].content.parts[0].text
        return response

    @staticmethod
    def format_transcript(transcript: str) -> list[dict[str]]:
        """Clean and format the transcript to extract each text passages
             assigned for the host or guest.

        Args:
            transcript (str): Podcast transcript text.

        Returns:
            list[dict[str]]: Formatted podcast transcript.
        """
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

    @staticmethod
    def process_file_for_vertex(uploaded_files: list) -> list[Part]:
        """Process and format files to be sent as part of a Vertex API request.

        Args:
            uploaded_files (list): List of files that will be processed.

        Returns:
            list[Part]: List of processed files.
        """
        logger.info("Processing files for Vertex")
        files = []

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_extension = file_name.split(".")[-1]
            logger.info(f"Processing file '{file_name}'")
            if file_extension in ["txt", "md"]:
                files.append(Part.from_text(uploaded_file.read().decode()))
            elif file_extension in ["pdf"]:
                files.append(
                    Part.from_data(uploaded_file.read(), mime_type="application/pdf")
                )

        return files


class TTSModel(BaseModel):
    def setup(self):
        """Run the model setup."""
        logger.info("Initializing TTS model")
        self.model = texttospeech.TextToSpeechClient()
        self.configs = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

    def tts_fn(self, text: str, voice: texttospeech.VoiceSelectionParams) -> bytes:
        """Generate speech from a text input based on voice parameters.

        Args:
            text (str): Text input.
            voice (texttospeech.VoiceSelectionParams):
                Voice parameters used for the voice generation.

        Returns:
            bytes: Generated speech.
        """
        synthesis_input = texttospeech.SynthesisInput(text=text)

        response = self.model.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=self.configs,
        )
        return response.audio_content

    def transcript_to_speech(
        self,
        transcript: list[dict[str]],
        host_voice: texttospeech.VoiceSelectionParams,
        guest_voice: texttospeech.VoiceSelectionParams,
    ) -> bytes:
        """Generate speech for the podcast based on each speaker parameters.

        Args:
            transcript (list[dict[str]]): Formatted podcast transcript text.
            host_voice (texttospeech.VoiceSelectionParams):
                 Parameters describing the host voice.
            guest_voice (texttospeech.VoiceSelectionParams):
                 Parameters describing the guest voice.

        Returns:
            bytes: Full podcast audio.
        """
        logger.info("Generating speeches")
        podcast_audio = bytes()

        for sentence in transcript:
            role = sentence["role"]
            text = sentence["text"]
            if role == "Host":
                response = self.tts_fn(text, host_voice)
            elif role == "Guest":
                response = self.tts_fn(text, guest_voice)
            podcast_audio += response

        return podcast_audio
