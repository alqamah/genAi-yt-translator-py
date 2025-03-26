import os
import logging
import argparse
import srt # Library for parsing SRT files
import torch
import numpy as np
import re # For text splitting
from scipy.io.wavfile import write as write_wav # To save audio
from transformers import AutoProcessor, VitsModel

# --- Configuration ---
# --- Language is typically inferred from input filename, but set a default ---
DEFAULT_TARGET_LANGUAGE_CODE = 'hi' # Options: 'hi' (Hindi), 'ur' (Urdu)
# -------------------------

# Define MMS TTS models based on target language
TTS_MODEL_MAP = {
    'hi': {
        'processor': 'facebook/mms-tts-hin',
        'model': 'facebook/mms-tts-hin'
    },
    'ur': {
        'processor': 'facebook/mms-tts-urd',
        'model': 'facebook/mms-tts-urd'
    }
}

# Directories and Filenames
INPUT_DIR = os.path.join("temp", "translated_subtitle") # Default input directory
OUTPUT_DIR = "output_audio" # Subdirectory for final audio
OUTPUT_FILENAME_TEMPLATE = "{video_id}_{lang}.wav"
LOG_FILENAME = "tts.log" # Log file specific to this script
MAX_TTS_CHUNK_LEN = 450 # Max characters per TTS chunk (adjust based on model limits/memory)

# --- Setup Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(LOG_FILENAME, encoding='utf-8') # Explicit UTF-8
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- Setup Device ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("GPU detected. Using CUDA for TTS.")
else:
    device = torch.device("cpu")
    logger.info("No GPU detected. Using CPU for TTS (this might be slow).")


# --- Helper Function for Text Splitting ---
def split_text(text, max_chunk_len=MAX_TTS_CHUNK_LEN):
    """
    Splits text into smaller chunks suitable for TTS processing, trying to respect sentences.
    """
    if not text:
        return []

    # Simple sentence splitting based on common terminators + Urdu full stop
    # Need to handle cases like 'Mr.', 'Mrs.', abbreviations, etc. for better splitting,
    # but this is a basic approach.
    sentence_enders = re.compile(r'(?<=[.!?۔])\s+') # Added Urdu full stop '۔'
    sentences = sentence_enders.split(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If a single sentence is too long, split it mid-sentence
        if len(sentence) > max_chunk_len:
            # If current chunk has content, add it first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Split the long sentence itself
            start = 0
            while start < len(sentence):
                # Find the last space before the max length
                split_point = sentence.rfind(' ', start, start + max_chunk_len)
                if split_point == -1 or split_point <= start: # No space found or only at the beginning
                    split_point = start + max_chunk_len # Force split

                chunks.append(sentence[start:split_point].strip())
                start = split_point + 1 # Move past the space
            continue # Move to the next sentence from the original list

        # Check if adding the sentence exceeds the limit
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_len:
            current_chunk += sentence + " "
        else:
            # Add the previous chunk if it wasn't empty
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk with the current sentence
            current_chunk = sentence + " "

    # Add the last remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Filter out any potentially empty chunks after stripping
    final_chunks = [chunk for chunk in chunks if chunk]
    logger.info(f"Split text into {len(final_chunks)} chunks for TTS.")
    return final_chunks


# --- Core TTS Function ---
def convert_srt_to_audio(input_srt_path, output_dir, target_lang=None):
    """
    Converts text from a translated SRT file to audio using a local MMS TTS model.

    Args:
        input_srt_path (str): Path to the input translated SRT file (e.g., ..._hi.srt).
        output_dir (str): Directory to save the output WAV file.
        target_lang (str, optional): Language code ('hi' or 'ur'). If None, attempts
                                      to infer from the filename (_hi.srt or _ur.srt).
                                      Defaults to DEFAULT_TARGET_LANGUAGE_CODE if inference fails.

    Returns:
        str: Path to the saved WAV file, or None on failure.
    """
    logger.info("--- Starting Text-to-Speech Conversion ---")
    if not os.path.exists(input_srt_path):
        logger.error(f"Input SRT file not found: {input_srt_path}")
        return None

    # --- Infer Language and Model ---
    inferred_lang = None
    base_filename = os.path.basename(input_srt_path)
    if base_filename.endswith("_hi.srt"):
        inferred_lang = 'hi'
    elif base_filename.endswith("_ur.srt"):
        inferred_lang = 'ur'

    if target_lang:
        if inferred_lang and target_lang != inferred_lang:
             logger.warning(f"Specified language '{target_lang}' differs from inferred language '{inferred_lang}' from filename. Using specified '{target_lang}'.")
        effective_lang = target_lang
    elif inferred_lang:
        effective_lang = inferred_lang
        logger.info(f"Inferred target language from filename: '{effective_lang}'")
    else:
        effective_lang = DEFAULT_TARGET_LANGUAGE_CODE
        logger.warning(f"Could not infer language from filename '{base_filename}'. Defaulting to '{effective_lang}'.")

    model_details = TTS_MODEL_MAP.get(effective_lang)
    if not model_details:
        logger.error(f"No TTS model configured for language code '{effective_lang}'. Cannot proceed.")
        return None

    model_name_processor = model_details['processor']
    model_name_model = model_details['model']

    # Extract video ID from filename
    video_id = base_filename.replace(f"_{effective_lang}.srt", "")
    if not video_id or video_id == base_filename:
        logger.warning(f"Could not reliably determine video ID from filename: {base_filename}. Using 'unknown_video' as ID.")
        video_id = "unknown_video"

    output_filename = OUTPUT_FILENAME_TEMPLATE.format(video_id=video_id, lang=effective_lang)
    output_path = os.path.join(output_dir, output_filename)

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")

        # Read and parse the input SRT file
        logger.info(f"Reading translated SRT file: {input_srt_path}")
        with open(input_srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        translated_subs = list(srt.parse(srt_content))

        if not translated_subs:
            logger.warning("Translated SRT file is empty. No audio to generate.")
            return None

        # Concatenate all text content
        # NOTE: This generates ONE single audio file, NOT synchronized with video timing.
        full_text = " ".join([sub.content for sub in translated_subs])
        full_text = full_text.strip() # Remove leading/trailing spaces from concatenated string

        if not full_text:
            logger.warning("No text content found in translated SRT after joining. Skipping TTS.")
            return None

        logger.info(f"Total text length: {len(full_text)} characters.")

        # --- Load TTS Model and Processor ---
        # This can take time and memory, especially first time
        logger.info(f"Loading TTS processor: {model_name_processor}...")
        processor = AutoProcessor.from_pretrained(model_name_processor)
        logger.info(f"Loading TTS model: {model_name_model}...")
        model = VitsModel.from_pretrained(model_name_model).to(device)
        model.eval() # Set model to evaluation mode
        sampling_rate = model.config.sampling_rate
        logger.info(f"TTS model and processor loaded. Sampling rate: {sampling_rate} Hz.")

        # --- Process Text in Chunks ---
        text_chunks = split_text(full_text)
        if not text_chunks:
             logger.warning("Text splitting resulted in no chunks. Skipping TTS.")
             return None

        all_audio_outputs = []
        num_chunks = len(text_chunks)
        logger.info(f"Generating audio for {num_chunks} text chunks...")

        for i, chunk in enumerate(text_chunks):
            logger.info(f"Processing chunk {i+1}/{num_chunks} (length: {len(chunk)} chars)")
            try:
                # Prepare input
                inputs = processor(text=chunk, return_tensors="pt").to(device)

                # Generate waveform
                with torch.no_grad(): # Important for inference
                    output = model(**inputs).waveform

                # Check for empty/invalid output (output shape is usually [1, num_samples])
                if output is None or output.shape[-1] == 0:
                    logger.warning(f"Chunk {i+1} produced no audio samples. Skipping.")
                    continue

                # Move to CPU, convert to numpy array, remove batch dim if present
                audio_waveform = output.squeeze().cpu().numpy()

                # Basic check if waveform is valid
                if audio_waveform.ndim == 0 or audio_waveform.size == 0:
                    logger.warning(f"Chunk {i+1} produced invalid audio waveform (ndim={audio_waveform.ndim}, size={audio_waveform.size}). Skipping.")
                    continue

                all_audio_outputs.append(audio_waveform)
                logger.info(f"Chunk {i+1} audio generated ({len(audio_waveform)/sampling_rate:.2f}s)")

                # Optional: Clear CUDA cache periodically if memory issues occur
                # if device.type == 'cuda':
                #    torch.cuda.empty_cache()

            except Exception as chunk_e:
                logger.exception(f"Error processing TTS chunk {i+1}: {chunk_e}. Skipping chunk.")
                # Log the chunk text that failed (first 100 chars)
                logger.debug(f"Failed chunk text: '{chunk[:100]}...'")


        if not all_audio_outputs:
            logger.error("No audio was successfully generated from any chunk. Cannot save file.")
            return None

        # Concatenate all audio chunks
        logger.info("Concatenating audio chunks...")
        final_audio = np.concatenate(all_audio_outputs)

        # Normalize and convert to int16 for WAV format
        # VITS output is typically float32 between -1 and 1
        logger.info("Normalizing and converting audio to 16-bit PCM...")
        # Check for silence or very low audio before normalizing
        max_val = np.max(np.abs(final_audio))
        if max_val == 0: # Avoid division by zero if completely silent
            logger.warning("Generated audio is completely silent.")
            final_audio_int16 = np.zeros_like(final_audio, dtype=np.int16)
        else:
            normalized_audio = final_audio / max_val
            final_audio_int16 = np.int16(normalized_audio * 32767)

        # Save the final audio as a WAV file
        logger.info(f"Saving final audio ({len(final_audio_int16)/sampling_rate:.2f}s) to {output_path}")
        write_wav(output_path, sampling_rate, final_audio_int16)

        logger.info(f"Audio saved successfully to: {output_path}")
        return output_path

    except FileNotFoundError:
        logger.error(f"Input SRT file not found at path: {input_srt_path}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during TTS conversion: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert translated SRT file text to speech using local MMS TTS.")
    parser.add_argument("input_srt_file", help="Path to the input translated SRT file (e.g., temp/translated_subtitle/videoID_hi.srt).")
    parser.add_argument("-l", "--lang", choices=['hi', 'ur'], default=None, # Default to None to trigger inference
                        help="Target language code ('hi', 'ur'). If omitted, tries to infer from filename.")
    parser.add_argument("-o", "--output-dir", default=OUTPUT_DIR,
                        help=f"Directory to save the output WAV file. Default: {OUTPUT_DIR}")

    args = parser.parse_args()

    # Call the main TTS function
    saved_audio_path = convert_srt_to_audio(
        input_srt_path=args.input_srt_file,
        output_dir=args.output_dir,
        target_lang=args.lang # Pass the language if specified, otherwise it will be inferred
    )

    if saved_audio_path:
        print(f"\nSuccess! Audio file created at: {saved_audio_path}")
    else:
        print(f"\nText-to-Speech conversion failed. Check '{LOG_FILENAME}' for details.")
        exit(1)