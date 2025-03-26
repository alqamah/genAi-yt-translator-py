import os
import logging
import argparse
import srt # Library for parsing/creating SRT files
import torch
from transformers import pipeline

# --- Configuration ---
# --- CHOOSE YOUR LANGUAGE ---
TARGET_LANGUAGE_CODE = 'hi' # Options: 'hi' (Hindi), 'ur' (Urdu)
# -------------------------

# Define models based on target language
MODEL_MAP = {
    'hi': 'Helsinki-NLP/opus-mt-en-hi',
    'ur': 'Helsinki-NLP/opus-mt-en-ur'
}
MODEL_NAME_TRANSLATE = MODEL_MAP.get(TARGET_LANGUAGE_CODE, 'Helsinki-NLP/opus-mt-en-hi') # Default to Hindi if code invalid

# Directories and Filenames
INPUT_DIR = os.path.join("temp", "subtitle") # Default input directory
OUTPUT_DIR = os.path.join("temp", "translated_subtitle")
OUTPUT_FILENAME_TEMPLATE = "{video_id}_{lang}.srt"
LOG_FILENAME = "translator.log" # Log file specific to this script
DEFAULT_BATCH_SIZE = 16 # Adjust based on GPU memory/CPU capability

# --- Setup Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(LOG_FILENAME)
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
    logger.info("GPU detected. Using CUDA for translation.")
    # You might need device=0 for pipeline if you have multiple GPUs
    # device_arg = 0
else:
    device = torch.device("cpu")
    logger.info("No GPU detected. Using CPU for translation (this might be slow).")
    # device_arg = -1 # pipeline uses -1 for CPU

# --- Test Function ---
def test_translation(sample_text="Hello, how are you?", target_lang=TARGET_LANGUAGE_CODE):
    """Loads the translation model and translates a sample text."""
    logger.info("--- Running Translation Test ---")
    logger.info(f"Target language: {target_lang}")
    model_name = MODEL_MAP.get(target_lang)
    if not model_name:
        logger.error(f"Invalid target language code '{target_lang}'. Cannot perform test.")
        return False

    try:
        logger.info(f"Loading translation model: {model_name}...")
        # Use device index (0 for first GPU, -1 for CPU) for pipeline
        translator = pipeline(f"translation_en_to_{target_lang}", model=model_name, device=0 if device.type == 'cuda' else -1)
        logger.info("Translation model loaded successfully.")

        logger.info(f"Original Text: '{sample_text}'")
        translated_result = translator(sample_text)

        # translator returns a list, even for single input
        if translated_result and isinstance(translated_result, list):
            translation = translated_result[0]['translation_text']
            logger.info(f"Translated Text: '{translation}'")
            logger.info("--- Translation Test Successful ---")
            return True
        else:
            logger.error(f"Translation test failed. Unexpected result format: {translated_result}")
            return False

    except Exception as e:
        logger.exception(f"Error during translation test: {e}")
        return False

# --- Core Translation Function ---
def translate_srt_file(input_srt_path, target_lang, output_dir, batch_size=DEFAULT_BATCH_SIZE):
    """
    Translates the content of an SRT file using a local transformer model.

    Args:
        input_srt_path (str): Path to the input SRT file (e.g., ..._en.srt).
        target_lang (str): Target language code ('hi' or 'ur').
        output_dir (str): Directory to save the translated SRT file.
        batch_size (int): Number of segments to translate at once.

    Returns:
        str: Path to the saved translated SRT file, or None on failure.
    """
    logger.info(f"--- Starting SRT Translation to {target_lang} ---")
    if not os.path.exists(input_srt_path):
        logger.error(f"Input SRT file not found: {input_srt_path}")
        return None

    model_name = MODEL_MAP.get(target_lang)
    if not model_name:
        logger.error(f"Invalid target language code '{target_lang}'. Cannot proceed.")
        return None

    # Extract video ID from filename (assuming format like videoId_en.srt)
    base_filename = os.path.basename(input_srt_path)
    video_id = base_filename.split('_en.srt')[0]
    if not video_id or video_id == base_filename: # Basic check if split worked
        logger.warning(f"Could not reliably determine video ID from filename: {base_filename}. Using 'unknown_video' as ID.")
        video_id = "unknown_video"

    output_filename = OUTPUT_FILENAME_TEMPLATE.format(video_id=video_id, lang=target_lang)
    output_path = os.path.join(output_dir, output_filename)

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")

        # Read and parse the input SRT file
        logger.info(f"Reading SRT file: {input_srt_path}")
        with open(input_srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        original_subs = list(srt.parse(srt_content))
        if not original_subs:
            logger.warning("Input SRT file is empty. No translation needed.")
            # Create an empty output file for consistency? Or return None? Let's create empty.
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("")
            logger.info(f"Empty translated SRT file created at: {output_path}")
            return output_path

        texts_to_translate = [sub.content for sub in original_subs]

        # Load the translation model
        logger.info(f"Loading translation model: {model_name}...")
        translator = pipeline(f"translation_en_to_{target_lang}", model=model_name, device=0 if device.type == 'cuda' else -1)
        logger.info("Translation model loaded.")

        # Translate in batches
        translated_texts = []
        num_segments = len(texts_to_translate)
        num_batches = (num_segments - 1) // batch_size + 1
        logger.info(f"Translating {num_segments} segments in {num_batches} batches (size: {batch_size})...")

        for i in range(0, num_segments, batch_size):
            batch_texts = texts_to_translate[i : i + batch_size]
            try:
                translated_batch = translator(batch_texts)
                # Ensure result is a list of dicts
                if isinstance(translated_batch, list) and all(isinstance(item, dict) and 'translation_text' in item for item in translated_batch):
                     translated_texts.extend([t['translation_text'] for t in translated_batch])
                     logger.info(f"Translated batch {i//batch_size + 1}/{num_batches}")
                else:
                    logger.error(f"Unexpected result format from translator for batch starting at index {i}. Got: {type(translated_batch)}. Skipping batch.")
                    # Add placeholders or skip - adding placeholders might be better
                    translated_texts.extend(["[Translation Error]" for _ in batch_texts])

            except Exception as batch_error:
                logger.error(f"Error translating batch starting at index {i}: {batch_error}")
                logger.warning(f"Adding placeholders for {len(batch_texts)} segments in this batch.")
                translated_texts.extend(["[Translation Error]" for _ in batch_texts])


        if len(translated_texts) != len(original_subs):
            logger.error(f"CRITICAL: Mismatch between original ({len(original_subs)}) and translated ({len(translated_texts)}) segment count! Aborting SRT creation.")
            return None # Prevent creating a potentially corrupt SRT

        # Create new SRT with translated content, preserving timing
        translated_subs = []
        logger.info("Reconstructing translated SRT file...")
        for i, original_sub in enumerate(original_subs):
            new_sub = srt.Subtitle(
                index=original_sub.index,
                start=original_sub.start,
                end=original_sub.end,
                content=translated_texts[i] # Use the corresponding translated text
            )
            translated_subs.append(new_sub)

        # Compose the final SRT string
        translated_srt_content = srt.compose(translated_subs)

        # Save the translated SRT content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_srt_content)

        logger.info(f"Translated SRT file saved successfully to: {output_path}")
        return output_path

    except FileNotFoundError:
        logger.error(f"Input SRT file not found at path: {input_srt_path}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during SRT translation: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate an English SRT subtitle file using local Transformers.")
    parser.add_argument("input_srt_file", help="Path to the input English SRT file (e.g., temp/subtitle/videoID_en.srt).")
    parser.add_argument("-l", "--lang", choices=['hi', 'ur'], default=TARGET_LANGUAGE_CODE,
                        help=f"Target language code ('hi' for Hindi, 'ur' for Urdu). Default: {TARGET_LANGUAGE_CODE}")
    parser.add_argument("-o", "--output-dir", default=OUTPUT_DIR,
                        help=f"Directory to save the translated SRT file. Default: {OUTPUT_DIR}")
    parser.add_argument("-b", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for translation (adjust based on memory). Default: {DEFAULT_BATCH_SIZE}")
    parser.add_argument("--run-test", action="store_true", help="Run a quick translation test before processing the file.")

    args = parser.parse_args()

    # Run the test if requested
    if args.run_test:
        test_successful = test_translation(target_lang=args.lang)
        if not test_successful:
            logger.error("Translation test failed. Please check model and dependencies. Aborting file processing.")
            exit(1)
        print("-" * 20) # Separator after test

    # Proceed with file translation
    saved_file_path = translate_srt_file(
        input_srt_path=args.input_srt_file,
        target_lang=args.lang,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )

    if saved_file_path:
        print(f"\nSuccess! Translated subtitle file created at: {saved_file_path}")
    else:
        print(f"\nTranslation failed. Check '{LOG_FILENAME}' for details.")
        exit(1)