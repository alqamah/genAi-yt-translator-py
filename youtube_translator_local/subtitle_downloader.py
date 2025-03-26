import os
import logging
import argparse
import srt  # Library for parsing/creating SRT files
import datetime # Needed for timedelta objects
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from urllib.parse import urlparse, parse_qs

# --- Configuration ---
TARGET_DIR = os.path.join("temp", "subtitle") # Relative path for output
OUTPUT_FILENAME_TEMPLATE = "{video_id}_en.srt"
LOG_FILENAME = "downloader.log" # Log file specific to this script

# --- Setup Logging ---
# Logs to both file and console
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File Handler
file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# Get the root logger and add handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set root logger level
# Remove existing handlers if any (especially relevant if run multiple times in interactive session)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# --- Helper Functions ---

def extract_video_id(url):
    """Extracts the YouTube video ID from various URL formats."""
    if not url:
        return None
    # Standard URL: youtube.com/watch?v=VIDEO_ID
    # Short URL: youtu.be/VIDEO_ID
    # Embed URL: youtube.com/embed/VIDEO_ID
    parsed_url = urlparse(url)
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            params = parse_qs(parsed_url.query)
            return params.get('v', [None])[0]
        if parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/embed/')[1].split('?')[0]
        if parsed_url.path.startswith('/v/'):
             return parsed_url.path.split('/v/')[1].split('?')[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:].split('?')[0] # Path starts with '/'

    # If it doesn't match known formats, assume it might BE the ID itself
    # Basic check: valid IDs are usually 11 chars, alphanumeric + '-' and '_'
    if len(url) == 11 and url.replace('-', '').replace('_', '').isalnum():
         return url

    logger.warning(f"Could not reliably extract video ID from input: {url}")
    return None


def segments_to_srt(sub_segments):
    """Converts youtube_transcript_api segments (expected to be objects) to SRT format string."""
    subs = []
    logger.info(f"Converting {len(sub_segments)} segments to SRT format.")
    for i, segment in enumerate(sub_segments):
        try:
            # --- Use attribute access instead of dictionary keys ---
            start_time = segment.start
            duration = segment.duration
            text = segment.text
            # -------------------------------------------------------

            end_time = start_time + duration
            # Clean text: remove leading/trailing whitespace and replace internal newlines with spaces
            cleaned_text = text.strip().replace('\n', ' ')

            # Create srt.Subtitle object using datetime.timedelta
            sub = srt.Subtitle(
                index=i + 1,
                start=datetime.timedelta(seconds=start_time),
                end=datetime.timedelta(seconds=end_time),
                content=cleaned_text
            )
            subs.append(sub)
        except AttributeError as e:
            # Log if attribute access fails (maybe it's a dictionary sometimes?)
            logger.error(f"AttributeError processing segment {i+1}: {e}")
            logger.error(f"Problematic segment data type: {type(segment)}, content: {segment}")
            continue # Skip problematic segment
        except Exception as e:
            # Catch any other unexpected errors during processing of a single segment
            logger.error(f"Unexpected error processing segment {i+1}: {e}")
            logger.error(f"Problematic segment data type: {type(segment)}, content: {segment}")
            continue # Skip problematic segment


    if not subs:
        logger.warning("No valid segments were converted to SRT format.")
        return "" # Return empty string if no subs were successfully created

    # Use the srt library's compose function
    try:
        composed_srt = srt.compose(subs)
        return composed_srt
    except Exception as e:
        logger.exception(f"Error composing final SRT string: {e}") # Use .exception to log stack trace
        return None # Indicate failure at the final step

# --- Core Function ---

def download_english_subtitles(video_id, output_dir):
    """
    Downloads English subtitles for a given video ID and saves them as an SRT file.

    Args:
        video_id (str): The YouTube video ID.
        output_dir (str): The directory to save the SRT file in.

    Returns:
        str: The path to the saved SRT file, or None if download failed.
    """
    if not video_id:
        logger.error("No valid video ID provided.")
        return None

    logger.info(f"Attempting to download 'en' subtitles for video ID: {video_id}")
    output_filename = OUTPUT_FILENAME_TEMPLATE.format(video_id=video_id)
    output_path = os.path.join(output_dir, output_filename)

    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try fetching manual English transcript first
        try:
            logger.info("Looking for manually created 'en' transcript...")
            transcript = transcript_list.find_transcript(['en'])
            logger.info("Found manually created 'en' transcript.")
        except NoTranscriptFound:
            logger.warning("No manual 'en' transcript found.")
            available_langs = [t.language for t in transcript_list]
            logger.info(f"Available languages: {available_langs}")
            # If manual not found, try generated English transcript
            try:
                logger.info("Looking for auto-generated 'en' transcript...")
                transcript = transcript_list.find_generated_transcript(['en'])
                logger.info("Found auto-generated 'en' transcript.")
            except NoTranscriptFound:
                logger.error(f"No transcript (manual or generated) found for 'en'. Cannot proceed.")
                return None # Critical failure if no english subs at all

        # Fetch the transcript segments
        sub_segments = transcript.fetch()
        logger.info(f"Successfully fetched {len(sub_segments)} subtitle segments.")

        if not sub_segments:
             logger.warning("Fetched transcript is empty. No SRT file will be generated.")
             return None

        # Convert segments to SRT format
        srt_content = segments_to_srt(sub_segments)

        # Check if srt_content generation failed
        if srt_content is None:
             logger.error("Failed to generate SRT content string.")
             return None
        if not srt_content and sub_segments: # If content is empty string but segments existed
            logger.warning("SRT content is empty after processing segments. File will be empty.")
            # Decide if you want to proceed or return None here.
            # Let's proceed but the file will be empty.

        # Save the SRT content to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        logger.info(f"Subtitles saved successfully in SRT format to: {output_path}")
        return output_path

    except TranscriptsDisabled:
        logger.error(f"Transcripts are disabled for video: {video_id}")
        return None
    except Exception as e:
        # Log the full exception details to the file/console
        logger.exception(f"An unexpected error occurred during subtitle download: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download English subtitles for a YouTube video.")
    parser.add_argument("video_url_or_id", help="The URL or Video ID of the YouTube video.")
    # Optional: Add argument for output directory if you want to override TARGET_DIR
    # parser.add_argument("-o", "--output", help="Directory to save the subtitle file.", default=TARGET_DIR)

    args = parser.parse_args()

    extracted_id = extract_video_id(args.video_url_or_id)

    if extracted_id:
        # Call the main function using the default TARGET_DIR
        saved_file_path = download_english_subtitles(extracted_id, TARGET_DIR)

        if saved_file_path:
            print(f"\nSuccess! Subtitle file created at: {saved_file_path}")
            # You could potentially print the path to stdout for scripting purposes
            # print(saved_file_path)
        else:
            print(f"\nFailed to download subtitles. Check '{LOG_FILENAME}' for details.")
            # Exit with a non-zero code to indicate failure in scripts
            exit(1)
    else:
        print("\nCould not extract a valid YouTube Video ID from the input.")
        logger.error(f"Invalid video input provided: {args.video_url_or_id}")
        exit(1)