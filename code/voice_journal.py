# journal_assistant.py
import os
import datetime
from pathlib import Path
import re
import shutil
import whisper
import ollama

# Configuration
AUDIO_DIR = Path("audio_recordings")
JOURNAL_DIR = Path("journal_notes")
ENTITIES_DIR = Path("journal_entities")

# Create directories if they don't exist
for directory in [AUDIO_DIR, JOURNAL_DIR, ENTITIES_DIR]:
    directory.mkdir(exist_ok=True)


def transcribe_audio(audio_path, language=None):
    """Transcribe audio using Whisper"""
    print(f"Transcribing {audio_path}...")
    model = whisper.load_model("medium")  # Adjust model size as needed

    if language:
        result = model.transcribe(str(audio_path), language=language)
    else:
        result = model.transcribe(str(audio_path))

    return result["text"], result["language"]


def extract_entities_simple(text):
    """Simple entity extraction (to be enhanced later)"""
    entities = []

    # Basic pattern for names: Capitalized words
    name_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    names = re.findall(name_pattern, text)

    # Filter out common words that might be capitalized
    common_words = [
        "I",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    filtered_names = [name for name in names if name not in common_words]

    entities = [(name, "person") for name in filtered_names]
    return entities


def improve_with_ai(transcribed_text, detected_language):
    """Improve transcription with Ollama"""
    print("Improving text with AI...")

    if detected_language == "uk":
        prompt = f"""Нижче наведено транскрибований особистий аудіощоденник. Будь ласка:
        1. Структуруйте це як належний запис у щоденнику, зберігаючи ВСІ думки та ідеї
        2. Виправте будь-які помилки транскрипції або нечіткі формулювання
        3. Організуйте текст у логічні абзаци навколо тем
        4. Додайте відповідне форматування (маркери дати, маркери для окремих думок, якщо потрібно)
        5. Визначте та позначте ключові теми, рішення чи повторювані проблеми [у дужках у відповідних місцях]
        6. Збережіть особистий голос та тон - це приватний щоденник
        7. НЕ додавайте інтерпретації або вміст, якого не було в оригіналі
        
        Транскрибований аудіощоденник:
        {transcribed_text}
        """
    else:  # Default to English
        prompt = f"""Below is a transcribed personal audio journal entry. Please:
        1. Structure this as a proper journal entry while preserving ALL thoughts and ideas
        2. Fix any transcription errors or unclear phrasing
        3. Organize into logical paragraphs around themes or topics
        4. Add appropriate formatting (date markers, bullet points for distinct thoughts if needed)
        5. Identify and tag any key themes, decisions, or recurring concerns [in brackets at relevant points]
        6. Maintain the personal voice and tone - this is a private journal
        7. Do NOT add interpretations or content that wasn't in the original
        
        Transcribed audio journal:
        {transcribed_text}
        """

    response = ollama.chat(
        model="mistral", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def create_markdown_entry(improved_text, entities, date_str=None):
    """Create a markdown journal entry with wiki links"""
    if date_str is None:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    time_str = datetime.datetime.now().strftime("%H:%M")

    # Add header
    header = f"# {date_str}\n{time_str}\n\n"

    # Add wiki links
    linked_text = improved_text
    for entity_name, entity_type in entities:
        safe_name = entity_name.replace(" ", "_")
        wiki_link = f"[[{safe_name}|{entity_name}]]"
        # Use regex with word boundaries to avoid partial replacements
        pattern = rf"\b{re.escape(entity_name)}\b"
        linked_text = re.sub(pattern, wiki_link, linked_text)

    full_entry = header + linked_text

    # Save entry
    file_path = JOURNAL_DIR / f"{date_str}.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_entry)

    print(f"Journal entry saved to {file_path}")
    return file_path


def create_or_update_entity_pages(
    entities, date_str, context_snippet="Mentioned in this entry"
):
    """Create or update entity markdown pages"""
    for entity_name, entity_type in entities:
        safe_name = entity_name.replace(" ", "_")
        file_path = ENTITIES_DIR / f"{safe_name}.md"

        # Create new file if it doesn't exist
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# {entity_name}\n\nType: {entity_type}\n\n## Mentions\n\n")

        # Add new mention
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"- [{date_str}]({date_str}.md): {context_snippet}\n")

        print(f"Updated entity page for {entity_name}")


def process_audio_journal(audio_path, language=None, date_str=None):
    """Process an audio journal from start to finish"""
    if date_str is None:
        # Use the current date if not specified
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    # 1. Transcribe audio
    transcribed_text, detected_language = transcribe_audio(audio_path, language)

    # 2. Improve with AI
    improved_text = improve_with_ai(transcribed_text, detected_language)

    # 3. Extract entities
    entities = extract_entities_simple(improved_text)

    # 4. Create journal entry with wiki links
    entry_path = create_markdown_entry(improved_text, entities, date_str)

    # 5. Create/update entity pages
    create_or_update_entity_pages(entities, date_str)

    # 6. Optionally move or archive the processed audio
    audio_filename = os.path.basename(audio_path)
    archive_path = AUDIO_DIR / "processed" / audio_filename
    archive_path.parent.mkdir(exist_ok=True)
    shutil.copy(audio_path, archive_path)

    return entry_path


if __name__ == "__main__":
    print("🎙️ Audio Journal Assistant 📝")
    print("=============================")

    # Simple CLI interface
    audio_path = input("Enter the path to your audio file: ")
    lang_choice = (
        input("Enter language (en/uk), or leave empty for auto-detection: ")
        .strip()
        .lower()
    )

    if lang_choice in ["en", "uk"]:
        process_audio_journal(audio_path, language=lang_choice)
    else:
        process_audio_journal(audio_path)
