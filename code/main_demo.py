import os
import whisper
import ollama


def transcribe_and_improve(audio_file, language=None):
    """
    Transcribe an audio file and improve the text quality using Ollama.

    Parameters:
        audio_file (str): Path to audio file
        language (str): Language code ('en' for English, 'uk' for Ukrainian, None for auto-detect)
    """
    # 1. Load Whisper model
    whisper_model = whisper.load_model(
        "medium"
    )  # Choose model size based on your needs

    # 2. Transcribe audio
    print(f"Transcribing {audio_file}...")
    if language:
        result = whisper_model.transcribe(audio_file, language=language)
    else:
        result = whisper_model.transcribe(audio_file)

    transcribed_text = result["text"]
    detected_language = result["language"]

    print(f"\nDetected language: {detected_language}")
    print(f"\nOriginal Transcription:\n{transcribed_text}")

    # 3. Prepare language-specific prompt for Ollama
    # For English
    if detected_language == "en" or language == "en":
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
    else:  # Ukrainian
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

    # 4. Send to Ollama for improvement
    print("\nImproving text with Ollama...")
    response = ollama.chat(
        model="mistral", messages=[{"role": "user", "content": prompt}]
    )

    improved_text = response["message"]["content"]

    # 5. Output improved text
    print(f"\nImproved Text:\n{improved_text}")

    # 6. Save results to files
    base_name = os.path.splitext(audio_file)[0]
    with open(f"{base_name}_original.txt", "w", encoding="utf-8") as f:
        f.write(transcribed_text)

    with open(f"{base_name}_improved.txt", "w", encoding="utf-8") as f:
        f.write(improved_text)

    print(f"\nSaved results to {base_name}_original.txt and {base_name}_improved.txt")

    return improved_text


# Example usage
if __name__ == "__main__":
    # For English audio
    # transcribe_and_improve("path_to_english_audio.mp3", language="en")

    # For Ukrainian audio
    # transcribe_and_improve("path_to_ukrainian_audio.mp3", language="uk")

    # Auto-detect language
    # transcribe_and_improve("path_to_any_audio.mp3")

    # Interactive mode
    audio_path = input("Enter the path to your audio file: ")
    lang_choice = (
        input("Enter language (en/uk), or leave empty for auto-detection: ")
        .strip()
        .lower()
    )

    if lang_choice in ["en", "uk"]:
        transcribe_and_improve(audio_path, language=lang_choice)
    else:
        transcribe_and_improve(audio_path)
