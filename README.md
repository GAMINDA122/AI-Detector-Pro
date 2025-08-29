# AI Detector Pro

A **professional, Flask-based web application** for detecting whether text is **AI-generated or human-written**. It leverages **Google Gemini API** and NLP techniques to analyze text and provide detailed insights, including AI model probabilities, text features, and highlighted AI-generated sentences.

---

## 🚀 Features

- **Overall AI Probability Score**: Calibrated probability of text being AI-generated.
- **Model-specific Detection**: Scores for ChatGPT, Claude, and Google Gemini.
- **Detailed Feature Analysis**:
  - Perplexity
  - Burstiness
  - Entropy
  - Sentence length variance
  - POS distribution
  - Repetition detection
  - Rare word ratio
  - Dictionary usage (formal, slang, unknown)
  - Typo error scoring
- **Highlight AI-generated Sentences**: Visual highlighting for high-confidence AI sentences.
- **Supports Multiple Text Formats**: PDF, DOCX, images with OCR, and plain text.

---

## 💻 Tech Stack

- **Backend**: Python, Flask
- **NLP & Text Analysis**: NLTK, Textstat, SpellChecker, LanguageTool
- **AI Detection**: Google Gemini API
- **Document Processing**: python-docx, mammoth, fitz, pdf2image, pytesseract, PIL
- **Environment Management**: dotenv
- **Frontend**: HTML, CSS, JavaScript (served via Flask templates)

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/GAMINDA122/AI-Detector-Pro.git
cd AI-Detector-Pro

# Install dependencies
pip install -r requirements.txt

# Create a .env file with your API key
echo "API_KEY=YOUR_GOOGLE_GEMINI_API_KEY" > .env

# Run the Flask app
python app.py
