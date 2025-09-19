import os
import re
import json
import math
import random
from collections import Counter
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import nltk
import textstat
from spellchecker import SpellChecker
import language_tool_python
import numpy as np
from docx import Document
import mammoth
import fitz
from PIL import Image
import io
from io import BytesIO
import pytesseract
from pdf2image import convert_from_path


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini API
API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize tools
spell_checker = SpellChecker()
grammar_tool = language_tool_python.LanguageTool('en-US')

class AIDetector:
    def __init__(self):
        self.model = model
    
    # 1. Input & Preprocessing Functions
    def normalize_text(self, text):
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s.,!?;:\'\"()-]', '', text)
        return text
    
    def tokenize_text(self, text):
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text.lower())
        return {
            'sentences': sentences,
            'words': words,
            'word_count': len(words),
            'sentence_count': len(sentences)
        }
    
    # 2. Feature Extraction Functions
    def calculate_perplexity(self, text):
        try:
            prompt = f"""Analyze this text for predictability on a scale of 0-100 where:
            0 = Very unpredictable/creative human writing
            100 = Very predictable/formulaic AI writing
            
            Text: "{text[:10000]}..."
            
            Return only a number between 0-100."""
            
            response = self.model.generate_content(prompt)
            score = float(re.findall(r'\d+', response.text)[0])
            return min(max(score, 0), 100)
        except:
            words = text.split()
            if len(words) < 10:
                return 50
            word_freq = Counter(words)
            total_words = len(words)
            perplexity = sum(freq * math.log(freq/total_words) for freq in word_freq.values())
            return min(abs(perplexity * 10), 100)
    
    def calculate_burstiness(self, text):
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return 50
        
        lengths = [len(sentence.split()) for sentence in sentences]
        mean_length = np.mean(lengths)
        variance = np.var(lengths)
        
        if mean_length == 0:
            return 50
        
        burstiness = variance / mean_length
        return min(burstiness * 10, 100)
    
    def calculate_entropy(self, text):
        words = text.lower().split()
        if not words:
            return 0
        
        word_freq = Counter(words)
        total_words = len(words)
        
        entropy = -sum((freq/total_words) * math.log2(freq/total_words) 
                      for freq in word_freq.values())
        
        max_entropy = math.log2(len(word_freq))
        return (entropy / max_entropy * 100) if max_entropy > 0 else 0
    
    def avg_sentence_length(self, text):
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return 0
        total_words = sum(len(sentence.split()) for sentence in sentences)
        return total_words / len(sentences)
    
    def sentence_length_variance(self, text):
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return 0
        lengths = [len(sentence.split()) for sentence in sentences]
        return np.var(lengths)
    
    def pos_distribution(self, text):
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        pos_counts = Counter(tag for word, tag in pos_tags)
        total_tags = len(pos_tags)
        pos_dist = {tag: (count/total_tags)*100 for tag, count in pos_counts.items()}
        return pos_dist
    
    def detect_repetition(self, text):
        words = text.lower().split()
        word_freq = Counter(words)
        repeated_words = sum(1 for freq in word_freq.values() if freq > 2)
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        bigram_freq = Counter(bigrams)
        repeated_phrases = sum(1 for freq in bigram_freq.values() if freq > 1)
        total_repetitions = repeated_words + repeated_phrases
        repetition_score = min((total_repetitions / len(words)) * 100, 100)
        return repetition_score
    
    def rare_word_ratio(self, text):
        words = [word.lower() for word in nltk.word_tokenize(text) if word.isalpha()]
        if not words:
            return 0
        difficult_words = textstat.difficult_words(text)
        return (difficult_words / len(words)) * 100
    
    def dictionary_usage(self, text):
        words = [word.lower() for word in nltk.word_tokenize(text) if word.isalpha()]
        if not words:
            return {'formal': 0, 'slang': 0, 'unknown': 0}
        
        unknown_words = spell_checker.unknown(words)
        unknown_ratio = (len(unknown_words) / len(words)) * 100
        
        formal_indicators = [
            'therefore', 'furthermore', 'consequently', 'nevertheless',
            'moreover', 'thus', 'henceforth', 'accordingly', 'whereas',
            'inasmuch', 'heretofore', 'therein', 'aforementioned', 'pursuant',
            'notwithstanding', 'hence', 'ergo', 'hitherto', 'aforetime',
            'respectively', 'hereafter', 'wherefore'
        ]

        slang_indicators = [
            'gonna', 'wanna', 'yeah', 'nah', 'cool', 'awesome',
            'lit', 'dope', 'bruh', 'bro', 'sis', 'yo', 'fam',
            'lmao', 'lol', 'omg', 'idk', 'btw', 'cuz', 'ayy',
            'yolo', 'meh', 'nahh', 'fr', 'cap', 'sus'
        ]
        
        formal_count = sum(1 for word in words if word in formal_indicators)
        slang_count = sum(1 for word in words if word in slang_indicators)
        
        return {
            'formal': (formal_count / len(words)) * 100,
            'slang': (slang_count / len(words)) * 100,
            'unknown': unknown_ratio
        }
    
    def typo_error_score(self, text):
        words = [word for word in nltk.word_tokenize(text) if word.isalpha()]
        if not words:
            return 0
        misspelled = spell_checker.unknown(words)
        return (len(misspelled) / len(words)) * 100
    
    # 3. Model Scoring Functions
    def predict_chatgpt_probability(self, features, text):
        try:
            prompt = f"""You are an AI text detector.
                        Analyze if the following text was written by ChatGPT (GPT-3.5 or GPT-4).

                        Text: "{text[:10000]}..."

                        Answer strictly with a single number from 0 to 100 
                        (no words, no explanation)."""
        
            response = self.model.generate_content(prompt)
            numbers = re.findall(r'\d+', response.text)
            score = float(numbers[0]) if numbers else 50.0  # default mid-point if unclear
            score = min(max(score, 0), 100)
        
            # Blend with features for more stability
            feature_score = (features['perplexity'] * 0.3 +
                            (100 - features['burstiness']) * 0.2 +
                            features['repetition'] * 0.2 +
                            (100 - features['typo_score']) * 0.3)
        
            return round((score * 0.7 + feature_score * 0.3), 2)
        except:
            # Improved fallback using more balanced weights
            score = (features['perplexity'] * 0.25 + 
                    (100 - features['burstiness']) * 0.25 + 
                    features['repetition'] * 0.25 +
                    (100 - features['typo_score']) * 0.25)
            return min(round(score, 2), 100)
    
    def predict_claude_probability(self, features, text):
        try:
            prompt = f"""You are an AI detector.
            Analyze if the following text was written by Claude AI.

            Text: "{text[:10000]}..."

            Answer strictly with a single number between 0 and 100."""
        
            response = self.model.generate_content(prompt)
            numbers = re.findall(r'\d+', response.text)
            score = float(numbers[0]) if numbers else 50.0
            score = min(max(score, 0), 100)
        
            # Blend with features typical of Claude
            feature_score = (features['dictionary']['formal'] * 0.4 +
                            features['avg_sentence_length'] * 1.5 +
                            (100 - features['entropy']) * 0.2 +
                            features['rare_word_ratio'] * 0.3)
        
            return round((score * 0.7 + feature_score * 0.3), 2)
        except:
            score = (features['dictionary']['formal'] * 0.35 +
                    features['avg_sentence_length'] * 1.5 +
                    (100 - features['entropy']) * 0.2 +
                    features['rare_word_ratio'] * 0.3)
            return min(round(score, 2), 100)
    
    def predict_gemini_probability(self, features, text):
        try:
            prompt = f"""You are an AI text detector.
            Analyze if the following text was written by Google Gemini.

            Text: "{text[:10000]}..."

            Answer with a single number only (0 to 100)."""
        
            response = self.model.generate_content(prompt)
            numbers = re.findall(r'\d+', response.text)
            score = float(numbers[0]) if numbers else 50.0
            score = min(max(score, 0), 100)
        
            # Blend with Gemini-like traits
            feature_score = (features['entropy'] * 0.3 +
                            features['burstiness'] * 0.3 +
                            (100 - features['typo_score']) * 0.2 +
                            features['dictionary']['slang'] * 0.2)
        
            return round((score * 0.7 + feature_score * 0.3), 2)
        except:
            score = (features['entropy'] * 0.3 +
                    features['burstiness'] * 0.3 +
                    (100 - features['typo_score']) * 0.2 +
                    features['dictionary']['slang'] * 0.2)
            return min(round(score, 2), 100)
    
    def predict_overall_probability(self, features, text):
        try:
            prompt = f"""You are an advanced AI text detection system. 
            Analyze the text and determine the probability that it is AI-generated vs human-written. 
            Consider deeply:
            - Predictability, fluency, and unnatural consistency (AI is too structured)
            - Lack of genuine human errors such as misplaced punctuation, casual dashes, or dropped apostrophes
            - Overuse of formal words vs slang or informal tone
            - Emotional depth, creativity, and originality
            - Repetition patterns and sentence length variation
            - How commas, dashes, and apostrophes are used:
            * Human → sometimes misplaced commas, casual — dashes, or missing apostrophes (dont, cant)
            * AI → commas are always in perfect places, dashes are evenly balanced, apostrophes always correct

            Text: "{text[:10000]}..."

            Return strictly a single number between 0 and 100:
            0 = Definitely human-written
            100 = Definitely AI-generated"""
        
            response = self.model.generate_content(prompt)
            score = float(re.findall(r'\d+', response.text)[0])
            return min(max(score, 0), 100)

        except:
            # Base AI feature indicators
            ai_indicators = [
                features['perplexity'] > 60,              # AI → structured text
                20 <= features['burstiness'] <= 60,       # AI → less variation in sentence length
                features['repetition'] < 10,              # AI → lower self-repetition
                features['typo_score'] <= 2,              # AI → almost no typos
                features['dictionary']['formal'] > 20     # AI → higher formal word usage
            ]

            ai_score = sum(ai_indicators) * 20

            boost = 0
            if features['typo_score'] <= 2:
                boost += 15
            if features['dictionary']['formal'] > 25:
                boost += 20
            if features['entropy'] > 45:
                boost += 10

            # 🔹 Punctuation usage patterns
            sentence_count = max(1, features['sentence_count'])
            comma_ratio = text.count(",") / sentence_count
            dash_ratio = text.count("-") / sentence_count

            # Apostrophes: correct vs missing
            apostrophe_correct = re.findall(r"\b(can't|won't|don't|it's|they're|you're|i'm)\b", text.lower())
            apostrophe_missing = re.findall(r"\b(cant|wont|dont|its|theyre|youre|im)\b", text.lower())
            apostrophe_spacing = re.findall(r"\s'([a-z]+)", text.lower())  # catches " 's" with space

            # 🔹 AI-style → punctuation looks "perfect"
            if 0.8 <= comma_ratio <= 2.0:  
                boost += 15
            if 0.2 <= dash_ratio <= 0.6:   
                boost += 10
            if len(apostrophe_correct) > 5 and len(apostrophe_missing) == 0 and not apostrophe_spacing:
                boost += 15

            # 🔹 Human-style → messy/unexpected punctuation reduces AI score
            if comma_ratio > 3.0 or comma_ratio < 0.3:   # scattered or excessive commas
                boost -= 15
            if dash_ratio > 1.0 or dash_ratio == 0:      # overused or missing dashes
                boost -= 10
            if len(apostrophe_missing) > 2 or apostrophe_spacing:  # missing or spaced apostrophes
                boost -= 20

            # 🔹 Additional human signal: misplaced comma + 's or dash spacing
            if re.search(r"\s, 's", text):
                boost -= 20
            if re.search(r"\s-\s", text):
                boost -= 15
            if re.search(r"\s's", text):
                boost -= 15

            ai_score += boost 

            # 🔹 Human penalty: too many typos = definitely human
            if features['typo_score'] > 10:
                ai_score -= 35

            return min(max(ai_score, 0), 100)
    
    # 4. Highlighting Function
    def highlight_sentences(self, text, overall_probability):
        sentences = nltk.sent_tokenize(text)
        highlighted_sentences = []

        for sentence in sentences:
            sentence_features = self.extract_features(sentence)
            sentence_prob = float(self.predict_overall_probability(sentence_features, sentence))

            # Apply the same conditional 20% reduction rule
            if sentence_prob > 20.0:
                calibrated_sentence_prob = sentence_prob - 20.0
            else:
                calibrated_sentence_prob = 0.0

            calibrated_sentence_prob = round(calibrated_sentence_prob, 1)

            # Highlight ONLY if AI probability is significant
            if calibrated_sentence_prob >= 55:   # Threshold for strong AI detection
                color_class = "bg-red-200"
                label = "AI Generated (High Confidence)"
                highlighted_sentences.append({
                    'text': sentence,
                    'class': color_class,
                    'label': label,
                '   probability': calibrated_sentence_prob
                })
            else:
                # Add normal (uncolored) sentences to keep full text visible
                highlighted_sentences.append({
                    'text': sentence,
                    'class': "",
                    'label': "Human/Not AI",
                    'probability': calibrated_sentence_prob
                })

        return highlighted_sentences
    
    # Main Analysis Function
    def extract_features(self, text):
        normalized_text = self.normalize_text(text)
        tokens = self.tokenize_text(normalized_text)
        
        features = {
            'perplexity': float(self.calculate_perplexity(text)),
            'burstiness': float(self.calculate_burstiness(text)),
            'entropy': float(self.calculate_entropy(text)),
            'avg_sentence_length': float(self.avg_sentence_length(text)),
            'sentence_length_variance': float(self.sentence_length_variance(text)),
            'pos_distribution': {k: float(v) for k, v in self.pos_distribution(text).items()},
            'repetition': float(self.detect_repetition(text)),
            'rare_word_ratio': float(self.rare_word_ratio(text)),
            'dictionary': {k: float(v) for k, v in self.dictionary_usage(text).items()},
            'typo_score': float(self.typo_error_score(text)),
            'word_count': int(tokens['word_count']),
            'sentence_count': int(tokens['sentence_count'])
        }
        
        return features
    
    def analyze_text(self, text, content_type='general'):
        if not text or len(text.strip()) < 10:
            return {'error': 'Text too short for analysis'}
    
        # Extract features
        features = self.extract_features(text)
    
        # Model-specific probabilities
        chatgpt_prob = self.predict_chatgpt_probability(features, text)
        claude_prob = self.predict_claude_probability(features, text)
        gemini_prob = self.predict_gemini_probability(features, text)
    
        # Raw AI probability
        raw_ai_prob = float(self.predict_overall_probability(features, text))
    
        # Apply conditional 20% reduction
        if raw_ai_prob > 15.0:
            calibrated_ai_prob = raw_ai_prob - 0.0
        else:
            calibrated_ai_prob = 0.0
    
        calibrated_ai_prob = self.human_article_safeguard(text, features, calibrated_ai_prob)
    
        # Highlight sentences using calibrated score
        highlighted_sentences = self.highlight_sentences(text, calibrated_ai_prob)
    
        return {
            'overall_score': calibrated_ai_prob,              # calibrated display score
            'breakdown': {
                'human': 0.0,                                 # always 0
                'likely_ai': 0.0,                             # always 0
                'ai_generated': calibrated_ai_prob            # calibrated display score
            },
            'model_detection': {                              # fixed spacing
                'chatgpt': round(chatgpt_prob, 1),
                'claude': round(claude_prob, 1),
                'gemini': round(gemini_prob, 1)
            },
            'features': features,
            'highlighted_sentences': highlighted_sentences,
            'content_type': content_type,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    def human_article_safeguard(self, text, features, ai_score):
        """
        Academic/research safeguard with balance.
        Forces 0% AI ONLY if there are strong signals of human writing.
        Prevents AI research-style text from being misclassified as human.
        """
        try:
            sentences = nltk.sent_tokenize(text)
            words = [w for w in nltk.word_tokenize(text) if any(c.isalpha() for c in w)]
            word_count = len(words)
            sent_count = len(sentences)

            # Skip very short text
            if word_count < 50 or sent_count < 2:
                return ai_score

            # --- Core metrics ---
            avg_sent_len = sum(len(s.split()) for s in sentences) / sent_count
            sent_len_var = np.var([len(s.split()) for s in sentences])
            unique_words = len(set(words))
            ttr = unique_words / word_count
            repetition = features.get('repetition', 0.0)
            complex_punct = sum(1 for s in sentences if ',' in s or ';' in s or '(' in s or ')') 
            complex_ratio = complex_punct / sent_count
            entropy = features.get('entropy', 50)
            rare_word_ratio = features.get('rare_word_ratio', 0)
            typo_score = features.get('typo_score', 0)

            # --- Human-like indicators ---
            human_signals = 0
            if ttr >= 0.45: human_signals += 1              # rich vocabulary
            if repetition > 7: human_signals += 1           # humans repeat a bit
            if typo_score >= 5: human_signals += 1          # humans make small errors
            if sent_len_var > 80: human_signals += 1        # humans vary sentence length more
            if complex_ratio >= 0.45: human_signals += 1    # academic punctuation
            if entropy >= 50: human_signals += 1            # high diversity

            # --- AI-like indicators ---
            ai_signals = 0
            if typo_score <= 2: ai_signals += 1             # too clean → AI
            if repetition < 5: ai_signals += 1              # too little repetition → AI
            if 5 <= avg_sent_len <= 15 and sent_len_var < 60: ai_signals += 1  # too balanced
            if ttr < 0.38: ai_signals += 1                  # shallow vocabulary
            if entropy < 45: ai_signals += 1                # low diversity

            # Decision:
            if human_signals >= 4 and ai_signals <= 1 and word_count >= 50:
                return 0.0  # strong evidence of human academic writing
            else:
                return ai_score  # keep detector’s decision

        except Exception:
            return ai_score



# Initialize detector
detector = AIDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '')
        content_type = data.get('content_type', 'general')
        
        if not text or len(text.strip()) < 10:
            return jsonify({'error': 'Please provide at least 10 characters of text to analyze'})
        
        result = detector.analyze_text(text, content_type)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'})

    
    
if __name__ == '__main__':
    app.run(debug=True)