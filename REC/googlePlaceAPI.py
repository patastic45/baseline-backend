import os
import googlemaps
# import sqlite3 # REMOVED: No longer needed
import numpy as np
from nltk.corpus import stopwords
import pandas as pd # Still used by trainLSTM
import re
from collections import defaultdict
import logging
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.util import ngrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Keep for trainLSTM
from datetime import datetime, timedelta # Added for cache check

# --- IMPORTANT: Import db and models from your main app file ---
# Adjust this import based on your main Flask file name and structure
try:
    from extensions import db, PlaceData, Review
    SQLALCHEMY_AVAILABLE = True
    # logging.info("Successfully imported db and models from extensions.") # Optional log
except ImportError as e:
    logging.error(f"Could not import db/models from extensions: {e}. Database operations will fail.")
    # Set to None so checks later will fail gracefully
    db = None
    PlaceData = None
    Review = None
    SQLALCHEMY_AVAILABLE = False


class APICall:

    def __init__(self):
        # Set up logging
        # Basic config is likely set in main app file, but setting here ensures logger exists
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Load lexicon with error handling (Keep this)
        try:
            # Ensure the path is correct relative to where the app runs
            lexicon_path = "REC/NRC-VAD-Lexicon-v2.1.txt" # Ensure this path is accessible from where Flask runs
            self._vad_dict, self._max_ngram = self._load_vad_lexicon(lexicon_path)
            if not self._vad_dict:
                 self.logger.warning(f"VAD lexicon loaded empty from {lexicon_path}. Check path and file content.")
            else:
                 self.logger.info(f"Loaded VAD lexicon with {len(self._vad_dict)} entries. Max N-gram: {self._max_ngram}")
        except Exception as e:
            self.logger.error(f"Failed to load VAD lexicon: {e}", exc_info=True)
            self._vad_dict = {}
            self._max_ngram = 1

        # --- REMOVED SQLite Initialization ---
        # self._con = sqlite3.connect("places.db")
        # self._cursor = self._con.cursor()
        # self._create_tables() # Table creation is now handled by db.create_all() in main app

        # --- Keep other initializations ---
        self._modelPredictedEnergy= []
        self._modelPredictedValence = []
        self._neutral_threshold = .2
        self._max_places = 1 # Consider making this configurable
        self._debug_info = defaultdict(list)


        self._use_spacy = False


        # Load comprehensive stop words list (Keep this)
        try:
            nltk.data.find('corpora/stopwords') # Check if downloaded
            self._stop_words = set(stopwords.words('english'))
            # Add contractions and other common stop words
            additional_stop_words = {
                "a", "an", "the", "and", "but", "or", "because", "as", "until",
                "while", "of", "at", "by", "for", "with", "about", "against",
                "between", "into", "through", "during", "before", "after", "above",
                "below", "to", "from", "up", "down", "in", "out", "on", "off",
                "over", "under", "again", "further", "then", "once", "here",
                "there", "when", "where", "why", "how", "all", "any", "both",
                "each", "few", "more", "most", "other", "some", "such", "no",
                "nor", "not", "only", "own", "same", "so", "than", "too", "very",
                "s", "t", "can", "will", "just", "don", "don't", "should", "now",
                "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't",
                "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't",
                "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn",
                "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn",
                "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn",
                "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't"
            }
            self._stop_words.update(additional_stop_words)
        except LookupError:
            self.logger.error("NLTK stopwords not found. Please download them: Run `python -m nltk.downloader stopwords`")
            self._stop_words = set() # Fallback to empty set


        # Set up Google Maps client using environment variable
        gmaps_api_key = os.getenv('GOOGLE_PLACES_API_KEY')
        if not gmaps_api_key:
            self.logger.error("GOOGLE_PLACES_API_KEY environment variable not set! Google Maps API calls will fail.")
            self._gmaps = None # Ensure client is None if key is missing
        else:
             try:
                 self._gmaps = googlemaps.Client(key=gmaps_api_key)
                 self.logger.info("Google Maps client initialized.")
             except Exception as e:
                 self.logger.error(f"Failed to initialize Google Maps client: {e}", exc_info=True)
                 self._gmaps = None


    # --- REMOVED _create_tables method ---

    def _load_vad_lexicon(self, file_path):
        """Load the VAD lexicon with enhanced error handling"""
        vad_dict = {}
        max_ngram = 1
        try:
            # Ensure file exists before trying to open
            abs_path = os.path.abspath(file_path)
            if not os.path.exists(abs_path):
                 self.logger.error(f"Lexicon file not found at resolved path: {abs_path}")
                 return {}, 1
            with open(abs_path, 'r', encoding="utf-8") as file:
                for line_num, line in enumerate(file, 1): # Add line numbers for errors
                    parts = line.strip().split('\t')
                    if len(parts) != 4:
                        continue # Skip malformed lines silently or log warning
                    word, val_str, aro_str, dom_str = parts
                    word = word.lower()
                    try:
                        # Convert to float and scale from [-1, 1] to [0, 1]
                        valence = (float(val_str) + 1) / 2.0
                        arousal = (float(aro_str) + 1) / 2.0
                        dominance = (float(dom_str) + 1) / 2.0

                        # Clip values to ensure they are within [0, 1] range
                        vad_dict[word] = {
                            "valence": max(0.0, min(1.0, valence)),
                            "arousal": max(0.0, min(1.0, arousal)),
                            "dominance": max(0.0, min(1.0, dominance))
                        }
                        ngram = len(word.split())
                        if ngram > max_ngram: max_ngram = ngram
                    except ValueError:
                        continue # Skip lines with invalid numbers silently or log warning
            # Log message moved to __init__
            return vad_dict, max_ngram
        except Exception as e:
            self.logger.error(f"Error loading lexicon from {file_path}: {e}", exc_info=True)
            return {}, 1

    def _tokenize_text(self, text):
        """Tokenize text using spaCy if available, or fall back to regex"""
        if self._use_spacy:
            try:
                #doc = self._nlp(text.lower())
                # Using lemma_ for base form, filtering stops/punct/space
                tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
                return tokens
            except Exception as e:
                 self.logger.warning(f"spaCy processing failed: {e}. Falling back to regex.")
                 return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
        else:
            # Basic regex tokenization
            return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())

    def _get_ngrams(self, tokens, max_ngram):
         """Get all n-grams up to max_ngram from tokens"""
         all_ngrams = []
         num_tokens = len(tokens)
         effective_max_ngram = min(max_ngram, num_tokens)
         for n in range(1, effective_max_ngram + 1):
             for i in range(num_tokens - n + 1):
                 all_ngrams.append(" ".join(tokens[i:i+n]))
         return all_ngrams

    def _is_stop_word(self, phrase):
         """Check if phrase is a stop word or contains only stop words"""
         if not self._stop_words: return False
         if ' ' not in phrase: return phrase in self._stop_words
         return all(word in self._stop_words for word in phrase.split())

    def _preprocess_review(self, text, lstm=False):
        """Tokenize and extract phrases with improved preprocessing"""
        # Clean text
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'&\w+;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if lstm:
            # --- Placeholder for LSTM Prediction ---
            # Needs trained model and tokenizer loaded in __init__
            self.logger.warning("LSTM prediction path in _preprocess_review not fully implemented.")
            return [0.5, 0.5] # Returning neutral as placeholder

        # --- VAD Path ---
        emoticons = re.findall(r'[:;=]-?[)D(\[\]{}|/\\pPoO]', text)
        tokens = self._tokenize_text(text)

        # Negation handling
        negated = [False] * len(tokens)
        negation_words = {'not', 'no', 'never', 'neither', 'nor', 'barely', 'hardly', 'scarcely', "don't", "doesn't", "didn't", "wasn't", "weren't", "isn't", "aren't", "haven't", "hasn't", "hadn't", "won't", "wouldn't", "can't", "cannot", "couldn't", "shouldn't", "mustn't"}
        for i, token in enumerate(tokens):
            if token in negation_words:
                for j in range(i+1, min(i+4, len(tokens))): negated[j] = True

        found_phrases = []
        # Check unigrams
        for i, token in enumerate(tokens):
             if token in self._vad_dict and not self._is_stop_word(token):
                 scores = dict(self._vad_dict[token])
                 is_neg = negated[i]
                 if is_neg: scores['valence'] = 1.0 - scores['valence']
                 found_phrases.append((token, scores, is_neg))

        # Check n-grams
        if self._max_ngram > 1 and len(tokens) > 1:
             all_possible_ngrams = self._get_ngrams(tokens, self._max_ngram)
             for phrase in all_possible_ngrams:
                 # Check length > 1 to avoid duplicating unigrams already checked
                 if len(phrase.split()) > 1 and phrase in self._vad_dict and not self._is_stop_word(phrase):
                      is_neg = any(neg_word in phrase for neg_word in negation_words)
                      scores = dict(self._vad_dict[phrase])
                      if is_neg: scores['valence'] = 1.0 - scores['valence']
                      found_phrases.append((phrase, scores, is_neg))

        # Handle emoticons
        for emoticon in emoticons:
            if any(c in emoticon for c in '):D]}>'): found_phrases.append((emoticon, {'valence': 0.9, 'arousal': 0.7, 'dominance': 0.6}, False))
            elif any(c in emoticon for c in '(:([{<'): found_phrases.append((emoticon, {'valence': 0.1, 'arousal': 0.6, 'dominance': 0.3}, False))

        return found_phrases

    def _analyze_reviews(self, location_name="unknown", location_id="unknown", reviews="", lstm=False):
        """Analyze the sentiment of reviews using VAD or LSTM prediction results"""
        self._debug_info.clear()
        self._debug_info['location'] = {'id': location_id, 'name': location_name}
        self._debug_info['review_count'] = len(reviews) if reviews else 0
        if not reviews:
            self._debug_info['error'] = "No reviews provided"
            return 0.5, 0.5, self._debug_info # Return neutral if no reviews

        all_phrases_or_preds = []
        raw_review_scores = [] # Stores {'valence': v, 'arousal': a} for each review

        for i, review in enumerate(reviews):
            processed_output = self._preprocess_review(review, lstm)
            if not processed_output: continue

            if lstm:
                valence, arousal = processed_output
                raw_review_scores.append({'review_index': i, 'valence': valence, 'arousal': arousal})
                all_phrases_or_preds.append(processed_output)
            else:
                phrases = processed_output
                all_phrases_or_preds.extend(phrases)
                if phrases:
                    review_valence = np.mean([s['valence'] for _, s, _ in phrases])
                    review_arousal = np.mean([s['arousal'] for _, s, _ in phrases])
                    raw_review_scores.append({'review_index': i, 'valence': review_valence, 'arousal': review_arousal})
                    # Debug info for VAD path
                    # phrase_info = [(p, s['valence'], s['arousal']) for p, s, _ in phrases[:5]]
                    # self._debug_info['reviews'].append({'index': i, 'phrase_count': len(phrases), 'phrases': phrase_info + [('...', 0, 0)] if len(phrases) > 5 else phrase_info})

        if not raw_review_scores:
             self._debug_info['error'] = "No valid scores derived from reviews"
             return 0.5, 0.5, self._debug_info # Return neutral

        # Calculate final scores (average of per-review averages)
        avg_valence = np.mean([r['valence'] for r in raw_review_scores])
        avg_arousal = np.mean([r['arousal'] for r in raw_review_scores])

        # Enhance Contrast
        def enhance_contrast(score, strength=2.0):
            centered = score - 0.5
            transformed = 0.5 + (1 / (1 + np.exp(-strength * centered)) - 0.5) * 2
            return np.clip(transformed, 0, 1)

        final_valence = enhance_contrast(avg_valence, strength=2.5)
        final_arousal = enhance_contrast(avg_arousal, strength=3.0)

        # Update debug info
        self._debug_info['final_scores'] = {'valence': final_valence, 'arousal': final_arousal}
        self._debug_info['score_stats'] = {'raw_valence': avg_valence, 'raw_arousal': avg_arousal}
        if not lstm: self._debug_info['phrase_stats'] = {'total_phrases': len(all_phrases_or_preds)}

        return final_valence, final_arousal, self._debug_info


    def _get_place_data(self, lat, lon, place_type, radius, name, lstm=False):
        """Get place data using SQLAlchemy, with caching"""
        # Check prerequisites
        if not self._gmaps:
            self.logger.error("Google Maps client not initialized. Cannot fetch place data.")
            return []
        if not SQLALCHEMY_AVAILABLE or not db or not PlaceData:
             self.logger.error("SQLAlchemy DB or Models not available. Cannot access database.")
             return []

        try:
            # Fetch nearby places from Google API
            places_api_result = self._gmaps.places_nearby(location=(lat, lon), radius=radius).get("results", [])
            places_to_process = places_api_result[:self._max_places] # Limit processing
            self.logger.info(f"Found {len(places_api_result)} places nearby, processing up to {self._max_places}.")

        except Exception as e:
            self.logger.error(f"Google Places API error: {e}", exc_info=True)
            return []

        processed_place_data = [] # Stores results for returning
        cache_max_age = timedelta(days=7) # Use timedelta for comparison

        for place_summary in places_to_process:
            place_id = place_summary.get("place_id")
            if not place_id:
                self.logger.warning("Skipping place with no place_id.")
                continue

            place_name_summary = place_summary.get("name", "Unknown Place")

            # --- Check Cache using SQLAlchemy ---
            try:
                # Use db.session.get for primary key lookup
                # Ensure this runs within an app context if called outside a request
                cached_place = db.session.get(PlaceData, place_id)
                cache_is_valid = False
                if cached_place and cached_place.LastUpdated:
                    # Make comparison timezone-aware if necessary (e.g., use timezone.utc)
                    cache_age = datetime.utcnow() - cached_place.LastUpdated
                    if cache_age < cache_max_age:
                        cache_is_valid = True

                if cache_is_valid:
                    self.logger.info(f"Using valid cache for '{cached_place.Name}' (ID: {place_id})")
                    processed_place_data.append({
                        "ID": cached_place.ID,
                        "Name": cached_place.Name,
                        "Valence": cached_place.Valence,
                        "Energy": cached_place.Energy
                    })
                    continue # Skip fetching fresh data

                # --- Fetch Fresh Data ---
                self.logger.info(f"Fetching fresh details for '{place_name_summary}' (ID: {place_id})")
                details = self._gmaps.place(
                    place_id,
                    fields=["name", "review"], # Fetch only needed fields
                    reviews_sort="most_relevant"
                ).get("result", {})

                place_name_detail = details.get("name", place_name_summary)
                reviews_raw = details.get("reviews", [])
                review_texts = [r.get("text", "") for r in reviews_raw if r.get("text")]

                if not review_texts:
                    self.logger.warning(f"No review text found for '{place_name_detail}' (ID: {place_id})")
                    continue

                # --- Analyze Reviews ---
                valence, energy, debug_info = self._analyze_reviews(
                    location_name=place_name_detail,
                    location_id=place_id,
                    reviews=review_texts,
                    lstm=lstm
                )
                self.logger.info(f"Analysis results for '{place_name_detail}': V={valence:.3f}, E={energy:.3f}")

                # --- Update or Insert Place Data using SQLAlchemy ---
                current_time_utc = datetime.utcnow()
                if cached_place: # Update existing record
                    self.logger.debug(f"Updating cache for {place_id}")
                    cached_place.Name = place_name_detail
                    cached_place.Valence = valence
                    cached_place.Energy = energy
                    cached_place.LastUpdated = current_time_utc
                    # db.session.add(cached_place) # Not strictly needed for updates on existing objects already in session
                else: # Insert new record
                     self.logger.debug(f"Inserting new data for {place_id}")
                     new_place = PlaceData(
                         ID=place_id,
                         Name=place_name_detail,
                         Valence=valence,
                         Energy=energy,
                         LastUpdated = current_time_utc
                     )
                     db.session.add(new_place) # Add the new object to the session

                # --- Store Reviews (Optional) ---
                # Delete old reviews first if updating and storing new ones
                # if cached_place and Review: # Check if Review model is available
                #    Review.query.filter_by(PlaceID=place_id).delete()
                # if Review: # Check if Review model is available
                #    for review_detail in reviews_raw:
                #        review_text = review_detail.get("text")
                #        rating = review_detail.get("rating")
                #        if review_text and len(review_text.strip()) > 10:
                #            new_review = Review(PlaceID=place_id, ReviewText=review_text, Rating=rating)
                #            db.session.add(new_review)

                # Add the newly processed data to our results list
                processed_place_data.append({
                    "ID": place_id,
                    "Name": place_name_detail,
                    "Valence": valence,
                    "Energy": energy
                })

                # Commit changes after processing each place (safer)
                db.session.commit()

            except Exception as e:
                self.logger.error(f"Error processing place '{place_name_summary}' (ID: {place_id}): {e}", exc_info=True)
                db.session.rollback() # Rollback changes for this specific place on error

        # Final commit is not needed if committing after each place
        return processed_place_data


    def _log_debug_info(self, debug_info):
        """Log key debug information"""
        if 'error' in debug_info: self.logger.warning(f"Analysis error: {debug_info['error']}")
        self.logger.debug(f"Reviews processed: {debug_info.get('review_count', 0)}")
        if 'final_scores' in debug_info: self.logger.debug(f"Final Scores: {debug_info['final_scores']}")
        # Add more specific logging as needed
        pass

    def get_location_mood(self, lat, lon, place_type, radius, name, lstm=False):
        """Get the overall mood of places in the area."""
        # This method now relies on _get_place_data which uses SQLAlchemy.
        # Ensure it's called within Flask app context if needed.
        place_data = self._get_place_data(lat, lon, place_type, radius, name, lstm)

        if not place_data:
            self.logger.warning(f"No place data returned for {place_type} at ({lat}, {lon}) for mood calculation.")
            return {"valence": 0.5, "energy": 0.5, "places_analyzed": []}

        # Filter out None values before calculating mean
        all_valences = [p["Valence"] for p in place_data if p.get("Valence") is not None]
        all_energies = [p["Energy"] for p in place_data if p.get("Energy") is not None]

        valence = np.mean(all_valences) if all_valences else 0.5
        energy = np.mean(all_energies) if all_energies else 0.5

        self.logger.info(f"Calculated mood for ({lat},{lon}), type '{place_type}': V={valence:.3f}, E={energy:.3f} from {len(place_data)} place(s).")

        return {
            "valence": np.clip(valence, 0, 1),
            "energy": np.clip(energy, 0, 1),
            "places_analyzed": [p.get("Name", "N/A") for p in place_data]
        }
def clean_text(text):
    """Basic text cleaning."""
    text = str(text).lower() # Ensure input is string and lowercased
    text = re.sub(r'http\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'@\S+', '', text) # Remove mentions
    text = re.sub(r'#\S+', '', text) # Remove hashtags
    text = re.sub(r'[\W_]+', ' ', text) # Replace non-alphanumeric (and underscore) with space
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text
