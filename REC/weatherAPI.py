import math
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import os
import logging
import time
from typing import Dict, Optional, Tuple, List, Union

logging.basicConfig(level=logging.INFO)  # Adjust logging level as needed

class WeatherAPI:

    def __init__(self, preferred_temp_c: float = 21.0):
        """
        Initialize the WeatherAPI with preferences.
        
        Args:
            preferred_temp_c: User's preferred temperature in Celsius (default 21°C/70°F)
        """
        self._load_nrc_vad_lexicon("REC/NRC-VAD-Lexicon-v2.1.txt")
        self.WEATHER_API_KEY = self._get_api_key()
        self.preferred_temp_c = preferred_temp_c
        
        # Season thresholds - adjustable based on region
        self._season_temp_offsets = {
            "winter": -5.0,  # Colder temperatures expected
            "spring": 0.0,
            "summer": 5.0,   # Warmer temperatures expected
            "fall": 0.0
        }

    def _get_api_key(self) -> str:
        """Loads the WeatherAPI key from an environment variable."""
        api_key = os.getenv('WEATHER_PLACES_API_KEY')
        if not api_key:
            raise ValueError("Weather API key not found in environment variable 'WEATHER_API_KEY'")
        return api_key

    def _load_nrc_vad_lexicon(self, file_path: str) -> None:
        """Loads the NRC VAD lexicon from a tab-separated file."""
        try:
            self._nrc_vad_lexicon = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 4:
                        word = parts[0]
                        try:
                            valence = float(parts[1])
                            arousal = float(parts[2])
                            dominance = float(parts[3])  # Adding dominance for future use
                            self._nrc_vad_lexicon[word] = {
                                'valence': valence, 
                                'arousal': arousal,
                                'dominance': dominance
                            }
                        except ValueError:
                            logging.warning(f"Skipping line with invalid VAD values: {line.strip()}")
        except FileNotFoundError:
            logging.critical(f"NRC VAD lexicon file not found: {file_path}")
            raise

    def get_weather_data(self, lat: float, lon: float, retries: int = 3, delay: int = 2) -> Optional[Dict]:
        """
        Retrieves weather data from the WeatherAPI and calculates valence/arousal.

        Args:
            lat: Latitude.
            lon: Longitude.
            retries: Number of times to retry the API call.
            delay: Delay (seconds) between retries.

        Returns:
            A dictionary containing valence, arousal, and raw weather data, or None on failure.
        """

        url = f"http://api.weatherapi.com/v1/current.json?key={self.WEATHER_API_KEY}&q={lat},{lon}&aqi=no"
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # Raise HTTPError for bad status codes
                return self._calculate_valence_arousal(response.json())
            except requests.exceptions.RequestException as e:
                logging.warning(f"WeatherAPI request failed (Attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    logging.error("All WeatherAPI retries failed.")
                    return None
        return None

    def _get_current_season(self, lat: float, date: datetime) -> str:
        """
        Determines current season based on date and hemisphere.
        
        Args:
            lat: Latitude to determine hemisphere
            date: Current date
            
        Returns:
            Season as string: "winter", "spring", "summer", or "fall"
        """
        # Adjust month for southern hemisphere
        month = date.month
        if lat < 0:  # Southern hemisphere
            month = (month + 6) % 12 or 12  # Shift by 6 months
            
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:  # [9, 10, 11]
            return "fall"

    def _extract_weather_data(self, api_response: Dict) -> Dict:
        """Safely extracts and transforms weather data from the API response."""

        current = api_response.get('current', {})
        location = api_response.get('location', {})
        weather_data = {}

        try:
            last_updated = current.get('last_updated')
            if last_updated:
                dt = datetime.strptime(last_updated, "%Y-%m-%d %H:%M")
                weather_data['hour'] = dt.hour
                weather_data['datetime'] = dt
            else:
                logging.warning("Missing 'last_updated'; defaulting to noon.")
                weather_data['hour'] = 12
                weather_data['datetime'] = datetime.now()
        except (ValueError, TypeError) as e:
            logging.error(f"Error parsing 'last_updated': {e}; defaulting to noon.")
            weather_data['hour'] = 12
            weather_data['datetime'] = datetime.now()

        # Extract location data for season calculation
        weather_data['lat'] = location.get('lat', 0.0)
        weather_data['season'] = self._get_current_season(weather_data['lat'], weather_data['datetime'])
        
        weather_data['temp_c'] = current.get('temp_c', 20.0)
        weather_data['feelslike_c'] = current.get('feelslike_c', weather_data['temp_c'])
        weather_data['wind_kph'] = current.get('wind_kph', 0.0)
        weather_data['gust_kph'] = current.get('gust_kph', weather_data['wind_kph'])
        weather_data['precip_mm'] = current.get('precip_mm', 0.0)
        weather_data['humidity'] = current.get('humidity', 65.0)
        weather_data['cloud'] = current.get('cloud', 50.0)
        weather_data['uv'] = current.get('uv', 5.0)
        weather_data['is_day'] = current.get('is_day', 1 if 6 <= weather_data['hour'] <= 18 else 0)
        weather_data['condition_text'] = current.get('condition', {}).get('text', '').lower()
        weather_data['condition_code'] = current.get('condition', {}).get('code', 1000)
        
        # Wind chill and heat index calculations
        if weather_data['temp_c'] < 10 and weather_data['wind_kph'] > 5:
            # Wind chill matters when temp is low and wind is present
            weather_data['wind_impact'] = 'cooling'
        elif weather_data['temp_c'] > 27 and weather_data['humidity'] > 60:
            # Heat index matters when temp is high and humidity is high
            weather_data['wind_impact'] = 'relief'
        else:
            weather_data['wind_impact'] = 'neutral'

        return weather_data

    def _normalize_weather_data(self, weather_data: Dict) -> Dict:
        """Normalizes weather parameters to a 0-1 range with seasonal adjustments."""

        normalized = {}
        
        # Adjust temperature normalization based on season
        season_offset = self._season_temp_offsets.get(weather_data['season'], 0)
        
        # Calculate temperature valence based on distance from preferred temp
        # adjusted by seasonal expectations
        preferred_temp_seasonal = self.preferred_temp_c + season_offset
        temp_distance = abs(weather_data['feelslike_c'] - preferred_temp_seasonal)
        
        # Normalize temp with seasonal context (closer to 1 = closer to preferred temp)
        normalized['temp'] = self._asymptotic_normalize(temp_distance, 0, 15)
        
        # UV normalization depends on is_day
        if weather_data['is_day']:
            normalized['uv'] = self._safe_normalize(weather_data['uv'], 0, 15)
        else:
            normalized['uv'] = 0.0  # No UV impact at night
            
        # Rain normalization with stronger impact for heavier rain (non-linear)
        rain_mm = min(weather_data['precip_mm'], 30)  # Cap at 30mm
        normalized['rain'] = self._safe_normalize(rain_mm**1.5, 0, 30**1.5)
        
        normalized['cloud'] = self._safe_normalize(weather_data['cloud'], 0, 100)
        
        # Wind impact varies based on temperature context
        wind_speed = max(weather_data['wind_kph'], weather_data['gust_kph'])
        normalized['wind'] = self._safe_normalize(wind_speed, 0, 80)
        
        # Wind impact context
        if weather_data['wind_impact'] == 'cooling':
            normalized['wind_comfort'] = 1 - normalized['wind']  # Wind is uncomfortable in cold
        elif weather_data['wind_impact'] == 'relief':
            normalized['wind_comfort'] = normalized['wind']  # Wind is pleasant in heat
        else:
            normalized['wind_comfort'] = 0.5  # Neutral impact
            
        normalized['humidity'] = self._safe_normalize(weather_data['humidity'], 20, 100)
        
        # Time of day with peak at solar noon (approximately 12-2PM)
        hour = weather_data['hour']
        solar_peak = 13  # 1 PM as approximate solar noon
        time_to_peak = min(abs(hour - solar_peak), abs(hour + 24 - solar_peak), abs(hour - 24 - solar_peak))
        normalized['daylight_peak'] = 1 - self._safe_normalize(time_to_peak, 0, 12)
        
        # Day vs Night factor (based on is_day)
        normalized['is_day'] = float(weather_data['is_day'])
        normalized['hour'] = weather_data['hour']


        return normalized

    def _safe_normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalizes a value to the range [0, 1], handling edge cases."""
        if max_val == min_val:
            return 0.5
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
    
    def _asymptotic_normalize(self, distance: float, target: float, scale: float) -> float:
        """
        Normalizes based on distance from target with asymptotic scaling.
        Returns 1.0 when distance = target, approaches 0 as distance increases.
        
        Args:
            distance: Value to normalize (distance from preferred)
            target: The ideal value (typically 0 for distance metrics)
            scale: Controls how quickly values approach 0
            
        Returns:
            Normalized value between 0 and 1
        """
        return np.exp(-((distance - target) ** 2) / (2 * scale ** 2))

    def _soft_bound(self, x: float, margin: float = 0.05) -> float:
        """Softly bounds a value to [0, 1]."""
        return np.where(x < 0, x / (x - margin), np.where(x > 1, 1 - (x - 1) / (x - 1 + margin), x))

    def _calculate_condition_sentiment(self, condition_text: str, condition_code: int) -> Tuple[float, float]:
        """
        Calculates valence and arousal from the weather condition text using NRC VAD lexicon.
        Also incorporates condition codes for more precise sentiment mapping.
        """
        # Code-based overrides for extreme weather conditions
        extreme_conditions = {
            # WeatherAPI condition codes with custom valence/arousal overrides
            # Thunderstorms
            1087: (0.3, 0.8),  # Thundery outbreaks
            1273: (0.2, 0.9),  # Patchy light rain with thunder
            1276: (0.1, 0.95), # Moderate or heavy rain with thunder
            
            # Snow and ice
            1066: (0.4, 0.6),  # Patchy snow
            1114: (0.3, 0.7),  # Blowing snow
            1117: (0.2, 0.8),  # Blizzard
            
            # Fog and low visibility
            1135: (0.4, 0.3),  # Fog
            1147: (0.3, 0.4),  # Freezing fog
            
            # Extreme rain
            1189: (0.3, 0.6),  # Moderate rain
            1192: (0.2, 0.7),  # Heavy rain
            1195: (0.1, 0.8),  # Heavy rain
            1246: (0.2, 0.7),  # Torrential rain
            
            # Pleasant conditions
            1000: (0.9, 0.4),  # Sunny/Clear
            1003: (0.8, 0.3),  # Partly cloudy
        }
        
        # Check if we have an override for this condition code
        if condition_code in extreme_conditions:
            return extreme_conditions[condition_code]

        # Otherwise use lexicon-based approach
        words = condition_text.split()
        valence_scores = []
        arousal_scores = []

        for i in range(len(words)):
            # Check for 2-grams
            if i < len(words) - 1:
                bigram = f"{words[i]} {words[i+1]}"
                if bigram in self._nrc_vad_lexicon:
                    valence_scores.append(self._nrc_vad_lexicon[bigram]['valence'])
                    arousal_scores.append(self._nrc_vad_lexicon[bigram]['arousal'])
                    continue  # Skip to the next iteration

            # Check for unigrams
            unigram = words[i]
            if unigram in self._nrc_vad_lexicon:
                valence_scores.append(self._nrc_vad_lexicon[unigram]['valence'])
                arousal_scores.append(self._nrc_vad_lexicon[unigram]['arousal'])

        # Add certain contextual words from weather that might not be in condition text
        weather_context_words = ["weather", "temperature", "sky"]
        for word in weather_context_words:
            if word in self._nrc_vad_lexicon:
                valence_scores.append(self._nrc_vad_lexicon[word]['valence'])
                arousal_scores.append(self._nrc_vad_lexicon[word]['arousal'])

        condition_valence = np.mean(valence_scores) if valence_scores else 0.5
        condition_arousal = np.mean(arousal_scores) if arousal_scores else 0.5

        return condition_valence, condition_arousal

    def _calculate_valence_arousal(self, api_response: Dict) -> Dict:
        """
        Calculates the overall valence and arousal from the weather data,
        incorporating research-based relationships between weather and mood.
        """
        weather_data = self._extract_weather_data(api_response)
        normalized = self._normalize_weather_data(weather_data)
        condition_valence, condition_arousal = self._calculate_condition_sentiment(
            weather_data['condition_text'], 
            weather_data['condition_code']
        )

        # --- Valence (Pleasure) Calculation ---
        # Research shows temperature, sunlight, and precipitation have strongest effects on mood
        # Sources: Klimstra et al. (2011), Keller et al. (2005)
        
        # Temperature effect (research shows inverted-U relationship)
        # Optimal temperature is personalized with seasonal expectations
        temp_valence = normalized['temp']
        
        # Day-night effect (research shows daylight affects mood positively)
        # Source: Keller et al. (2005), Parrott & Sabini (1990)
        daylight_factor = 0.6 * normalized['is_day'] + 0.4
        
        # Enhanced valence calculation based on research findings
        valence = (
            0.30 * temp_valence +                      # Temperature comfort (strongest effect)
            0.20 * (1 - normalized['rain']) +          # Rain has negative impact
            0.15 * (1 - normalized['cloud']) * daylight_factor + # Clouds matter more during day
            0.10 * normalized['wind_comfort'] +        # Wind context matters
            0.05 * (1 - normalized['humidity']) +      # High humidity slightly negative
            0.10 * normalized['daylight_peak'] * normalized['is_day'] # Time of day effect (day only)
        )

        # --- Arousal Calculation ---
        # Research shows arousal is affected by extremes and rapid changes
        # Source: Howarth & Hoffman (1984), Denissen et al. (2008)
        
        # Time-based arousal (circadian rhythm)
        # Source: Monk (2005) - alertness peaks in afternoon, dips in early morning
        time_arousal = 0.5 + 0.3 * math.cos(2 * math.pi * (normalized['hour'] - 15) / 24)
        
        # Calculate arousal from weather extremes
        # Extreme weather (high winds, storms, heavy rain) increases arousal
        weather_extremity = max(
            normalized['wind'],
            normalized['rain'],
            1 - normalized['temp'] if normalized['temp'] < 0.3 else 0,  # Very cold
            normalized['temp'] if normalized['temp'] > 0.7 else 0       # Very hot
        )
        
        arousal = (
            0.25 * time_arousal +                  # Time of day (circadian factors)
            0.30 * weather_extremity +             # Weather extremes increase arousal
            0.15 * normalized['wind'] +            # Wind increases arousal
            0.10 * normalized['uv'] * normalized['is_day'] + # UV relevant during day
            0.10 * normalized['rain'] +            # Rain increases arousal/alertness
            0.10 * (1 - normalized['cloud']) * normalized['is_day'] # Clear skies during day increase arousal
        )

        # Weight condition text more heavily for extreme conditions
        # Research shows language-based descriptors correlate strongly with emotional responses
        # Source: Baylis et al. (2018)
        if weather_data['condition_code'] in [1087, 1273, 1276, 1117, 1192, 1195, 1246]:
            # Extreme conditions - rely more on condition-based sentiment
            text_weight = 0.5
        else:
            # Normal conditions - balance formula and text
            text_weight = 0.3
            
        formula_weight = 1 - text_weight
            
        final_valence = self._soft_bound(formula_weight * valence + text_weight * condition_valence)
        final_arousal = self._soft_bound(formula_weight * arousal + text_weight * condition_arousal)

        # Add detailed breakdown for debugging and analysis
        return {
            'valence': round(float(final_valence), 2),
            'arousal': round(float(final_arousal), 2),
            'weather_data': weather_data,
            'details': {
                'formula_valence': round(float(valence), 2),
                'condition_valence': round(float(condition_valence), 2),
                'formula_arousal': round(float(arousal), 2),
                'condition_arousal': round(float(condition_arousal), 2),
                'text_weight': text_weight
            }
        }
