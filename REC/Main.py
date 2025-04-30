from REC import googlePlaceAPI
from REC import situationalEmotion
from REC import EucRec
from REC import dynamicSituation
import pandas as pd

from REC import weatherAPI
from datetime import datetime
import nltk

import numpy as np

def rec(City, Name, situation_terms, selection, lstm=False):
    music_df = pd.read_csv('REC/cleaned_music_library.csv')
    api = googlePlaceAPI.APICall()
    emotion = dynamicSituation.VisualizationGenerator()
    recommender = EucRec.EucMusicRecommender(music_df)
    weather = weatherAPI.WeatherAPI()



    # Example usage (Times Square cafe)
    spot = api.get_location_mood(City[0], City[1], "cafe",250, "cafe", lstm)
    weatherEmo = weather.get_weather_data(City[0], City[1])
    print(f"Spot Atmosphere: Valence={spot['valence']:.2f}, Energy={spot['energy']:.2f}")
    print(f"Weather Atmosphere: Valence={weatherEmo['valence']:.2f}, Energy={weatherEmo['arousal']:.2f}")
    
    print(selection)
    print(situation_terms)
    line_data = emotion.generate_line_new(situation_terms, selection)
    print(f"Recommendation Line: {line_data}")

    weather_Rec = recommender.recommend_hybrid_three_targets(
        spot1_valence = spot['valence'], 
        spot1_energy = spot['energy'],
        spot2_valence = weatherEmo['valence'],
        spot2_energy = weatherEmo['arousal'],
        line_data = line_data,
        n=10, 
        spot1_weight=0.4, 
        spot2_weight=0.3, 
        line_weight=0.3
    )

    print("\nTop Weather Recommendations (Paper Method D):")
    track_IDs = []
    for i, track in enumerate(weather_Rec, 1):
        track_IDs.append("spotify:track:"+track['spotify_id'])
        print(f"{track['spotify_id']}")
    return track_IDs
