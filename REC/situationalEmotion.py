import pandas as pd

class SituationalEmotion:

    def generate_line(self, location_type, situation, selection_type, selection):
        energy_table = {
            ("cafe", "Change of Pace"): {"bright": 0.52, "dark": 0.55},
            ("cafe", "Killing Time"): {"bright": 0.51, "dark": 0.4},
            ("cafe", "Work"): {"bright": 0.35, "dark": 0.3},
            ("train", "Change of Pace"): {"bright": 0.58, "dark": 0.49},
            ("train", "Killing Time"): {"bright": 0.6, "dark": 0.35},
            ("train", "Work"): {"bright": 0.4, "dark": 0.22},
            ("park", "Rest"): {"bright": 0.3, "dark": 0.2},
            ("park", "Picnic"): {"bright": 0.55, "dark": 0.38},
            ("park", "Walking"): {"bright": 0.52, "dark": 0.4},
            ("department_store", "Shopping"): {"bright": 0.6, "dark": 0.42},
        }

        valence_table = {
            ("cafe", "Change of Pace"): {"high": 0.7, "low": 0.5},
            ("cafe", "Killing Time"): {"high": 0.75, "low": 0.55},
            ("cafe", "Work"): {"high": 0.6, "low": 0.4},
            ("park", "Rest"): {"high": 0.8, "low": 0.6},
            ("park", "Picnic"): {"high": 0.7, "low": 0.6},
            ("park", "Walking"): {"high": 0.6, "low": 0.4},
            ("department_store", "Shopping"): {"high": 0.75, "low": 0.55},
        }

        if (location_type, situation) not in energy_table:
            raise ValueError(f"Invalid combination: {location_type} + {situation}")
        
        if selection_type == "mood":
            energy = energy_table[(location_type, situation)][selection]
            valence_range = (0.5, 1.0) if selection == "bright" else (0.0, 0.5)
            return {"type": "horizontal", "x_range": valence_range, "y": energy}
        
        elif selection_type == "energy":
            valence = valence_table[(location_type, situation)][selection]
            energy_range = (0.5, 1.0) if selection == "high" else (0.0, 0.5)
            return {"type": "vertical", "y_range": energy_range, "x": valence}
        
        else:
            raise ValueError("selection_type must be 'mood' or 'energy'.")
    