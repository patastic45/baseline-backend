import json
import os

class VisualizationGenerator:
    def __init__(self, nrc_vad_file="REC/NRC-VAD-Lexicon-v2.1.txt"):
        """
        Initialize the VisualizationGenerator with the NRC VAD database.
        """
        self._nrc_vad_db, self_ngrams = self._load_vad_lexicon(nrc_vad_file)

    def _load_vad_lexicon(self, file_path):
        """Load the VAD lexicon with enhanced error handling"""
        vad_dict = {}
        max_ngram = 1

        try:
            with open(file_path, 'r', encoding="utf-8") as file:
                for line in file:
                    parts = line.strip().split('\t')
                    if len(parts) != 4:
                        continue
                        
                    word, val, aro, dom = parts
                    word = word.lower()
                    
                    try:
                        vad_dict[word] = {
                            "valence": (float(val)+1)/2.0,
                            "arousal": (float(aro)+1)/2.0,
                            "dominance": (float(dom)+1)/2.0
                        }
                        
                        # Track max n-gram length
                        ngram = len(word.split())
                        if ngram > max_ngram:
                            max_ngram = ngram
                    except ValueError:
                        continue
                        
            return vad_dict, max_ngram
        except Exception as e:
            self.logger.error(f"Error loading lexicon from {file_path}: {e}")
            # Return empty dictionary and default n-gram length
            return {}, 1

    def generate_line_new(self, situation_terms, selection):
        """
        Generate visualization lines based on predefined options or user-selected terms.
        
        Args:
            situation_terms: A list of terms selected by the user to define their situation
            selection: Either one of the base selections ('bright', 'dark', 'tense', 'relaxed')
                         or a term from the NRC VAD database
            
        Returns:
            Dictionary with line visualization parameters
        """
        # Calculate the aggregate valence and energy for the situation terms
        situation_valence = 0.0
        situation_energy = 0.0
        valid_terms = 0

        for term in situation_terms:
            print(term)
            if term.lower() in self._nrc_vad_db:
                situation_valence += self._nrc_vad_db[term]["valence"]
                situation_energy += self._nrc_vad_db[term]["arousal"]  # Using arousal as energy
                valid_terms += 1
        
        # Handle case where no valid terms were provided
        if valid_terms == 0:
            raise ValueError("No valid situation terms found in NRC VAD database")
        
        # Calculate the average valence and energy
        situation_valence /= valid_terms
        situation_energy /= valid_terms
        
        # Handle the four base selections
        if selection == "bright":
            valence_range = (0.5, 1.0)
            energy = situation_energy
            return {"type": "horizontal", "x_range": valence_range, "y": energy}
        elif selection == "dark":
            valence_range = (0, 0.5)
            energy = situation_energy
            return {"type": "horizontal", "x_range": valence_range, "y": energy}
        elif selection == "tense":
            energy_range = (0.5, 1.0)
            valence = situation_valence
            return {"type": "vertical", "y_range": energy_range, "x": valence}
        elif selection == "relaxed":
            energy_range = (0, 0.5)
            valence = situation_valence
            return {"type": "vertical", "y_range": energy_range, "x": valence}
        
        # If not a base selection, check if it's in the NRC VAD database
        elif selection in self._nrc_vad_db:
            # Get the VAD values for the selected term
            term_vad = self._nrc_vad_db[selection]
            term_valence = term_vad["valence"]
            term_arousal = term_vad["arousal"]  # Assuming "arousal" is equivalent to "energy"
            
            # Determine line type based on which dimension has a more significant difference from neutral
            valence_diff = abs(term_valence - 0.5)
            arousal_diff = abs(term_arousal - 0.5)
            
            if valence_diff >= arousal_diff:
                # The term is more characterized by its valence (horizontal line)
                if term_valence > 0.5:
                    # Positive valence (bright-like)
                    valence_range = (0.5, 1.0)
                else:
                    # Negative valence (dark-like)
                    valence_range = (0, 0.5)
                
                return {"type": "horizontal", "x_range": valence_range, "y": situation_energy}
            else:
                # The term is more characterized by its arousal/energy (vertical line)
                if term_arousal > 0.5:
                    # High arousal (tense-like)
                    energy_range = (0.5, 1.0)
                else:
                    # Low arousal (relaxed-like)
                    energy_range = (0, 0.5)
                
                return {"type": "vertical", "y_range": energy_range, "x": situation_valence}
        else:
            raise ValueError(f"Selection '{selection}' is not a base option or found in NRC VAD database")

#Example usage
if __name__ == "__main__":
    generator = VisualizationGenerator()

    #Example Situations
    situation1 = ["shopping", "relaxing", "dinner"]
    situation2 = ["work", "meeting", "stress"]
    situation3 = ["game", "party", "laugh"]

    #Example Selections
    selection1 = "bright"
    selection2 = "tense"
    selection3 = "shopping"

    try:
        line1 = generator.generate_line_new(situation1, selection1)
        print(f"Line 1: {line1}")

        line2 = generator.generate_line_new(situation2, selection2)
        print(f"Line 2: {line2}")

        line3 = generator.generate_line_new(situation3, selection3)
        print(f"Line 3: {line3}")

    except ValueError as e:
        print(f"Error: {e}")