import numpy as np
from collections import defaultdict

class EucMusicRecommender:
    def __init__(self, music_df):
        self.music_df = music_df
        self.tracks = music_df.to_dict('records')
        self.valences = music_df['valence'].values
        self.energies = music_df['energy'].values
        
        # Build indexes for faster searching
        self._build_index()
    
    def _build_index(self):
        """Create binned indexes for valence and energy"""
        self.valence_bins = np.linspace(0, 1, 21)  # 0.05 increments
        self.energy_bins = np.linspace(0, 1, 21)
        
        self.valence_index = defaultdict(list)
        self.energy_index = defaultdict(list)
        
        for idx, (v, e) in enumerate(zip(self.valences, self.energies)):
            v_bin = np.digitize(v, self.valence_bins)
            e_bin = np.digitize(e, self.energy_bins)
            self.valence_index[v_bin].append(idx)
            self.energy_index[e_bin].append(idx)
    
    def distance_to_line(self, track_v, track_e, line_data):
        """EXACT implementation from paper (Method D)"""
        if line_data["type"] == "vertical":
            # Vertical line segment: x = constant, between y_min and y_max
            line_x = line_data["x"]
            y_min, y_max = line_data["y_range"]
            
            # Project point onto line segment
            projected_y = max(y_min, min(track_e, y_max))
            
            # Distance is hypotenuse if outside y-range, else just horizontal distance
            if y_min <= track_e <= y_max:
                return abs(track_v - line_x)
            else:
                return np.sqrt((track_v - line_x)**2 + (track_e - projected_y)**2)
        
        elif line_data["type"] == "horizontal":
            # Horizontal line segment: y = constant, between x_min and x_max
            line_y = line_data["y"]
            x_min, x_max = line_data["x_range"]
            
            # Project point onto line segment
            projected_x = max(x_min, min(track_v, x_max))
            
            # Distance is hypotenuse if outside x-range, else just vertical distance
            if x_min <= track_v <= x_max:
                return abs(track_e - line_y)
            else:
                return np.sqrt((track_e - line_y)**2 + (track_v - projected_x)**2)
    
    def recommend_in_range(self, spot_valence, spot_energy, line_data, n=10):
        """Returns ONLY tracks within the line's range, using full distance calculations"""
        # Get all tracks in range
        indices = [
            i for i in range(len(self.tracks)) 
            if self._is_in_range(self.valences[i], self.energies[i], line_data)
        ]
        
        # Calculate distances using the full distance method
        spot_dists = np.sqrt(
            (self.valences[indices] - spot_valence)**2 + 
            (self.energies[indices] - spot_energy)**2
        )
        line_dists = np.array([
            self.distance_to_line(self.valences[i], self.energies[i], line_data)
            for i in indices
        ])
        total_dists = spot_dists + line_dists
        
        # Get top n
        top_indices = np.argpartition(total_dists, min(n, len(total_dists)-1))[:n]
        return [{
            **self.tracks[indices[i]],
            'distance': total_dists[i],
            'spot_distance': spot_dists[i],
            'line_distance': line_dists[i]
        } for i in top_indices]
    
    def recommend_hybrid_three_targets(
    self, 
    spot1_valence, 
    spot1_energy, 
    spot2_valence, 
    spot2_energy, 
    line_data, 
    n=10,
    spot1_weight=0.4, 
    spot2_weight=0.3, 
    line_weight=0.3,
    epsilon=1e-6
    ):
        
        indices = [
            i for i in range(len(self.tracks)) 
            if self._is_in_range(self.valences[i], self.energies[i], line_data)
        ]

        # Calculate distances to spot1
        spot1_dists = np.sqrt(
            (self.valences[indices] - spot1_valence)**2 + 
            (self.energies[indices] - spot1_energy)**2
        )
        
        # Calculate distances to spot2
        spot2_dists = np.sqrt(
            (self.valences[indices] - spot2_valence)**2 + 
            (self.energies[indices] - spot2_energy)**2
        )
        
        # Calculate distances to the line
        line_dists = np.array([
            self.distance_to_line(self.valences[i], self.energies[i], line_data)
            for i in indices
        ])
        '''
        spot1_dists_norm = spot1_dists / np.max(spot1_dists)
        spot2_dists_norm = spot2_dists / np.max(spot2_dists)
        line_dists_norm = line_dists / np.max(line_dists)
        
        # Combine with weights
        total_scores = (
            spot1_weight * spot1_dists_norm +
            spot2_weight * spot2_dists_norm +
            line_weight * line_dists_norm
        )
        '''
        spot1_norm = (spot1_dists / spot1_dists.max()) * spot1_weight
        spot2_norm = (spot2_dists / spot2_dists.max()) * spot2_weight
        line_norm = (line_dists / line_dists.max()) * line_weight
        
        # Weighted Chebyshev distance (true Pareto weighting)
        total_scores = np.maximum.reduce([spot1_norm, spot2_norm, line_norm])
        
        # Secondary sorting by weighted sum for tie-breaking
        weighted_sums = spot1_norm + spot2_norm + line_norm
        top_indices = np.lexsort((weighted_sums, total_scores))[:n]
        
        return [{
            **self.tracks[indices[i]],
            'distance': total_scores[i],
            'spot1_distance': spot1_dists[i],
            'spot2_distance': spot2_dists[i],
            'line_distance': line_dists[i],
        } for i in top_indices]

    def recommend_any(self, spot_valence, spot_energy, line_data, n=10):
        """Returns recommendations from ALL tracks, penalizing out-of-range ones"""
        # Calculate distances for all tracks
        spot_dists = np.sqrt(
            (self.valences - spot_valence)**2 + 
            (self.energies - spot_energy)**2
        )
        line_dists = np.array([
            self.distance_to_line(self.valences[i], self.energies[i], line_data)
            for i in range(len(self.tracks))
        ])
        total_dists = spot_dists + line_dists
        
        # Get top n
        top_indices = np.argpartition(total_dists, min(n, len(total_dists)-1))[:n]
        return [{
            **self.tracks[i],
            'distance': total_dists[i],
            'spot_distance': spot_dists[i],
            'line_distance': line_dists[i],
            'in_range': self._is_in_range(self.valences[i], self.energies[i], line_data)
        } for i in top_indices]

    def _is_in_range(self, v, e, line_data):
        """Check if track falls within the line's range"""
        if line_data["type"] == "vertical":
            return line_data["y_range"][0] <= e <= line_data["y_range"][1]
        return line_data["x_range"][0] <= v <= line_data["x_range"][1]

    
    def _get_candidate_indices(self, target_v, target_e, margin=0.2):
        """Get potential candidates using binned indexes"""
        v_bin = np.digitize(target_v, self.valence_bins)
        e_bin = np.digitize(target_e, self.energy_bins)
        
        # Get nearby bins (±2 bins = ±0.1 in valence/energy)
        v_neighbors = range(max(1, v_bin-2), min(len(self.valence_bins)+1, v_bin+3))
        e_neighbors = range(max(1, e_bin-2), min(len(self.energy_bins)+1, e_bin+3))
        
        # Find intersection of candidates
        candidates = set()
        for v in v_neighbors:
            candidates.update(self.valence_index.get(v, []))
        
        temp_set = set()
        for e in e_neighbors:
            temp_set.update(self.energy_index.get(e, []))
        
        if temp_set:
            candidates.intersection_update(temp_set)
        
        return list(candidates) if candidates else list(range(len(self.tracks)))