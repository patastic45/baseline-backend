

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime # Import if needed for default values in models

# Initialize SQLAlchemy extension object WITHOUT the app yet
# This object will be imported by the main app and other modules
db = SQLAlchemy()

# --- Define Database Models here ---
# Models now use the 'db' object imported from this file
class PlaceData(db.Model):
    __tablename__ = 'placeData' # Explicit table name matching your SQLite schema
    ID = db.Column(db.Text, primary_key=True) # Google Place ID
    Name = db.Column(db.Text, nullable=False) # Name of the place
    Valence = db.Column(db.Float) # Calculated valence score (0-1)
    Energy = db.Column(db.Float) # Calculated energy/arousal score (0-1)
    # Use DateTime for better timestamp handling, default to current UTC time
    LastUpdated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    reviews = db.relationship('Review', backref='place', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        # Helpful representation for debugging
        return f'<PlaceData {self.ID}: {self.Name}>'

class Review(db.Model):
    __tablename__ = 'reviews' # Explicit table name
    # Auto-incrementing integer primary key for reviews table
    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    # Foreign key linking back to the PlaceData table's primary key (ID)
    PlaceID = db.Column(db.Text, db.ForeignKey('placeData.ID'), nullable=False)
    # Text content of the review
    ReviewText = db.Column(db.Text)
    # Timestamp when the review was added to *our* database
    ReviewTime = db.Column(db.DateTime, default=datetime.utcnow)
    # Rating from Google (if available, otherwise might be null/default)
    Rating = db.Column(db.Integer)
    # Optional: Store calculated VAD for the individual review if needed later
    # Valence = db.Column(db.Float)
    # Energy = db.Column(db.Float)

    def __repr__(self):
        # Helpful representation for debugging
        return f'<Review {self.ID} for Place {self.PlaceID}>'

# You could initialize other extensions here as well, e.g.:
# from flask_caching import Cache
# cache = Cache()
