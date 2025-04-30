import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import Flask, redirect, url_for, session, request, jsonify
from flask_cors import CORS # Keep if needed for specific routes
from dotenv import load_dotenv
import logging
import re
from datetime import datetime, timedelta

# --- MODIFIED: Import db from extensions ---
from extensions import db

from REC import Main

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:8000/callback")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000") # Your React app's URL
DATABASE_URL = os.getenv('DATABASE_URL', "sqlite:///places.db") # Default to SQLite for local dev
if not DATABASE_URL:
    logging.warning("DATABASE_URL not set, falling back to default SQLite (not recommended for production)")
    DATABASE_URL = "sqlite:///places_fallback.db" # Example fallback

# --- Flask App Setup ---
app = Flask(__name__)
if not FLASK_SECRET_KEY:
    raise ValueError("FLASK_SECRET_KEY is required. Set it in your environment variables.")
app.secret_key = FLASK_SECRET_KEY

# Configure CORS globally
CORS(app, supports_credentials=True, origins=[FRONTEND_URL])
logging.basicConfig(level=logging.DEBUG) # Set logging level

# Session Configuration
app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = os.getenv('SESSION_COOKIE_SAMESITE', 'Lax')
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# --- SQLAlchemy Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = False # Set to True for debugging SQL

# --- MODIFIED: Initialize db with app ---
# This links the db object imported from extensions to our Flask app
db.init_app(app)

# --- REMOVED: Model definitions ---


from REC import Main
# --- Spotify OAuth Setup ---
SPOTIFY_SCOPE = "streaming user-read-email user-read-private playlist-modify-public playlist-modify-private user-read-playback-state user-modify-playback-state user-read-currently-playing"

# Configuration Checks
if not CLIENT_ID or not CLIENT_SECRET or not REDIRECT_URI:
     raise ValueError("Spotify CLIENT_ID, CLIENT_SECRET, and REDIRECT_URI must be set.")
logging.info(f"Flask app configured. Frontend URL: {FRONTEND_URL}, Backend Redirect URI: {REDIRECT_URI}")


# --- Helper Function for Token Validation/Refresh ---
def get_valid_token():
    """Gets token_info from session, refreshes if needed, returns access_token or None."""
    token_info = session.get("spotify_token_info")
    if not token_info:
        logging.warning("Token info not found in session.")
        return None

    # Create OAuth object inside function to ensure latest config is used
    sp_oauth = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SPOTIFY_SCOPE)

    try:
        valid_token_info = sp_oauth.validate_token(token_info)
        if not valid_token_info:
             logging.info("Token expired/invalid, attempting refresh.")
             if 'refresh_token' not in token_info:
                  logging.error("Refresh token missing.")
                  session.pop("spotify_token_info", None)
                  return None
             new_token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
             session["spotify_token_info"] = new_token_info # Update session
             logging.debug("Token refreshed successfully.")
             return new_token_info['access_token']
        else:
             session["spotify_token_info"] = valid_token_info # Store potentially updated info
             logging.debug("Token validated successfully.")
             return valid_token_info['access_token']
    except spotipy.SpotifyOauthError as e:
        logging.exception(f"Spotify OAuth error during token validation/refresh: {e}")
        session.pop("spotify_token_info", None)
        return None
    except Exception as e:
        logging.exception(f"Unexpected error during token validation/refresh: {e}")
        return None


# --- Routes ---

@app.route("/")
def index():
    """Basic index route."""
    return "Flask backend is running. Use /login to authenticate."

@app.route("/login")
def login():
    """Initiates the Spotify authorization flow."""
    sp_oauth = SpotifyOAuth(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI, scope=SPOTIFY_SCOPE,
        show_dialog=True # Force login prompt
    )
    auth_url = sp_oauth.get_authorize_url()
    logging.debug(f"Redirecting to Spotify auth URL: {auth_url}")
    return redirect(auth_url)

@app.route("/callback")
def callback():
    """Handles the Spotify redirect, exchanges code for token."""
    error = request.args.get('error')
    if error:
        logging.error(f"Error from Spotify callback: {error}")
        return redirect(f"{FRONTEND_URL}?error={error}") # Redirect to frontend with error

    code = request.args.get("code")
    if not code:
        logging.error("Authorization code missing in callback.")
        return redirect(f"{FRONTEND_URL}?error=auth_code_missing")

    logging.debug(f"Callback received code: {code[:10]}...") # Log truncated code
    try:
        sp_oauth = SpotifyOAuth(
            client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI, scope=SPOTIFY_SCOPE
        )
        token_info = sp_oauth.get_access_token(code, as_dict=True, check_cache=False)
        logging.debug("Token info successfully retrieved from Spotify.")

        session["spotify_token_info"] = token_info
        session.permanent = True
        logging.debug("Token info stored in session.")

        # Redirect back to frontend, indicating success via the code param
        return redirect(f"{FRONTEND_URL}?code={code}")

    except spotipy.SpotifyOauthError as e:
         logging.exception(f"OAuth Error during token exchange: {e}")
         return redirect(f"{FRONTEND_URL}?error=token_exchange_failed")
    except Exception as e:
        logging.exception(f"Unexpected error in callback: {e}")
        return redirect(f"{FRONTEND_URL}?error=callback_exception")

@app.route("/spotify_token", methods=['GET'])
def get_token_route():
    """Provides the access token to the frontend (called after callback)."""
    logging.debug("Frontend requesting /spotify_token")
    access_token = get_valid_token() # Use helper function
    if not access_token:
        error_msg = session.get("token_error", "Authentication required or token expired. Please log in again.")
        session.pop("token_error", None)
        return jsonify({"error": error_msg}), 401

    token_info = session.get("spotify_token_info", {})
    expires_at = token_info.get("expires_at")

    return jsonify({"access_token": access_token, "expires_at": expires_at})


@app.route("/playlist", methods=['POST'])
def create_playlist():
    """
    Creates a Spotify playlist based on location, vibe, and description.
    Passes vibe (selection) and description terms (situation_terms) to REC.py.
    """
    logging.debug("Entering /playlist route")
    access_token = get_valid_token()
    if not access_token:
        return jsonify({"error": "Authentication required or token expired. Please log in again."}), 401

    data = request.get_json()
    if not data:
        logging.error("Invalid request data received for /playlist.")
        return jsonify({"error": "Invalid request data"}), 400

    # Extract data
    place_id = data.get('place_id')
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    vibe_selection = data.get('dynamicSituation')
    user_description = data.get('userDescription', '')

    logging.debug(f"Received playlist request: PlaceID={place_id}, Lat={latitude}, Lng={longitude}, Vibe='{vibe_selection}', Desc='{user_description[:50]}...'")

    # Validation
    if not all([place_id, latitude is not None, longitude is not None]):
         logging.error("Missing required location data.")
         return jsonify({"error": "Missing required location data"}), 400
    if not vibe_selection:
         logging.error("Missing required vibe selection.")
         return jsonify({"error": "Missing required vibe selection"}), 400

    # Initialize Spotify Client
    try:
        sp = spotipy.Spotify(auth=access_token)
        user = sp.me()
        logging.debug(f"Authenticated Spotify user: {user.get('display_name', 'N/A')}")
    except Exception as e:
        logging.exception("Error initializing Spotify client or getting user.")
        return jsonify({"error": "Server error during Spotify setup."}), 500

    # Prepare Input Data for REC.py
    situation_terms = user_description.lower().split() if user_description else []
    situation_terms = [term for term in situation_terms if term]
    logging.info(f"Prepared input for REC.py: Vibe Selection='{vibe_selection}', Situation Terms={situation_terms}")

    # --- Call Recommendation Logic (REC.py) ---
    city = (latitude, longitude)
    city_name = f"City_{place_id}"

    try:
        # Ensure Main.rec accepts these arguments
        logging.info(f"Calling Main.rec for {city_name} with terms={situation_terms}, selection='{vibe_selection}'")
        sample_tracks = Main.rec(city, city_name, situation_terms=situation_terms, selection=vibe_selection)

        if not sample_tracks:
             logging.warning(f"Main.rec returned no tracks.")
             return jsonify({"message": f"Couldn't find matching tracks for this location/vibe.", "playlist_url": None}), 200

    except Exception as e:
         logging.exception(f"Error calling recommendation logic (Main.rec): {e}")
         return jsonify({"error": "Failed to generate recommendations."}), 500

    # --- Create and Populate Spotify Playlist ---
    try:
        playlist_name = f"Placeify: {place_id} - {vibe_selection}"
        playlist_description = f"Generated by Placeify for location ({latitude}, {longitude}). Vibe: {vibe_selection}."
        if user_description:
             playlist_description += f" Description: {user_description[:100]}..."

        logging.info(f"Creating playlist '{playlist_name}' for user {user.get('id')}")
        playlist = sp.user_playlist_create(user['id'], playlist_name, public=False, description=playlist_description)
        playlist_id = playlist['id']
        playlist_url = playlist.get('external_urls', {}).get('spotify')

        if not playlist_id:
             logging.error("Playlist created but ID missing.")
             return jsonify({"error": "Failed to get playlist ID after creation."}), 500

        logging.info(f"Playlist created ID: {playlist_id}. Adding tracks...")

        if sample_tracks:
            offset = 0; batch_size = 100
            while offset < len(sample_tracks):
                batch = sample_tracks[offset:offset + batch_size]
                try:
                    sp.playlist_add_items(playlist_id, batch)
                    logging.debug(f"Added batch of {len(batch)} tracks (offset {offset})")
                    offset += batch_size
                except spotipy.SpotifyException as batch_error:
                    logging.error(f"Error adding batch (offset {offset}): {batch_error.msg}", exc_info=True)
                    offset += batch_size # Ensure loop progresses
        else:
            logging.warning(f"No tracks to add to playlist {playlist_id}.")

        if not playlist_url:
             logging.warning(f"Playlist URL missing after creation.")
             return jsonify({"message": "Playlist created, but URL retrieval failed.", "playlist_id": playlist_id}), 200

        logging.info(f"Successfully created playlist: {playlist_url}")
        return jsonify({"playlist_url": playlist_url}), 200

    except spotipy.SpotifyException as e:
        logging.error(f"Spotify API error during playlist creation/population: {e.msg} (Status: {e.http_status})", exc_info=True)
        if e.http_status in [401, 403]:
             session.pop("spotify_token_info", None)
             return jsonify({"error": f"Spotify permission error ({e.http_status}). Please log in again."}), e.http_status
        return jsonify({"error": f"Spotify API error: {e.msg}"}), e.http_status
    except Exception as e:
        logging.exception("Unexpected error creating/populating playlist")
        return jsonify({"error": "An unexpected server error occurred during playlist finalization."}), 500


if __name__ == "__main__":
     port = int(os.environ.get("PORT", 8000))
     app.run(host='0.0.0.0', port=port, debug=False) # Debug MUST be False
