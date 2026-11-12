"""
Song Recognition System
ENGR 13300 Individual Project

Author: Simarleen Kaur
Collaborators: None
Date: Fall 2025

Academic Integrity Statement:
I certify that this work is entirely my own and complies with 
Purdue's academic integrity policies.

Description:
This program uses audio fingerprinting with Fourier transforms
to identify Kirtan songs from a user's recording. It matches
against a database of MP3 files and returns top 5 matches with
confidence percentages.

New Concept: Short-Time Fourier Transform (scipy.signal.spectrogram)
for frequency-domain audio analysis - NOT covered in ENGR 13300.

Required Libraries:
pip install numpy scipy librosa matplotlib
"""
import numpy as np
import librosa # music and audio analysis library
import pickle # used to convert python items into byte streams
import os 
from scipy import signal # Fourier Transform
from scipy.ndimage import maximum_filter # Finding peaks
from collections import defaultdict # Counts the matches in the audios
import warnings
warnings.filterwarnings('ignore')

# ===================================================================
# AUDIO PROCESSOR MODULE - Loading and preparing adio files
# ===================================================================

def load_audio(file_path, sample_rate = 22050):
    # Converts the audio file into numbers by reading it and ensures same sample rate.
    
    # ERROR CHECK - File exists?

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try: # error handling method
        # load file using librosa
        # sample rate - converts to standard speed
        # mono - usually a L and R but this converts into one (2 channels --> 1 channel)
        audio_data, sr = librosa.load(file_path, sr=sample_rate, mono=True)

        # Normalize values - all numbers should be between -1 to 1
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Message for when file is loaded and also the length of audio
        print(f"Audio Loaded: {os.path.basename(file_path)} ({len(audio_data)/sr:.1f}s)")
        return audio_data, sr
    
    except Exception as e: # if the audio file isn't able to load
        raise ValueError(f"Failed to load audio file: {str(e)}")
    

# ===================================================================
# FINGERPRINT GENERATOR MODULE - creating unique "fingerprints" from audio
# ===================================================================

def generate_spectrogram(audio_data, sample_rate, window_size=4096, overlap=0.5):
    # Audio is converted into picture showing frequencies over time.
    # X-axis is time, Y-axis is frequency (high/low notes), Brightness is how load frequency is at that time
    # Uses Fourier Transform

    hop_length = int(window_size * (1 - overlap)) # window_size is how many samples to look at once


    # Fourier Transform - breaks sound into frequency components and checks musical notes in each clip

    frequencies, times, spec = signal.spectogram(
        audio_data, 
        fs = sample_rate, # samples per second
        window='hann', # type of window (smooth edges)
        nperseg=window_size, # size of window
        noverlap=window_size - hop_length, # overlap in windows
        scaling='spectrum' # scaling for output
    )

    spectrogram = 10 * np.log10(spec + 1e-10)
    #converts to decibles, adds small numbers to avoid log(0) - debug by AI

    return frequencies, times, spectrogram

def find_peaks(spectrogram, freq_bins, time_bins, threshold_percentile=85):
    # Find loudests points in spectrogram - spectral peaks
    # threshold percentile - keeps peaks above 85%

    peaks = [] # Empty list to add peaks to

    threshold = np.percentile(spectrogram, threshold_percentile) # min intensity to be a peak

    neighborhood_size = 20 # checks neighbors 20 in each direction
    local_max = maximum_filter(spectrogram, size=neighborhood_size) # find local max

    # Iterate through each frequency row
    for freq_idx in range(spectrogram.sjape[0]):
        # Iterate through ach time column
        for time_idx in range(spectrogram.shape[1]):
            intensity = spectrogram[freq_idx, time_idx]

            if intensity == local_max[freq_idx, time_idx] and intensity > threshold:
                peaks.append((time_idx, freq_idx, intensity)) # add peak to list
            else:
                continue # don't add beause not a peak

    print(f"  Found {len(peaks)} spectral peaks")
    return peaks
        
def create_fingerprints(peaks, fan_value=5, target_zone_width=20):
    # choose a peak as the anchor, connect it to nearby peaks in future
    # fan_values - how many peaks connect with each anchor
    # given range when creating fingerprints

    peaks = sorted(peaks, key=lambda x: x[0]) #sorting peaks by time, song in chronological order

    fingerprints = defaultdict(list) #dictionary to store fingerprints

    # Iterate through each peak as anchor
    for i in range(len(peaks)):
        anchor_time, anchor_freq, _ = peaks[i]

        pairs_found = 0 # look for pair within fan_value
        j = i + 1 # next peak

        while j < len(peaks) and pairs_found < fan_value:
            target_time, target_freq, _ = peaks[j]

            time_delta = target_time - anchor_time # time difference between anchor and target

            if time_delta > target_zone_width:
                # target is too far into the future
                break
            elif time_delta > 0:
                # valid pair --> create a fingerprint hash
                # Hash --> difference between nite of anchor v target --> unique ID
                hash_value = (anchor_freq, target_freq, time_delta)

                # save the time when this pattern occurrs
                fingerprints[hash_value].append(anchor_time)

                pairs_found += 1
            j += 1 # moves to next possible target peak

    print(f"  Created {len(fingerprints)} unique fingerprints")

    return dict(fingerprints) #defaultdict --> regular dict

def generate_song_fingerprint(audio_data, sample_rate):
    # Main function: make spectrogram --> find peaks --> create fingerprints

    #calls the different functions to get information
    print("  Generating spectrogram...")
    frequencies, times, spectrogram = generate_spectrogram(audio_data, sample_rate)

    print("  Detecting peaks...")
    peaks = find_peaks(spectrogram, frequencies, times)

    print("  Creating fingerprints...")
    fingerprints = create_fingerprints(peaks)

    return fingerprints



# ===================================================================
# DATABASE MANAGER MODULE - Storing and loading song fingerprints
# ===================================================================

def build_database(songs_folder, database_file="kirtan_database.pkl"):
    # Able to process all songs in a folder and create a searchable database

    # ERROR CHECK - check if songs folder exists

    if not os.path.exists(songs_folder):
        raise FileNotFoundError(f"Songs folder not found: {songs_folder}")
    
    # empty dictionary to story song fingerprints
    database = {}

    # gets list of mp3 files in the folder
    mp3_files = [f for f in os.listdir(songs_folder) if f.endswith('.mp3')]

    # chekc if there are songs in the folder
    if len(mp3_files) == 0:
        raise ValueError(f"No MP3 files found in {songs_folder}")
    else:
        print(f"\nBuilding database from {len(mp3_files)} songs...")

    # iterate through each song file
    for idx, filename in enumerate(mp3_files, 1):
        file_path = os.path.join(songs_folder, filename) # folder + filename --> full path

        song_name = os.path.splitext(filename)[0] #song anem without .mp3 extension

        # Progress bar shows what song it is processing from list
        print(f"\n[{idx}/{len(mp3_files)}] Processing: {song_name}")

        try:
            # Load audio file
            audio_data, sr = load_audio(file_path)

            # Generate fingerprint
            fingerprints = generate_song_fingerprint(audio_data, sr)

            # Store in database
            database[song_name] = fingerprints

        except Exception as e:
            # just skip the song if song fails
            print(f"  Error Processing {filename}: {str(e)}")
            print(f"  Skipping this file...")
            continue # moves to next song

    save_database(database, database_file) #saves to file

    # print message once database completed
    print(f"\n✓ Database built successfully with {len(database)} songs")
    print(f"✓ Saved to: {database_file}")

    return database

def save_database(database, file_path):
    with open(file_path, 'wb') as f: # 'wb' = write binary
        pickle.dump(database, f)

def load_database(file_path):
    # load a saved database from before from file

    # ERROR CHECK - database file exists?
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Database file not found: {file_path}")
    
    # Load the pickled database
    with open(file_path, 'rb') as f: # rb = read binary
        database = pickle.load(f)

    print(f"✓ Loaded database with {len(database)} songs")
    return database


# ===================================================================
# MATCHER MODULE - Compares input audio with database
# ===================================================================

def match_song(input_fingerprints, database):
    # dictionary to count matchs for each song, automatically creates empty list for new keys
    song_matches = defaultdict(list)

    print("\nMatching fingerprints...")

    # Iterates through each fingerprint from input recording
    for input_hash, input_times in input_fingerprints.items():
        # Check each of the songs in the database
        for song_name, song_fingerprints, in database.items():
            
            # exact matching fingerprint?
            if input_hash in song_fingerprints:
                song_times = song_fingerprints[input_hash]

                # check all times the input occurrs
                for input_time in input_times:
                    # check all the song occurrences
                    for song_time in song_times:
                        # calculate the offset = song_time - input_time
                        time_offset = song_time - input_time

                        #save this match and the offset
                        song_matches[song_name].append((input_hash, time_offset))
        
        match_scores = [] #calculate score based on time alignment

        # Iterate through each song that has matches
        for song_name, matches_list in song_matches.itemns():
            if len(matches_list) > 0:
                # calculates how good of a matchj it is
                score = calculate_match_score(matches_list, len(input_fingerprints))
                match_scores.append((song_name, score))
            else:
                continue # if no matches, skip the song

        match_scores.sort(key=lambda x: x[1], reverse=True) #sort score by highest percent first
        return match_scores

def calculate_match_score(matches_list, total_input_hashes):
    # match quality based on time alignment histogram
    # histogram for each time offset --> find most common offset --> count matches at that offset --> calc percent
    offset_counts = defaultdict(int)

    for _, offset in matches_list:
        offset_counts[offset] += 1 # count for each offset

    # most comment offset, histogram peak
    if len(offset_counts) > 0:
        best_offset = max(offset_counts, key=offset_counts.get)
        aligned_matches = offset_counts[best_offset]
    else:
        aligned_matches = 0

    # calc percent match
    match_percentage = min(100.0, (aligned_matches/total_input_hashes) * 100)

    return match_percentage


# ===================================================================
# MAIN
# ===================================================================

def display_menu():
    # User menu
    print("\n" + "="*50)
    print("    KIRTAN SONG RECOGNITION SYSTEM")
    print("="*50)
    print("\n1. Build Database (Process MP3 folder)")
    print("2. Recognize Song (Match input audio)")
    print("3. Exit")
    print("-"*50)

def get_user_choice():
    # asking until it gets a valid input
    while True:
        # get user input
        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            return 1
        elif choice == '2':
            return 2
        elif choice == '3':
            return 3
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue

def get_folder_path():
    while True:
        folder_path = input("\nENter path to songs folder: ").strip()
        # clean up folder path input
        folder_path = folder_path.strip('"').strip("'")

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            return folder_path
        else:
            print(f"Folder not found: {folder_path}")
            print("Please enter a valid folder path.")

def get_audio_file_path():
    file_path = input("\nEnter path to audio file (MP3/WAV): ").strip()
    # clean file_path input
    file_path = file_path.strip('"').strip("'")
    
    
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return file_path 
    else:
        print(f"File not found: {file_path}")
        print("Please enter a valid file path.")

def display_results(matches, top_n=5):
    # display header for results
    print("\n" + "="*50)
    print("    TOP 5 MATCHES")
    print("="*50)
    
    if len(matches) == 0:
        # no matches found
        print("\nNo matches found.")
        print("  Try recordign a clearer sample or rebuilding database.")
    else:
        # show top 5 matches
        for i in range(min(top_n, len(matches))):
            song_name, percentage = matches[i]
            print(f"\n{i+1}. {song_name}")
            print(f"   Match: {percentage:.2f}%")

            #visual bar graph of matches
            bar_length = int(percentage / 2)
            print(f"   [{'█' * bar_length}{' ' * (50-bar_length)}]")

    print("\n" + "="*50)

def recognize_song_from_file(database):
    """
    1. user input for file
    2. load and process file
    3. generate fingerprint
    4. match with database
    5. show results
    """
    print("\n--- SONG RECOGNITION ---")

    # 1
    input_file = get_audio_file_path()

    try:
        # 2
        print("\nProcessing input audio...")
        audio_data, sr = load_audio(input_file)

        # 3
        input_fingerprints = generate_song_fingerprint(audio_data, sr)

        # 4
        matches = match_song(input_fingerprints, database)

        # 5
        display_results(matches)

    except Exception as e:
        # if something goes wrong in process
        print(f"\nError during recognition: {str(e)}")
        print("Please try again with a different file.")

def main():
    database = None # hold loaded database
    database_file = "kirtan_database.pkl"

    while True:
        display_menu() # MENU

        choice = get_user_choice() # USER CHOICE

        if choice == 1: # BUILD DATABASE
            print("\n--- DATABASE BUILDER ---")
            # folder path
            folder_path = get_folder_path()

            try:
                database = build_database(folder_path, database_file)
            except Exception as e:
                print(f"\nError building database: {str(e)}")
        elif choice == 2: # RECOGNIZE SONG
            if database is None: # checks if database is loaded
                try:
                    database = load_database(database_file)
                except FileNotFoundError:
                    print("\nNo database found. Please build database first (Option 1).")
                    continue
            recognize_song_from_file(database)

        else: # EXIT
            print("\nThank you for using Kirtan Recognition System!")
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()

