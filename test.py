import csv
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from xphonebr import Phonemizer



phone = Phonemizer(normalizer=True)

def phonemize_text(text):
    return phone.phonemise(text)

phonemized_transcriptions = True
# Manually set speaker IDs for Francisco and Antonio
speaker_ids = {
    'Francisco': 3001,
    'Antonio': 3002,
    'James': 3003,
    'Bia': 3004,
    'Paulo': 3005,
    "Lucas": 3006
}

# Read the CSV file
with open('./Data/metadata.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='|')
    lines = [row for row in reader if row['speaker_name'] in speaker_ids]

# Function to select samples based on user choice
def select_samples(lines, speaker_ids, samples_per_speaker="all"):
    speaker_samples = {speaker: [] for speaker in speaker_ids.values()}
    
    for row in lines:
        speaker = speaker_ids[row['speaker_name']]
        if samples_per_speaker == "all" or len(speaker_samples[speaker]) < samples_per_speaker:
            speaker_samples[speaker].append(row)

    filenames = []
    transcriptions = []
    speakers = []
    for speaker, samples in speaker_samples.items():
        for sample in samples:
            filenames.append(sample['audio_file'])
            transcriptions.append(sample['text'])
            speakers.append(speaker)

    return filenames, transcriptions, speakers

# User input for sample selection
sample_selection_mode = input("Select samples: all or specific number per speaker (enter 'all' or a number): ")

if sample_selection_mode == "all":
    filenames, transcriptions, speakers = select_samples(lines, speaker_ids, samples_per_speaker="all")
else:
    samples_per_speaker = int(sample_selection_mode)
    filenames, transcriptions, speakers = select_samples(lines, speaker_ids, samples_per_speaker)



if phonemized_transcriptions:
    # Phonemize the transcriptions using multiprocessing
    with Pool(cpu_count()) as pool:
        phonemized = list(tqdm(pool.imap(phonemize_text, transcriptions), total=len(transcriptions), desc="Phonemizing", unit="text"))
else:
    phonemized = transcriptions

phonemized_lines = []
# Iterate over the lists in parallel using zip
for filename, phonemized_text, speaker in zip(filenames, phonemized, speakers):
    phonemized_lines.append((filename, f'{filename}|{phonemized_text}|{speaker}\n'))

# Sort based on filename (if needed, adjust sorting as required)
phonemized_lines.sort(key=lambda x: x[0])

# Split into training and validation sets
split_index = int(len(phonemized_lines) * 0.9)
train_lines = phonemized_lines[:split_index]
val_lines = phonemized_lines[split_index:]

if phonemized_transcriptions:
    # Save to files without headers
    with open('./Data/train_list.txt', 'w+', newline='') as f:
        for _, line in train_lines:
            f.write(line)

    with open('./Data/val_list.txt', 'w+', newline='') as f:
        for _, line in val_lines:
            f.write(line)
else:
    # Save to files without headers
    with open('./Data/train_no_phones.txt', 'w+', newline='') as f:
        for _, line in train_lines:
            f.write(line)

    with open('./Data/val_no_phones.txt', 'w+', newline='') as f:
        for _, line in val_lines:
            f.write(line)
