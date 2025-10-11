# Create the main project folder and navigate into it
mkdir Virus_Classifier
cd Virus_Classifier

# Create all sub-directories in one command
mkdir -p data/raw data/processed src models results/plots results/reports

# Create the empty files
touch README.md requirements.txt .gitignore src/01_download_data.py src/02_preprocess_data.py src/03_train_model.py src/04_evaluate_model.py src/utils.py