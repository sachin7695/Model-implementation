import os 
from sklearn.model_selection import train_test_split 
import pandas as pd 
import librosa 

def random_split_prep_ljspeech_metadata(path_to_data, 
                                        path_to_save="data/",
                                        seed=42,
                                        test_split_percent = 0.01, 
                                        sort = True):
    path_to_metadata = os.path.join(path_to_data, "metadata.csv")
    metadata = pd.read_csv(path_to_metadata, sep = "|", header = None)
    metadata.columns = ["file_path", "raw_transcript", "normalized_transcript"] 

    metadata = metadata[["file_path", "normalized_transcript"]]
    metadata = metadata.dropna(subset=['normalized_transcript']).reset_index(drop=True)


    full_path_func = lambda x: os.path.join(path_to_data, "wavs", f"{x}.wav")
    metadata["file_path"] = metadata["file_path"].apply(full_path_func)

    verify_func = lambda x:os.path.isfile(x)
    exists = metadata["file_path"].apply(verify_func)
    assert all(list(exists)), "Check path_to_data or files, something is missing"


    ### Get Durations ###
    duration_func = lambda x: librosa.get_duration(path=x)
    metadata["duration"] = metadata["file_path"].apply(duration_func)

    ### Random Split ###
    train_df, test_df = train_test_split(metadata, test_size=test_split_percent, random_state=seed)

    ### Sort From Longest to Shortest (so Padding is minimized) ###
    if sort:
        train_df = train_df.sort_values(by=["duration"], ascending=False)

    train_df.to_csv(os.path.join(path_to_save, "train_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(path_to_save, "test_metadata.csv"), index=False)

    print("Done") 


if __name__ == "__main__":

    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--path_to_ljspech", type=str, required=True)
    # parser.add_argument("--path_to_save", type=str, required=True)
    # parser.add_argument("--seed", type=int, default=42)

    # args = parser.parse_args()

    
    path_to_ljspeech = "/home/cmi_10101/Documents/coding/pytorch/" \
    "architecture-implementation/" \
    "tts/LJSpeech-1.1"

    path_to_save = "/home/cmi_10101/Documents/coding/pytorch/" \
    "architecture-implementation/" \
    "tts/LJSpeech-1.1"
    random_split_prep_ljspeech_metadata(path_to_ljspeech, 
                                            path_to_save,
                                            seed=42)