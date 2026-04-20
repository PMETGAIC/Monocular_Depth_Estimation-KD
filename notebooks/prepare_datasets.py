import os
from datasets import load_dataset

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_dati = os.path.abspath(os.path.join(base_dir, "../data"))

    dataset = load_dataset(
        "sayakpaul/nyu_depth_v2", 
        trust_remote_code=True, 
        cache_dir=path_dati
    )
    
if __name__ == "__main__":
    main()