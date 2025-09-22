import pickle
from pathlib import Path

# Load the LDA direction
lda_dir = Path("./results/LDA")
lda_files = list(lda_dir.glob("lda_results_*.pkl"))
if lda_files:
    latest_file = max(lda_files, key=lambda f: f.stat().st_mtime)
    print(f"Latest file: {latest_file}")

    with open(latest_file, "rb") as f:
        data = pickle.load(f)

    print(f"Keys: {list(data.keys())}")
    if "lda_directions" in data:
        print(f'Layers: {sorted(data["lda_directions"].keys())}')
        if 22 in data["lda_directions"]:
            print(f'Layer 22 direction shape: {data["lda_directions"][22].shape}')
        else:
            print("Layer 22 direction not found")
    else:
        print("lda_directions key not found")
else:
    print("No LDA result files found")
