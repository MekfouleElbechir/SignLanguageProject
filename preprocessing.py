"""
05_Code_Source — preprocessing.py
Projet ARSL — Preparation des donnees pour l'approche multimodale (Late Fusion)

CE QUE FAIT CE SCRIPT :
1. Charge les sequences .npy organisees en train/val/test/<label>/*.npy
   (structure produite par inventory_npy_and_split.py)
2. Pour chaque sequence, SEPARE le vecteur de 1662 features en 4 modalites :
   - pose        (132 features : 33 points x 4)
   - main_gauche (63 features  : 21 points x 3)
   - main_droite (63 features  : 21 points x 3)
   - visage      (1404 features: 468 points x 3)
3. Uniformise la longueur temporelle (padding/troncature) a une longueur fixe
   (par defaut 150 frames, proche de la moyenne reelle du corpus = 148.4)
4. Encode les labels (texte -> entier) et sauvegarde un mapping label2id
5. Sauvegarde tout dans des fichiers .npz prets a l'emploi pour l'entrainement
   (un seul fichier par split : train.npz, val.npz, test.npz)

INSTALLATION :
    pip install numpy --break-system-packages

UTILISATION :
    python preprocessing.py --data_dir /chemin/vers/02_Donnees_organisee --output_dir /chemin/vers/02_Donnees_organisee/prepared --max_len 150

Le dossier data_dir doit contenir train/, val/, test/, chacun avec des sous-dossiers <label>/*.npy
(c'est exactement la structure produite par inventory_npy_and_split.py)
"""

import argparse
import json
from pathlib import Path

import numpy as np

# --- Definition des tranches (slices) de chaque modalite dans le vecteur de 1662 features ---
# Ordre MediaPipe Holistic : pose (33x4) + main_gauche (21x3) + main_droite (21x3) + visage (468x3)
MODALITY_SLICES = {
    'pose':        slice(0, 132),      # 33 * 4
    'main_gauche': slice(132, 195),    # 21 * 3
    'main_droite': slice(195, 258),    # 21 * 3
    'visage':      slice(258, 1662),   # 468 * 3
}


def split_modalities(sequence):
    """
    Prend une sequence (n_frames, 1662) et retourne un dict avec les 4 modalites separees.
    """
    return {name: sequence[:, sl] for name, sl in MODALITY_SLICES.items()}


def pad_or_truncate(sequence, max_len):
    """
    Ramene une sequence (n_frames, n_features) a exactement max_len frames.
    - Si trop courte : padding avec des zeros a la fin
    - Si trop longue  : troncature (on garde les max_len premieres frames)
    Retourne aussi un mask (1 = frame reelle, 0 = padding) utile pour le Masking layer de Keras.
    """
    n_frames, n_features = sequence.shape
    if n_frames >= max_len:
        return sequence[:max_len], np.ones(max_len, dtype=np.float32)
    padded = np.zeros((max_len, n_features), dtype=sequence.dtype)
    padded[:n_frames] = sequence
    mask = np.zeros(max_len, dtype=np.float32)
    mask[:n_frames] = 1.0
    return padded, mask


def load_split(split_dir, max_len):
    """
    Parcourt un dossier <split>/<label>/*.npy et construit les arrays multimodaux.
    Retourne un dict : {modality_name: array(N, max_len, n_features_modality)}, labels (N,), mask (N, max_len)
    """
    split_dir = Path(split_dir)
    labels = []
    per_modality_sequences = {name: [] for name in MODALITY_SLICES}
    masks = []

    label_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    for label_dir in label_dirs:
        label = label_dir.name
        for npy_file in sorted(label_dir.glob('*.npy')):
            seq = np.load(npy_file, allow_pickle=True)
            if seq.ndim != 2 or seq.shape[1] != 1662:
                print(f"ATTENTION : {npy_file.name} ignore (forme inattendue : {seq.shape})")
                continue

            modalities = split_modalities(seq)
            padded_modalities = {}
            frame_mask = None
            for name, arr in modalities.items():
                padded_arr, mask = pad_or_truncate(arr, max_len)
                padded_modalities[name] = padded_arr
                frame_mask = mask  # le mask est identique pour toutes les modalites (meme longueur temporelle)

            for name in MODALITY_SLICES:
                per_modality_sequences[name].append(padded_modalities[name])
            masks.append(frame_mask)
            labels.append(label)

    data = {name: np.stack(seqs) for name, seqs in per_modality_sequences.items()}
    data['mask'] = np.stack(masks)
    data['labels'] = np.array(labels)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Dossier contenant train/, val/, test/')
    parser.add_argument('--output_dir', required=True, help='Dossier de sortie pour les .npz prepares')
    parser.add_argument('--max_len', type=int, default=150, help='Longueur temporelle fixe (frames)')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # --- Construction du mapping label -> id a partir du train uniquement (coherence garantie) ---
    train_dir = Path(args.data_dir) / 'train'
    all_labels = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    label2id = {label: i for i, label in enumerate(all_labels)}
    with open(Path(args.output_dir) / 'label2id.json', 'w', encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    print(f"Classes ({len(label2id)}) : {label2id}")

    for split_name in ['train', 'val', 'test']:
        print(f"\nTraitement du split '{split_name}'...")
        split_dir = Path(args.data_dir) / split_name
        data = load_split(split_dir, args.max_len)

        n_samples = len(data['labels'])
        print(f"  {n_samples} sequences chargees")
        for name in MODALITY_SLICES:
            print(f"  {name} : shape {data[name].shape}")

        label_ids = np.array([label2id[l] for l in data['labels']])

        out_path = Path(args.output_dir) / f'{split_name}.npz'
        np.savez_compressed(
            out_path,
            pose=data['pose'],
            main_gauche=data['main_gauche'],
            main_droite=data['main_droite'],
            visage=data['visage'],
            mask=data['mask'],
            labels=label_ids,
        )
        print(f"  Sauvegarde : {out_path}")

    print("\nTERMINE ! Fichiers prets pour l'entrainement dans :", args.output_dir)
    print("Utilise model.py pour charger ces .npz et entrainer le modele multimodal.")


if __name__ == '__main__':
    main()
