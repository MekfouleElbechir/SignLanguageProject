"""
02_Donnees — Inventaire des landmarks deja extraits (.npy) + creation du manifeste + split
Projet ARSL

CONTEXTE :
Tu as deja des fichiers .npy (landmarks extraits) tous dans UN SEUL dossier, sans manifest.csv
ni labels.json. Ce script :
1. Parcourt tous les .npy du dossier
2. Extrait le label depuis le nom du fichier (meme logique que pour les videos :
   signe_<nom_du_signe>_<numero>.npy -> label = "nom_du_signe")
3. Genere manifest.csv et labels.json
4. Copie/organise les fichiers dans train/<label>/, val/<label>/, test/<label>/
   (garde au moins 1 fichier par classe en val et en test quand possible)
5. Affiche toutes les statistiques a copier dans Data_Dictionary.md

INSTALLATION :
    pip install numpy pandas --break-system-packages

UTILISATION :
    python inventory_npy_and_split.py --input_dir /chemin/vers/tes/npy --output_dir /chemin/vers/02_Donnees/data_organisee

Si le format de nom de fichier est different, modifie uniquement parse_label().
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_label(filename):
    """
    Extrait le label depuis un nom de fichier du type : HowAreYou_12_01.npy -> "HowAreYou"

    Regle : le label est forme de tous les segments (separes par "_") AVANT le premier
    segment purement numerique. Tout ce qui vient apres (ex: "12", "01") est considere
    comme des identifiants (personne, prise, etc.), pas comme faisant partie du label.

    Exemples :
        HowAreYou_12_01.npy   -> "HowAreYou"
        signe_bonjour_01.npy  -> "signe_bonjour"   (attention : ici "signe" fait partie du
                                                      label car aucun prefixe n'est retire
                                                      automatiquement — adapte si besoin)
        merci_03.npy           -> "merci"

    ADAPTE CETTE FONCTION si ton format est different.
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) < 2:
        return stem if stem else None

    first_numeric_idx = None
    for i, p in enumerate(parts):
        if p.isdigit():
            first_numeric_idx = i
            break

    if first_numeric_idx is None:
        # aucun segment numerique trouve : on prend tout sauf le dernier segment par defaut
        label_parts = parts[:-1]
    elif first_numeric_idx == 0:
        # le tout premier segment est deja numerique -> format inattendu
        return None
    else:
        label_parts = parts[:first_numeric_idx]

    label = '_'.join(label_parts)
    return label if label else None


def build_manifest(input_dir):
    input_dir = Path(input_dir)
    rows = []
    skipped = []
    for f in sorted(input_dir.glob('*.npy')):
        label = parse_label(f.name)
        if label is None:
            skipped.append(f.name)
            continue
        # on ouvre le fichier pour verifier qu'il est lisible et recuperer sa forme (shape)
        try:
            arr = np.load(f, allow_pickle=True)
            shape = arr.shape
        except Exception as e:
            print(f"ERREUR de lecture pour {f.name} : {e}")
            skipped.append(f.name)
            continue
        rows.append({
            'filepath': str(f),
            'filename': f.name,
            'label': label,
            'n_frames': shape[0] if len(shape) > 0 else None,
            'n_features': shape[1] if len(shape) > 1 else None,
        })

    if skipped:
        print(f"\nATTENTION : {len(skipped)} fichiers ignores (format non reconnu ou illisible) :")
        for name in skipped[:15]:
            print(f"   - {name}")
        if len(skipped) > 15:
            print(f"   ... et {len(skipped) - 15} autres")

    df = pd.DataFrame(rows)
    return df


def print_statistics(df):
    print("\n" + "=" * 60)
    print("STATISTIQUES DU CORPUS (a copier dans Data_Dictionary.md)")
    print("=" * 60)
    print(f"Nombre total de fichiers .npy valides : {len(df)}")
    print(f"Nombre de classes (signes distincts)  : {df['label'].nunique()}")
    counts = df['label'].value_counts()
    print(f"Nombre moyen de fichiers par classe    : {counts.mean():.1f}")
    print(f"Nombre minimum de fichiers pour une classe : {counts.min()} (classe : {counts.idxmin()})")
    print(f"Nombre maximum de fichiers pour une classe : {counts.max()} (classe : {counts.idxmax()})")
    if df['n_frames'].notna().any():
        print(f"Nombre moyen de frames par fichier      : {df['n_frames'].mean():.1f}")
    if df['n_features'].notna().any():
        print(f"Nombre de features par frame (dimension) : {df['n_features'].mode().iloc[0]}")
    print("\nDistribution complete par classe :")
    print(counts.to_string())
    print("=" * 60)

    # --- Validation de la coherence des formes (shapes) ---
    print("\n" + "=" * 60)
    print("VALIDATION — COHERENCE DES FORMES (n_frames, n_features)")
    print("=" * 60)
    shape_counts = df.groupby(['n_frames', 'n_features']).size().reset_index(name='count')
    shape_counts = shape_counts.sort_values('count', ascending=False)
    n_unique_shapes = len(shape_counts)

    if n_unique_shapes == 1:
        r = shape_counts.iloc[0]
        print(f"OK — tous les fichiers ont la meme forme : "
              f"{int(r['n_frames'])} frames x {int(r['n_features'])} features")
    else:
        print(f"ATTENTION — {n_unique_shapes} formes differentes detectees parmi les fichiers :")
        print(shape_counts.to_string(index=False))
        print(
            "\nCe n'est pas forcement grave (un nombre de frames different par video est normal\n"
            "si tu prevois un padding/troncature dans le preprocessing), MAIS le nombre de\n"
            "FEATURES (2eme colonne) doit etre identique partout — sinon les fichiers ne sont\n"
            "pas compatibles entre eux et il faut re-extraire les landmarks des fichiers en trop\n"
            "ou en moins avec le meme pipeline MediaPipe."
        )
        n_feature_values = df['n_features'].nunique()
        if n_feature_values > 1:
            print(f"\n>>> A CORRIGER EN PRIORITE : {n_feature_values} dimensions de features "
                  f"differentes trouvees : {sorted(df['n_features'].dropna().unique().tolist())}")
    print("=" * 60)


def split_dataset(df, test_per_class=5, val_per_class=5, seed=42):
    rng = np.random.RandomState(seed)
    train_idx, val_idx_all, test_idx_all = [], [], []
    for label, group in df.groupby('label'):
        idx = group.index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_test = min(test_per_class, max(0, n - 2))
        n_val = min(val_per_class, max(0, n - n_test - 1))
        test_idx_all += idx[:n_test]
        val_idx_all += idx[n_test:n_test + n_val]
        train_idx += idx[n_test + n_val:]
    return df.loc[train_idx], df.loc[val_idx_all], df.loc[test_idx_all]


def copy_split(df_split, split_name, output_dir):
    out_root = Path(output_dir) / split_name
    for _, row in df_split.iterrows():
        label_dir = out_root / row['label']
        label_dir.mkdir(parents=True, exist_ok=True)
        dest = label_dir / row['filename']
        shutil.copy2(row['filepath'], dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Dossier contenant tous les .npy (non tries)')
    parser.add_argument('--output_dir', required=True, help='Dossier de sortie organise (train/val/test)')
    args = parser.parse_args()

    print("ETAPE 1 : Inventaire des fichiers .npy...")
    df = build_manifest(args.input_dir)
    if len(df) == 0:
        print("Aucun fichier .npy valide trouve. Verifie le format des noms de fichiers.")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.output_dir) / 'manifest.csv'
    df.to_csv(manifest_path, index=False)
    print(f"Manifeste sauvegarde : {manifest_path}")

    labels = sorted(df['label'].unique())
    labels_path = Path(args.output_dir) / 'labels.json'
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"Labels sauvegardes : {labels_path}")

    print_statistics(df)

    print("\nETAPE 2 : Decoupage train / val / test...")
    train_df, val_df, test_df = split_dataset(df)
    print(f"Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")

    print("\nETAPE 3 : Copie des fichiers dans la structure organisee...")
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        copy_split(split_df, split_name, args.output_dir)
        print(f"  {split_name} : {len(split_df)} fichiers copies")

    print("\nTERMINE ! Structure finale :")
    print(f"  {args.output_dir}/manifest.csv")
    print(f"  {args.output_dir}/labels.json")
    print(f"  {args.output_dir}/train/<label>/*.npy")
    print(f"  {args.output_dir}/val/<label>/*.npy")
    print(f"  {args.output_dir}/test/<label>/*.npy")
    print("\n>>> Copie les statistiques ci-dessus dans Data_Dictionary.md et Protocole_Splits.md <<<")


if __name__ == '__main__':
    main()