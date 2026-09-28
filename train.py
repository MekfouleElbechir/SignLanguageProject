"""
05_Code_Source — train.py
Projet ARSL — Entrainement du modele multimodal (BiLSTM+Attention+Late Fusion)

CE QUE FAIT CE SCRIPT :
1. Charge train.npz / val.npz / test.npz (produits par preprocessing.py)
2. Construit le modele multimodal (model.py) -- ou le modele Early Fusion si --fusion=early
3. Entraine avec des callbacks anti-overfitting :
   - EarlyStopping (arrete des que val_loss ne s'ameliore plus, restaure les meilleurs poids)
   - ModelCheckpoint (sauvegarde uniquement le meilleur modele)
   - ReduceLROnPlateau (diminue le taux d'apprentissage si stagnation)
4. Affiche et sauvegarde les courbes d'entrainement (train vs val) -- essentiel pour
   diagnostiquer visuellement l'overfitting (ecart entre les deux courbes)
5. Evalue sur le test set (utilise UNE SEULE FOIS, a la fin)
6. Sauvegarde le modele final dans 06_Modeles_IA

INSTALLATION :
    pip install tensorflow numpy matplotlib scikit-learn --break-system-packages

UTILISATION :
    python train.py --data_dir /chemin/vers/prepared --output_dir /chemin/vers/06_Modeles_IA --fusion late --epochs 100

    --fusion late  : modele principal (Late Fusion, architecture cible du memoire)
    --fusion early : modele de comparaison (Early Fusion, pour l'experience ablative H3)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from model import (build_multimodal_model, build_early_fusion_model, build_hands_only_model,
                    build_multimodal_model_no_attention, AttentionPooling)


def load_npz_split(path):
    data = np.load(path)
    inputs = {
        'pose': data['pose'],
        'main_gauche': data['main_gauche'],
        'main_droite': data['main_droite'],
        'visage': data['visage'],
    }
    labels = data['labels']
    return inputs, labels


def plot_training_curves(history, output_path):
    """
    Trace loss et accuracy (train vs val) sur le meme graphique.
    Un ecart qui se creuse entre train et val = signe d'overfitting a surveiller.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss (train vs validation)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Accuracy (train vs validation)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Courbes d'entrainement sauvegardees : {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Dossier contenant train.npz, val.npz, test.npz')
    parser.add_argument('--output_dir', required=True, help='Dossier de sortie pour le modele et les resultats')
    parser.add_argument('--fusion', choices=['late', 'early', 'mains_only', 'no_attention'], default='late')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=15, help='Patience pour EarlyStopping')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Chargement des donnees...")
    X_train, y_train = load_npz_split(data_dir / 'train.npz')
    X_val, y_val = load_npz_split(data_dir / 'val.npz')
    X_test, y_test = load_npz_split(data_dir / 'test.npz')
    max_len = X_train['pose'].shape[1]
    num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)

    print(f"Train : {len(y_train)} | Val : {len(y_val)} | Test : {len(y_test)}")
    print(f"max_len = {max_len} | num_classes = {num_classes}")

    with open(data_dir / 'label2id.json', 'r', encoding='utf-8') as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    print(f"\nConstruction du modele (fusion={args.fusion})...")
    if args.fusion == 'late':
        model = build_multimodal_model(max_len=max_len, num_classes=num_classes)
    elif args.fusion == 'early':
        model = build_early_fusion_model(max_len=max_len, num_classes=num_classes)
    elif args.fusion == 'mains_only':
        model = build_hands_only_model(max_len=max_len, num_classes=num_classes)
    else:
        model = build_multimodal_model_no_attention(max_len=max_len, num_classes=num_classes)
    model.summary()
    print(f"Nombre total de parametres : {model.count_params():,}")

    # Ne garder que les entrees reellement utilisees par le modele (ex: mains_only
    # n'utilise pas 'pose' ni 'visage' -- les filtrer evite une erreur de shape mismatch)
    required_inputs = set(inp.name for inp in model.inputs)
    X_train = {k: v for k, v in X_train.items() if k in required_inputs}
    X_val = {k: v for k, v in X_val.items() if k in required_inputs}
    X_test = {k: v for k, v in X_test.items() if k in required_inputs}
    print(f"Entrees utilisees par ce modele : {sorted(required_inputs)}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model_path = output_dir / f'model_{args.fusion}_fusion_best.keras'
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(model_path), monitor='val_loss', save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1
        ),
    ]

    print("\nDebut de l'entrainement...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    plot_training_curves(history, output_dir / f'training_curves_{args.fusion}_fusion.png')

    # --- Diagnostic overfitting automatique ---
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    gap = final_train_acc - final_val_acc
    print(f"\n{'='*60}")
    print("DIAGNOSTIC OVERFITTING")
    print(f"{'='*60}")
    print(f"Accuracy train (dernier epoch avant restore_best_weights) : {final_train_acc:.3f}")
    print(f"Accuracy val   (dernier epoch avant restore_best_weights) : {final_val_acc:.3f}")
    print(f"Ecart train - val : {gap:.3f}")
    if gap > 0.15:
        print("ATTENTION : ecart important (> 0.15) -- overfitting probable.")
        print("Pistes : augmenter dropout_rate/l2_reg dans model.py, reduire lstm_units,")
        print("ou envisager de la data augmentation sur les sequences.")
    else:
        print("Ecart raisonnable -- pas de signe fort d'overfitting.")
    print(f"{'='*60}")

    # --- Evaluation finale sur le TEST SET (une seule fois) ---
    print("\nEvaluation sur le test set (utilise une seule fois)...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy : {test_acc:.3f} | Test loss : {test_loss:.3f}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    target_names = [id2label[i] for i in range(num_classes)]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=3)
    print("\nRapport de classification (test set) :")
    print(report)

    with open(output_dir / f'classification_report_{args.fusion}_fusion.txt', 'w', encoding='utf-8') as f:
        f.write(f"Test accuracy : {test_acc:.3f}\nTest loss : {test_loss:.3f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(target_names, rotation=45, ha='right')
    ax.set_yticklabels(target_names)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Verite terrain')
    ax.set_title(f'Matrice de confusion — {args.fusion} fusion')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{args.fusion}_fusion.png', dpi=150)
    plt.close()

    model.save(output_dir / f'model_{args.fusion}_fusion_final.keras')
    print(f"\nTERMINE ! Resultats sauvegardes dans : {output_dir}")
    print(f"  - model_{args.fusion}_fusion_best.keras  (meilleur modele pendant l'entrainement)")
    print(f"  - model_{args.fusion}_fusion_final.keras (modele apres restore_best_weights)")
    print(f"  - training_curves_{args.fusion}_fusion.png")
    print(f"  - confusion_matrix_{args.fusion}_fusion.png")
    print(f"  - classification_report_{args.fusion}_fusion.txt")


if __name__ == '__main__':
    main()