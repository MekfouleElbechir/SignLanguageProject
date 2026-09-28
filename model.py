"""
05_Code_Source — model.py
Projet ARSL — Architecture multimodale : BiLSTM + Attention par branche + Late Fusion

CONTEXTE :
Le modele precedent (Copie_de_direct_classifier_model.h5) etait un simple BiLSTM
sur les 1662 features fusionnees des le depart (une seule branche), sans attention,
avec seulement Dropout(0.2). Avec 286 exemples d'entrainement, ce modele overfittait.

CE MODELE CORRIGE 3 CHOSES :
1. Architecture multi-branches (Late Fusion) : chaque modalite (pose, main_gauche,
   main_droite, visage) est traitee separement avant fusion -- conforme a la
   methodologie du memoire (H3 : Late Fusion vs Early Fusion).
2. Mecanisme d'attention par branche : pondere les frames les plus discriminantes
   au lieu de ne garder que le dernier etat du BiLSTM.
3. Regularisation renforcee (necessaire vu la petite taille du dataset) :
   - Dropout plus eleve (0.3-0.5) a plusieurs endroits
   - L2 regularization sur les poids des BiLSTM et Dense
   - recurrent_dropout dans les LSTM
   - Branches volontairement "etroites" (peu d'unites) pour limiter le nombre
     de parametres par rapport aux ~286 exemples disponibles

UTILISATION (depuis train.py) :
    from model import build_multimodal_model
    model = build_multimodal_model(max_len=150, num_classes=7)
    model.summary()
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AttentionPooling(layers.Layer):
    """
    Couche d'attention additive (style Bahdanau) qui resume une sequence
    (batch, time, features) en un seul vecteur (batch, features), en apprenant
    a ponderer les frames les plus importantes plutot que de prendre juste le
    dernier etat du BiLSTM.

    Respecte le masking (les frames de padding, mask=False, recoivent un poids
    d'attention quasi nul et n'influencent pas le resultat).
    """

    def __init__(self, units=32, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.score_dense1 = layers.Dense(
            self.units, activation='tanh',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name=f'{self.name}_score1'
        )
        self.score_dense2 = layers.Dense(
            1, kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name=f'{self.name}_score2'
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        # inputs: (batch, time, features)
        scores = self.score_dense2(self.score_dense1(inputs))  # (batch, time, 1)
        scores = tf.squeeze(scores, axis=-1)  # (batch, time)

        if mask is not None:
            mask = tf.cast(mask, dtype=scores.dtype)
            scores = scores + (1.0 - mask) * -1e9  # neutralise le padding

        weights = tf.nn.softmax(scores, axis=1)  # (batch, time)
        weights_expanded = tf.expand_dims(weights, axis=-1)  # (batch, time, 1)
        context = tf.reduce_sum(inputs * weights_expanded, axis=1)  # (batch, features)
        return context

    def compute_mask(self, inputs, mask=None):
        # apres l'attention, la sequence temporelle disparait -> plus de mask a propager
        return None

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'l2_reg': self.l2_reg})
        return config


def build_branch(input_tensor, name, proj_dim=48, lstm_units=32, l2_reg=1e-4, dropout_rate=0.4):
    """
    Construit une branche complete pour UNE modalite :
    Masking -> projection (reduction de dimension) -> BiLSTM -> Attention -> Dropout

    La projection (TimeDistributed Dense) est importante pour la branche 'visage'
    (1404 features brutes) : sans elle, le BiLSTM aurait beaucoup trop de parametres
    par rapport aux 286 exemples d'entrainement disponibles.
    """
    x = layers.Masking(mask_value=0.0, name=f'{name}_masking')(input_tensor)
    x = layers.TimeDistributed(
        layers.Dense(proj_dim, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(l2_reg)),
        name=f'{name}_projection'
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True,
                     dropout=dropout_rate, recurrent_dropout=0.1,
                     kernel_regularizer=keras.regularizers.l2(l2_reg)),
        name=f'{name}_bilstm'
    )(x)
    x = AttentionPooling(units=lstm_units, l2_reg=l2_reg, name=f'{name}_attention')(x)
    x = layers.Dropout(dropout_rate, name=f'{name}_dropout')(x)
    return x


def build_multimodal_model(max_len=150, num_classes=7, l2_reg=1e-4, dropout_rate=0.4,
                             fusion_units=64):
    """
    Construit le modele complet : 4 branches (pose, main_gauche, main_droite, visage)
    -> Late Fusion (concatenation APRES traitement temporel independant de chaque branche)
    -> classification.

    Les dimensions d'entree correspondent exactement aux features produites par
    MediaPipe Holistic (voir preprocessing.py / Data_Dictionary.md) :
        pose : 132, main_gauche : 63, main_droite : 63, visage : 1404
    """
    input_pose = keras.Input(shape=(max_len, 132), name='pose')
    input_main_g = keras.Input(shape=(max_len, 63), name='main_gauche')
    input_main_d = keras.Input(shape=(max_len, 63), name='main_droite')
    input_visage = keras.Input(shape=(max_len, 1404), name='visage')

    # Branches independantes -- volontairement etroites (peu d'unites) car peu de donnees
    branch_pose = build_branch(input_pose, 'pose', proj_dim=32, lstm_units=24,
                                 l2_reg=l2_reg, dropout_rate=dropout_rate)
    branch_main_g = build_branch(input_main_g, 'main_gauche', proj_dim=32, lstm_units=24,
                                   l2_reg=l2_reg, dropout_rate=dropout_rate)
    branch_main_d = build_branch(input_main_d, 'main_droite', proj_dim=32, lstm_units=24,
                                   l2_reg=l2_reg, dropout_rate=dropout_rate)
    branch_visage = build_branch(input_visage, 'visage', proj_dim=48, lstm_units=32,
                                   l2_reg=l2_reg, dropout_rate=dropout_rate)

    # --- LATE FUSION : concatenation APRES le traitement temporel de chaque branche ---
    fused = layers.Concatenate(name='late_fusion')(
        [branch_pose, branch_main_g, branch_main_d, branch_visage]
    )

    x = layers.Dense(fusion_units, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(l2_reg),
                       name='fusion_dense')(fused)
    x = layers.Dropout(0.5, name='fusion_dropout')(x)  # dropout fort avant la sortie finale
    output = layers.Dense(num_classes, activation='softmax', name='classification')(x)

    model = keras.Model(
        inputs=[input_pose, input_main_g, input_main_d, input_visage],
        outputs=output,
        name='ARSL_multimodal_bilstm_attention_latefusion'
    )
    return model


def build_early_fusion_model(max_len=150, num_classes=7, l2_reg=1e-4, dropout_rate=0.4,
                               proj_dim=32):
    """
    Modele de comparaison pour l'experience ablative H3 (Early Fusion) :
    les 4 modalites sont fusionnees AVANT le traitement temporel (BiLSTM), au lieu
    d'etre traitees separement comme dans build_multimodal_model() (Late Fusion).

    IMPORTANT (correction apres Exp2 v1) : chaque modalite est d'abord projetee a la
    MEME dimension (proj_dim) avant la concatenation. Sans cette etape, la concatenation
    brute donnerait un vecteur ou le visage (1404 features) represente 85% du total
    contre 15% pour les mains+pose reunies -- le signal des mains (souvent le plus
    discriminant) est alors noye numeriquement des le depart, ce qui a cause un
    effondrement de l'apprentissage (accuracy bloquee ~14-20%, proche du hasard).
    Cette projection egalise l'influence de chaque modalite AVANT la fusion, tout en
    respectant la definition de l'Early Fusion (fusion avant le BiLSTM/traitement
    temporel, contrairement a la Late Fusion qui fusionne apres).
    """
    input_pose = keras.Input(shape=(max_len, 132), name='pose')
    input_main_g = keras.Input(shape=(max_len, 63), name='main_gauche')
    input_main_d = keras.Input(shape=(max_len, 63), name='main_droite')
    input_visage = keras.Input(shape=(max_len, 1404), name='visage')

    def project(inp, name):
        x = layers.Masking(mask_value=0.0, name=f'{name}_masking')(inp)
        x = layers.TimeDistributed(
            layers.Dense(proj_dim, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(l2_reg)),
            name=f'{name}_projection'
        )(x)
        return x

    proj_pose = project(input_pose, 'pose')
    proj_main_g = project(input_main_g, 'main_gauche')
    proj_main_d = project(input_main_d, 'main_droite')
    proj_visage = project(input_visage, 'visage')

    # --- EARLY FUSION : concatenation AVANT le BiLSTM, mais APRES projection egalisee ---
    fused = layers.Concatenate(name='early_fusion')(
        [proj_pose, proj_main_g, proj_main_d, proj_visage]
    )

    x = layers.Bidirectional(
        layers.LSTM(48, return_sequences=True, dropout=dropout_rate, recurrent_dropout=0.1,
                     kernel_regularizer=keras.regularizers.l2(l2_reg)),
        name='bilstm'
    )(fused)
    x = AttentionPooling(units=48, l2_reg=l2_reg, name='attention')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg),
                       name='dense')(x)
    x = layers.Dropout(0.5, name='final_dropout')(x)
    output = layers.Dense(num_classes, activation='softmax', name='classification')(x)

    model = keras.Model(
        inputs=[input_pose, input_main_g, input_main_d, input_visage],
        outputs=output,
        name='ARSL_multimodal_bilstm_attention_earlyfusion'
    )
    return model


def build_hands_only_model(max_len=150, num_classes=7, l2_reg=1e-4, dropout_rate=0.4,
                             fusion_units=48):
    """
    Modele de comparaison pour l'experience ablative H1 (mains seules vs holistic complet) :
    seules les deux branches main_gauche et main_droite sont utilisees (pose et visage
    retires), avec la meme logique de Late Fusion que build_multimodal_model().

    L'ecart de performance avec build_multimodal_model() (holistic complet, voir Exp1)
    quantifie directement l'apport des modalites non-manuelles (visage + posture) --
    c'est la mesure recherchee par l'hypothese H1.
    """
    input_main_g = keras.Input(shape=(max_len, 63), name='main_gauche')
    input_main_d = keras.Input(shape=(max_len, 63), name='main_droite')

    branch_main_g = build_branch(input_main_g, 'main_gauche', proj_dim=32, lstm_units=24,
                                   l2_reg=l2_reg, dropout_rate=dropout_rate)
    branch_main_d = build_branch(input_main_d, 'main_droite', proj_dim=32, lstm_units=24,
                                   l2_reg=l2_reg, dropout_rate=dropout_rate)

    fused = layers.Concatenate(name='late_fusion_hands')([branch_main_g, branch_main_d])

    x = layers.Dense(fusion_units, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(l2_reg),
                       name='fusion_dense')(fused)
    x = layers.Dropout(0.5, name='fusion_dropout')(x)
    output = layers.Dense(num_classes, activation='softmax', name='classification')(x)

    model = keras.Model(
        inputs=[input_main_g, input_main_d],
        outputs=output,
        name='ARSL_hands_only_bilstm_attention'
    )
    return model


def build_branch_no_attention(input_tensor, name, proj_dim=48, lstm_units=32, l2_reg=1e-4,
                                dropout_rate=0.4):
    """
    Variante de build_branch() SANS mecanisme d'attention, utilisee pour l'experience
    ablative H2. Remplace AttentionPooling par un simple GlobalAveragePooling1D (moyenne
    non-ponderee sur le temps) -- pour isoler precisement l'apport du mecanisme d'attention.
    """
    x = layers.Masking(mask_value=0.0, name=f'{name}_masking')(input_tensor)
    x = layers.TimeDistributed(
        layers.Dense(proj_dim, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(l2_reg)),
        name=f'{name}_projection'
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True,
                     dropout=dropout_rate, recurrent_dropout=0.1,
                     kernel_regularizer=keras.regularizers.l2(l2_reg)),
        name=f'{name}_bilstm'
    )(x)
    x = layers.GlobalAveragePooling1D(name=f'{name}_avgpool')(x)  # <-- seule difference avec build_branch()
    x = layers.Dropout(dropout_rate, name=f'{name}_dropout')(x)
    return x


def build_multimodal_model_no_attention(max_len=150, num_classes=7, l2_reg=1e-4,
                                          dropout_rate=0.4, fusion_units=64):
    """
    Modele de comparaison pour l'experience ablative H2 (avec vs sans attention).
    Architecture identique a build_multimodal_model() (4 branches + Late Fusion),
    SAUF que chaque branche utilise GlobalAveragePooling1D au lieu d'AttentionPooling.
    L'ecart de performance avec Exp1 quantifie directement l'apport du mecanisme
    d'attention -- c'est la mesure recherchee par l'hypothese H2.
    """
    input_pose = keras.Input(shape=(max_len, 132), name='pose')
    input_main_g = keras.Input(shape=(max_len, 63), name='main_gauche')
    input_main_d = keras.Input(shape=(max_len, 63), name='main_droite')
    input_visage = keras.Input(shape=(max_len, 1404), name='visage')

    branch_pose = build_branch_no_attention(input_pose, 'pose', proj_dim=32, lstm_units=24,
                                              l2_reg=l2_reg, dropout_rate=dropout_rate)
    branch_main_g = build_branch_no_attention(input_main_g, 'main_gauche', proj_dim=32, lstm_units=24,
                                                l2_reg=l2_reg, dropout_rate=dropout_rate)
    branch_main_d = build_branch_no_attention(input_main_d, 'main_droite', proj_dim=32, lstm_units=24,
                                                l2_reg=l2_reg, dropout_rate=dropout_rate)
    branch_visage = build_branch_no_attention(input_visage, 'visage', proj_dim=48, lstm_units=32,
                                                l2_reg=l2_reg, dropout_rate=dropout_rate)

    fused = layers.Concatenate(name='late_fusion')(
        [branch_pose, branch_main_g, branch_main_d, branch_visage]
    )
    x = layers.Dense(fusion_units, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(l2_reg),
                       name='fusion_dense')(fused)
    x = layers.Dropout(0.5, name='fusion_dropout')(x)
    output = layers.Dense(num_classes, activation='softmax', name='classification')(x)

    model = keras.Model(
        inputs=[input_pose, input_main_g, input_main_d, input_visage],
        outputs=output,
        name='ARSL_multimodal_bilstm_no_attention'
    )
    return model


if __name__ == '__main__':
    # Sanity check rapide : construit le modele et affiche son architecture + nb de parametres
    model = build_multimodal_model(max_len=150, num_classes=7)
    model.summary()
    print(f"\nNombre total de parametres : {model.count_params():,}")
    print("Compare ce nombre a celui du modele precedent (direct_classifier) --")
    print("si celui-ci est nettement plus petit ou du meme ordre, c'est bon signe")
    print("pour limiter l'overfitting avec ~286 exemples d'entrainement.")