import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


DATA_DIR = Path("data")
MAX_SEQ_LEN = 50   # max sesiones por usuario
N_APPS = 5         # consistente con generate_multimodal_data
RNG = np.random.default_rng(123)


# ==========================
# 1. DATASET
# ==========================

def load_data():
    users_path = DATA_DIR / "users_ltv.parquet"
    events_path = DATA_DIR / "events_sessions.parquet"

    if not users_path.exists() or not events_path.exists():
        raise FileNotFoundError(
            f"No encuentro {users_path} o {events_path}. "
            "Ejecuta primero: python src/generate_multimodal_data.py"
        )

    print(f"📂 Leyendo {users_path} y {events_path} ...")
    users = pd.read_parquet(users_path)
    events = pd.read_parquet(events_path)

    return users, events


def encode_categoricals(users: pd.DataFrame):
    """Convierte variables categóricas en índices numéricos y devuelve mapeos."""
    cat_cols = ["country", "device", "install_source", "segment"]
    cat_maps = {}

    for col in cat_cols:
        cats = sorted(users[col].unique().tolist())
        mapping = {v: i for i, v in enumerate(cats)}
        cat_maps[col] = mapping
        users[col + "_idx"] = users[col].map(mapping).astype("int64")

    return users, cat_maps


def build_sequences(events: pd.DataFrame, users: pd.DataFrame):
    """
    Construye tensores de secuencia por usuario:
    - app_id (índices)
    - features numéricas por sesión: [session_length, revenue, day_since_install/30]
    """
    # asegurar orden temporal
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    events = events.sort_values(["user_id", "timestamp"])

    seq_app_ids = []
    seq_num_feats = []

    # indexar por user_id para acceso rápido
    grouped = events.groupby("user_id", sort=False)

    for uid in users["user_id"]:
        if uid in grouped.groups:
            grp = grouped.get_group(uid)
            # recortar a MAX_SEQ_LEN últimas sesiones
            grp = grp.tail(MAX_SEQ_LEN)
            # app_id -> [1..N_APPS]
            apps = grp["app_id"].to_numpy(dtype="int64")
            # features numéricas
            num = np.stack(
                [
                    grp["session_length"].to_numpy(dtype="float32"),
                    grp["revenue"].to_numpy(dtype="float32"),
                    (grp["day_since_install"].to_numpy(dtype="float32") / 30.0),
                ],
                axis=1,
            )
        else:
            apps = np.zeros((0,), dtype="int64")
            num = np.zeros((0, 3), dtype="float32")

        # padding a longitud fija
        pad_len = MAX_SEQ_LEN - len(apps)
        if pad_len > 0:
            apps = np.concatenate([apps, np.zeros(pad_len, dtype="int64")])
            num_pad = np.zeros((pad_len, 3), dtype="float32")
            num = np.concatenate([num, num_pad], axis=0)
        else:
            apps = apps[-MAX_SEQ_LEN:]
            num = num[-MAX_SEQ_LEN:, :]

        seq_app_ids.append(apps)
        seq_num_feats.append(num)

    seq_app_ids = np.stack(seq_app_ids, axis=0)      # [N, T]
    seq_num_feats = np.stack(seq_num_feats, axis=0)  # [N, T, 3]
    return seq_app_ids, seq_num_feats


class LtvDataset(Dataset):
    def __init__(self, users_df, seq_app_ids, seq_num_feats, tab_scaler=None, fit_scaler=False):
        self.users_df = users_df.reset_index(drop=True)
        self.seq_app_ids = seq_app_ids
        self.seq_num_feats = seq_num_feats

        # columnas tabulares que usaremos (numéricas + categoricas codificadas)
        self.tab_cols = [
            "age",
            "days_since_install",
            "total_sessions_30d",
            "total_revenue_30d",
            "avg_session_length",
            "country_idx",
            "device_idx",
            "install_source_idx",
            "segment_idx",
        ]

        tab = self.users_df[self.tab_cols].to_numpy(dtype="float32")

        if tab_scaler is None:
            tab_scaler = StandardScaler()

        if fit_scaler:
            tab = tab_scaler.fit_transform(tab)
        else:
            tab = tab_scaler.transform(tab)

        self.tab_feats = tab.astype("float32")
        self.tab_scaler = tab_scaler

        # targets
        self.y_ltv = self.users_df["ltv_true_30d"].to_numpy(dtype="float32")
        self.y_ret = self.users_df["return_7d"].to_numpy(dtype="float32")

    def __len__(self):
        return len(self.users_df)

    def __getitem__(self, idx):
        seq_apps = self.seq_app_ids[idx]
        seq_num = self.seq_num_feats[idx]
        tab = self.tab_feats[idx]
        y_ltv = self.y_ltv[idx]
        y_ret = self.y_ret[idx]
        return (
            torch.tensor(seq_apps, dtype=torch.long),
            torch.tensor(seq_num, dtype=torch.float32),
            torch.tensor(tab, dtype=torch.float32),
            torch.tensor(y_ltv, dtype=torch.float32),
            torch.tensor(y_ret, dtype=torch.float32),
        )


# ==========================
# 2. MODELO TRANSFORMER + TABULAR
# ==========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class LtvTransformerModel(pl.LightningModule):
    def __init__(
        self,
        n_apps: int,
        seq_num_dim: int,
        tab_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        lr: float = 1e-3,
        ltv_loss_weight: float = 1.0,
        ret_loss_weight: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.app_emb = nn.Embedding(num_embeddings=n_apps + 1, embedding_dim=d_model)  # 0 = padding
        self.num_proj = nn.Linear(seq_num_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=128,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=MAX_SEQ_LEN)

        fusion_dim = d_model + tab_dim

        self.mlp_ltv = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.mlp_ret = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.lr = lr
        self.ltv_loss_weight = ltv_loss_weight
        self.ret_loss_weight = ret_loss_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, seq_apps, seq_num, tab_feats):
        """
        seq_apps: [B, T]
        seq_num:  [B, T, 3]
        tab_feats:[B, tab_dim]
        """
        app_emb = self.app_emb(seq_apps)          # [B, T, D]
        num_emb = self.num_proj(seq_num)          # [B, T, D]
        x = app_emb + num_emb                     # fusiona

        x = self.pos_enc(x)
        x = self.transformer(x)                   # [B, T, D]

        # pooling simple: media
        seq_repr = x.mean(dim=1)                  # [B, D]

        fusion = torch.cat([seq_repr, tab_feats], dim=1)  # [B, D+tab_dim]

        ltv = self.mlp_ltv(fusion).squeeze(-1)
        ret_logit = self.mlp_ret(fusion).squeeze(-1)
        return ltv, ret_logit

    def training_step(self, batch, batch_idx):
        seq_apps, seq_num, tab, y_ltv, y_ret = batch
        pred_ltv, pred_ret_logit = self(seq_apps, seq_num, tab)

        loss_ltv = self.mse(pred_ltv, y_ltv)
        loss_ret = self.bce(pred_ret_logit, y_ret)

        loss = self.ltv_loss_weight * loss_ltv + self.ret_loss_weight * loss_ret

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ltv_mse", loss_ltv)
        self.log("train_ret_bce", loss_ret)
        return loss

    def validation_step(self, batch, batch_idx):
        seq_apps, seq_num, tab, y_ltv, y_ret = batch
        pred_ltv, pred_ret_logit = self(seq_apps, seq_num, tab)

        loss_ltv = self.mse(pred_ltv, y_ltv)
        loss_ret = self.bce(pred_ret_logit, y_ret)
        loss = self.ltv_loss_weight * loss_ltv + self.ret_loss_weight * loss_ret

        # métricas adicionales
        pred_ret_prob = torch.sigmoid(pred_ret_logit)
        pred_ret_label = (pred_ret_prob > 0.5).float()
        acc = (pred_ret_label == y_ret).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_ltv_mse", loss_ltv, prog_bar=True)
        self.log("val_ret_bce", loss_ret)
        self.log("val_ret_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ==========================
# 3. MAIN TRAINING + BASELINE
# ==========================

def train_model():
    users, events = load_data()
    users, cat_maps = encode_categoricals(users)
    seq_app_ids, seq_num_feats = build_sequences(events, users)

    # Train/val split
    train_idx, val_idx = train_test_split(
        np.arange(len(users)), test_size=0.2, random_state=42
    )

    # scaler tabular
    scaler = StandardScaler()
    train_ds = LtvDataset(
        users.iloc[train_idx],
        seq_app_ids[train_idx],
        seq_num_feats[train_idx],
        tab_scaler=scaler,
        fit_scaler=True,
    )
    val_ds = LtvDataset(
        users.iloc[val_idx],
        seq_app_ids[val_idx],
        seq_num_feats[val_idx],
        tab_scaler=scaler,
        fit_scaler=False,
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    tab_dim = train_ds.tab_feats.shape[1]
    seq_num_dim = train_ds.seq_num_feats.shape[2]

    print(f"Tabular dim: {tab_dim}, seq_num_dim: {seq_num_dim}")
    print(f"Train users: {len(train_ds)}, Val users: {len(val_ds)}")

    # Modelo
    model = LtvTransformerModel(
        n_apps=N_APPS,
        seq_num_dim=seq_num_dim,
        tab_dim=tab_dim,
        d_model=64,
        n_heads=4,
        num_layers=2,
        lr=1e-3,
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    # ======================
    # Evaluación en valid con métricas clásicas
    # ======================
    model.eval()
    all_y = []
    all_pred = []

    with torch.no_grad():
        for batch in val_loader:
            seq_apps, seq_num, tab, y_ltv, y_ret = batch
            pred_ltv, _ = model(seq_apps, seq_num, tab)
            all_y.append(y_ltv.numpy())
            all_pred.append(pred_ltv.numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print("\n=== Transformer+Tabular model ===")
    print(f"R^2:  {r2:.4f}")
    print(f"MAE:  {mae:.4f}")

    # ======================
    # Baseline tabular simple (sin secuencias)
    # ======================
    print("\nEntrenando baseline LinearRegression solo tabular...")

    X_all = users[train_ds.tab_cols].to_numpy(dtype="float32")
    y_all = users["ltv_true_30d"].to_numpy(dtype="float32")

    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]

    scaler_bl = StandardScaler()
    X_train_sc = scaler_bl.fit_transform(X_train)
    X_val_sc = scaler_bl.transform(X_val)

    lr_model = LinearRegression()
    lr_model.fit(X_train_sc, y_train)

    y_bl_pred = lr_model.predict(X_val_sc)
    r2_bl = r2_score(y_val, y_bl_pred)
    mae_bl = mean_absolute_error(y_val, y_bl_pred)

    print("\n=== Baseline LinearRegression (solo tabular) ===")
    print(f"R^2:  {r2_bl:.4f}")
    print(f"MAE:  {mae_bl:.4f}")

    print("\n📌 Comparación (más alto R^2, más bajo MAE es mejor):")
    print(f"  Transformer+Tabular → R^2={r2:.4f}, MAE={mae:.4f}")
    print(f"  Baseline LinearReg  → R^2={r2_bl:.4f}, MAE={mae_bl:.4f}")


if __name__ == "__main__":
    pl.seed_everything(42)
    train_model()

