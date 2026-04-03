from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROFILES = {
    "Battery": {"hazard_base": 90, "contains_lithium": 1, "contains_lead": 0, "contains_mercury": 0, "contains_cadmium": 1, "contains_cfc": 0, "recyclable": 0, "material_type": "electrochemical", "weight_class": "light"},
    "PCB": {"hazard_base": 85, "contains_lithium": 0, "contains_lead": 1, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 1, "material_type": "composite", "weight_class": "light"},
    "Mobile": {"hazard_base": 80, "contains_lithium": 1, "contains_lead": 0, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 1, "material_type": "composite", "weight_class": "light"},
    "Television": {"hazard_base": 82, "contains_lithium": 0, "contains_lead": 1, "contains_mercury": 1, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 0, "material_type": "composite", "weight_class": "heavy"},
    "Laptop": {"hazard_base": 78, "contains_lithium": 1, "contains_lead": 1, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 1, "material_type": "composite", "weight_class": "medium"},
    "light bulbs": {"hazard_base": 75, "contains_lithium": 0, "contains_lead": 0, "contains_mercury": 1, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 0, "material_type": "glass", "weight_class": "light"},
    "Refrigerator": {"hazard_base": 88, "contains_lithium": 0, "contains_lead": 0, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 1, "recyclable": 1, "material_type": "metal_cfc", "weight_class": "heavy"},
    "Air-Conditioner": {"hazard_base": 85, "contains_lithium": 0, "contains_lead": 0, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 1, "recyclable": 1, "material_type": "metal_cfc", "weight_class": "medium"},
    "Microwave": {"hazard_base": 60, "contains_lithium": 0, "contains_lead": 0, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 1, "material_type": "metal", "weight_class": "heavy"},
    "Washing Machine": {"hazard_base": 45, "contains_lithium": 0, "contains_lead": 0, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 1, "material_type": "metal", "weight_class": "heavy"},
    "Printer": {"hazard_base": 55, "contains_lithium": 0, "contains_lead": 1, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 1, "material_type": "composite", "weight_class": "medium"},
    "Microchip-IC": {"hazard_base": 62, "contains_lithium": 0, "contains_lead": 1, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 1, "material_type": "silicon", "weight_class": "light"},
    "Keyboard": {"hazard_base": 20, "contains_lithium": 0, "contains_lead": 0, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 1, "material_type": "plastic", "weight_class": "light"},
    "Mouse": {"hazard_base": 18, "contains_lithium": 0, "contains_lead": 0, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 1, "material_type": "plastic", "weight_class": "light"},
    "Resistor": {"hazard_base": 15, "contains_lithium": 0, "contains_lead": 0, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 1, "material_type": "ceramic", "weight_class": "light"},
    "transistor": {"hazard_base": 20, "contains_lithium": 0, "contains_lead": 0, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 1, "material_type": "semiconductor", "weight_class": "light"},
    "heat-sink": {"hazard_base": 10, "contains_lithium": 0, "contains_lead": 0, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 1, "material_type": "metal", "weight_class": "light"},
    "Passive-Component": {"hazard_base": 22, "contains_lithium": 0, "contains_lead": 0, "contains_mercury": 0, "contains_cadmium": 0, "contains_cfc": 0, "recyclable": 1, "material_type": "semiconductor", "weight_class": "light"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ann hazard scoring pipeline")
    parser.add_argument("--output-dir", type=str, default="models/ann")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class TabDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class HazardAnn(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def build_tabular_dataset(samples_per_class: int, seed: int) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []

    for component, profile in PROFILES.items():
        for _ in range(samples_per_class):
            age = np.random.uniform(0.5, 12)
            wt = np.random.uniform(0.01, 60)
            cond = np.random.choice(["working", "damaged", "broken"], p=[0.25, 0.45, 0.30])
            reg = np.random.choice(["low", "medium", "high"], p=[0.25, 0.45, 0.30])
            disp = np.random.choice(["formal", "informal", "none"], p=[0.30, 0.35, 0.35])

            score = np.clip(
                profile["hazard_base"]
                + np.random.normal(0, 4)
                + min(age * 1.2, 12)
                + {"working": -5, "damaged": 2, "broken": 8}[cond]
                + {"low": -4, "medium": 0, "high": 6}[reg]
                + {"formal": -3, "informal": 4, "none": 2}[disp]
                + profile["contains_lithium"] * 8
                + profile["contains_lead"] * 10
                + profile["contains_mercury"] * 12
                + profile["contains_cadmium"] * 9
                + profile["contains_cfc"] * 15,
                0,
                100,
            )

            rows.append(
                {
                    "component": component,
                    "age_years": round(age, 2),
                    "weight_kg": round(wt, 2),
                    "contains_lithium": profile["contains_lithium"],
                    "contains_lead": profile["contains_lead"],
                    "contains_mercury": profile["contains_mercury"],
                    "contains_cadmium": profile["contains_cadmium"],
                    "contains_cfc": profile["contains_cfc"],
                    "recyclable": profile["recyclable"],
                    "material_type": profile["material_type"],
                    "weight_class": profile["weight_class"],
                    "condition": cond,
                    "region_risk": reg,
                    "disposal_history": disp,
                    "hazard_score": round(score, 2),
                }
            )

    return pd.DataFrame(rows)


def to_hazard_class(scores: np.ndarray) -> list[str]:
    return ["HIGH" if x >= 70 else "MEDIUM" if x >= 40 else "LOW" for x in scores]


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    df = build_tabular_dataset(args.samples_per_class, args.seed)
    csv_path = output_dir / "ewaste_tabular_18cls.csv"
    df.to_csv(csv_path, index=False)

    le_comp = LabelEncoder()
    le_mat = LabelEncoder()
    le_wc = LabelEncoder()
    le_cond = LabelEncoder()
    le_reg = LabelEncoder()
    le_disp = LabelEncoder()

    df["comp_enc"] = le_comp.fit_transform(df["component"])
    df["mat_enc"] = le_mat.fit_transform(df["material_type"])
    df["wc_enc"] = le_wc.fit_transform(df["weight_class"])
    df["cond_enc"] = le_cond.fit_transform(df["condition"])
    df["reg_enc"] = le_reg.fit_transform(df["region_risk"])
    df["disp_enc"] = le_disp.fit_transform(df["disposal_history"])

    features = [
        "comp_enc",
        "age_years",
        "weight_kg",
        "contains_lithium",
        "contains_lead",
        "contains_mercury",
        "contains_cadmium",
        "contains_cfc",
        "recyclable",
        "mat_enc",
        "wc_enc",
        "cond_enc",
        "reg_enc",
        "disp_enc",
    ]

    x = df[features].values.astype(np.float32)
    y = df["hazard_score"].values.astype(np.float32)

    x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, test_size=0.30, random_state=args.seed)
    x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.50, random_state=args.seed)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    train_loader = DataLoader(TabDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TabDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TabDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = HazardAnn(len(features), dropout=0.3).to(device)
    criterion = nn.HuberLoss(delta=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    patience = 25

    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            print(f"epoch {epoch:03d} | train={train_loss:.4f} val={val_loss:.4f}")

    if best_state is None:
        raise RuntimeError("training failed to capture a best checkpoint")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "ann_best_18cls.pth")

    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred.extend(model(xb).cpu().squeeze().numpy())
            true.extend(yb.squeeze().numpy())

    pred = np.array(pred)
    true = np.array(true)

    mae = float(mean_absolute_error(true, pred))
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    r2 = float(r2_score(true, pred))
    mape = float(np.mean(np.abs((true - pred) / (true + 1e-8))) * 100)

    cls_true = to_hazard_class(true)
    cls_pred = to_hazard_class(pred)
    cls_acc = float(accuracy_score(cls_true, cls_pred))
    cls_f1 = float(f1_score(cls_true, cls_pred, average="macro"))

    results = {
        "model": "HazardAnn",
        "features": features,
        "n_samples": {
            "train": len(x_train),
            "val": len(x_val),
            "test": len(x_test),
        },
        "regression_metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "mape": round(mape, 4),
        },
        "classification_metrics": {
            "hazard_class_accuracy": round(cls_acc, 4),
            "hazard_macro_f1": round(cls_f1, 4),
        },
        "classification_report": classification_report(
            cls_true,
            cls_pred,
            labels=["HIGH", "MEDIUM", "LOW"],
            output_dict=True,
            zero_division=0,
        ),
        "history": history,
    }

    with (output_dir / "ann_results_18cls.json").open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print("ann hazard pipeline complete")
    print(f"mae={mae:.4f} rmse={rmse:.4f} r2={r2:.4f} hazard_acc={cls_acc:.4f}")


if __name__ == "__main__":
    main()
