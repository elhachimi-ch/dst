"""
Under MIT License by EL HACHIMI CHOUAIB
"""
from .dataframe import DataFrame # from .dataframe import DataFrame in production
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from math import sqrt
from catboost import CatBoostRegressor, CatBoostClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import os
import psutil
import pyarrow.parquet as pq
import pyarrow as pa
from torch.utils.data import IterableDataset
from sklearn.model_selection import StratifiedKFold, KFold

try:
    from accelerate import Accelerator
    _ACCELERATE_AVAILABLE = True
except ImportError:
    _ACCELERATE_AVAILABLE = False
    
import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.warp import reproject, Resampling

import torch
import torch.nn as nn
import torch.nn.functional as F


DATE_RE = re.compile(r".*_(\d{8})(?:\.\w+)?$")  # captures YYYYMMDD at end


def parse_date_int(path: str) -> int:
    m = DATE_RE.match(os.path.splitext(os.path.basename(path))[0])
    if not m:
        raise ValueError(f"Cannot parse date from filename: {path}")
    return int(m.group(1))


def parse_band_id(path: str) -> int:
    # expects ..._B01_YYYYMMDD
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"_B(\d{2})_", base)
    if not m:
        raise ValueError(f"Cannot parse band id from: {path}")
    return int(m.group(1))


def doy_from_yyyymmdd(date_int: int) -> int:
    import datetime as dt
    s = str(date_int)
    d = dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    return int(d.strftime("%j"))  # 1..366


def sincos_doy(doy: int) -> Tuple[float, float]:
    # 365.25 for smoothness; you can use 365 if you want
    x = 2.0 * np.pi * (doy / 365.25)
    return float(np.sin(x)), float(np.cos(x))


def ang_sincos(deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # deg array per-pixel; convert to rad
    rad = np.deg2rad(deg.astype(np.float32))
    return np.sin(rad), np.cos(rad)


@dataclass
class SamplePaths:
    date: int
    aqua_bands: Dict[int, str]   # band_id -> path
    terra_bands: Dict[int, str]  # band_id -> path
    sza: str
    vza: str


def index_datalake(
    data_lake: str,
    require_bands: List[int] = [1,2,3,4,5,6,7],
    *,
    prefix_aqua: str = "aqua_toa_B",
    prefix_terra: str = "terra_toa_B",
    prefix_solar_zenith: str = "SolarZenith_",
    prefix_sensor_zenith: str = "SensorZenith_",
) -> List[SamplePaths]:
    # Find files
    aqua_files  = glob.glob(os.path.join(data_lake, f"{prefix_aqua}*_*.tif")) + \
                  glob.glob(os.path.join(data_lake, f"{prefix_aqua}*_*"))
    terra_files = glob.glob(os.path.join(data_lake, f"{prefix_terra}*_*.tif")) + \
                  glob.glob(os.path.join(data_lake, f"{prefix_terra}*_*"))
    sza_files   = glob.glob(os.path.join(data_lake, f"{prefix_solar_zenith}*_*.tif")) + \
                  glob.glob(os.path.join(data_lake, f"{prefix_solar_zenith}*_*"))
    vza_files   = glob.glob(os.path.join(data_lake, f"{prefix_sensor_zenith}*_*.tif")) + \
                  glob.glob(os.path.join(data_lake, f"{prefix_sensor_zenith}*_*"))

    def group_by_date_band(files: List[str]) -> Dict[int, Dict[int, str]]:
        out: Dict[int, Dict[int, str]] = {}
        for p in files:
            date = parse_date_int(p)
            b = parse_band_id(p)
            out.setdefault(date, {})[b] = p
        return out

    def group_by_date(files: List[str]) -> Dict[int, str]:
        out: Dict[int, str] = {}
        for p in files:
            date = parse_date_int(p)
            out[date] = p
        return out

    aqua = group_by_date_band(aqua_files)
    terra = group_by_date_band(terra_files)
    sza = group_by_date(sza_files)
    vza = group_by_date(vza_files)

    # intersection by date where all required bands + angles exist
    common_dates = sorted(set(aqua.keys()) & set(terra.keys()) & set(sza.keys()) & set(vza.keys()))
    samples: List[SamplePaths] = []

    for d in common_dates:
        if all(b in aqua[d] for b in require_bands) and all(b in terra[d] for b in require_bands):
            samples.append(SamplePaths(
                date=d,
                aqua_bands={b: aqua[d][b] for b in require_bands},
                terra_bands={b: terra[d][b] for b in require_bands},
                sza=sza[d],
                vza=vza[d],
            ))
    return samples


def read_raster(path: str) -> Tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    return arr, transform, crs


def reproject_to_match(
    src_arr: np.ndarray,
    src_transform,
    src_crs,
    dst_shape: Tuple[int, int],
    dst_transform,
    dst_crs,
) -> np.ndarray:
    dst = np.empty(dst_shape, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return dst


class TerraAquaPatchDataset(Dataset):
    """
    Returns:
      x: [C_in, H, W]  = Aqua(7) + SZA_sin/cos(2) + VZA_sin/cos(2) + DOY_sin/cos(2) + lat/lon(2) + DEM(1)
      y: [7, H, W]     = Terra(7)
    """
    def __init__(
        self,
        data_lake: str,
        dem_path: str,
        patch_size: int = 128,
        samples_per_image: int = 64,
        years: Optional[Tuple[int, int]] = None,  # (min_year, max_year) inclusive
        nodata_value: Optional[float] = None,
        prefix_aqua: str = "aqua_toa_B",
        prefix_terra: str = "terra_toa_B",
        prefix_solar_zenith: str = "SolarZenith_",
        prefix_sensor_zenith: str = "SensorZenith_",
    ):
        self.samples = index_datalake(
            data_lake,
            prefix_aqua=prefix_aqua,
            prefix_terra=prefix_terra,
            prefix_solar_zenith=prefix_solar_zenith,
            prefix_sensor_zenith=prefix_sensor_zenith,
        )
        if years is not None:
            y0, y1 = years
            self.samples = [s for s in self.samples if y0 <= int(str(s.date)[:4]) <= y1]

        if len(self.samples) == 0:
            raise RuntimeError("No valid paired dates found (Aqua+Terra bands 1-7 + SZA+VZA).")

        self.dem_path = dem_path
        self.patch = patch_size
        self.spi = samples_per_image
        self.nodata_value = nodata_value

        # Load DEM once (will reproject per scene if needed)
        self.dem_arr, self.dem_transform, self.dem_crs = read_raster(dem_path)

    def __len__(self) -> int:
        # each date yields multiple random patches
        return len(self.samples) * self.spi

    def _get_scene(self, sample: SamplePaths):
        # read one band to get reference grid
        ref_path = sample.aqua_bands[1]
        ref, transform, crs = read_raster(ref_path)

        H, W = ref.shape

        # read all Aqua bands
        aqua = []
        for b in range(1, 8):
            arr, _, _ = read_raster(sample.aqua_bands[b])
            aqua.append(arr)
        aqua = np.stack(aqua, axis=0)  # [7,H,W]

        # read all Terra bands
        terra = []
        for b in range(1, 8):
            arr, _, _ = read_raster(sample.terra_bands[b])
            terra.append(arr)
        terra = np.stack(terra, axis=0)  # [7,H,W]

        # angles
        sza, _, _ = read_raster(sample.sza)
        vza, _, _ = read_raster(sample.vza)

        # DEM reproject if needed
        if self.dem_crs != crs or self.dem_transform != transform or self.dem_arr.shape != (H, W):
            dem = reproject_to_match(
                self.dem_arr, self.dem_transform, self.dem_crs,
                (H, W), transform, crs
            )
        else:
            dem = self.dem_arr

        return aqua, terra, sza, vza, dem, transform

    def _random_patch(self, H: int, W: int) -> Tuple[int, int]:
        ps = self.patch
        if H < ps or W < ps:
            raise ValueError(f"Patch {ps} bigger than raster {H}x{W}.")
        top = np.random.randint(0, H - ps + 1)
        left = np.random.randint(0, W - ps + 1)
        return top, left

    def __getitem__(self, idx: int):
        scene_idx = idx // self.spi
        s = self.samples[scene_idx]

        aqua, terra, sza, vza, dem, transform = self._get_scene(s)
        _, H, W = aqua.shape
        top, left = self._random_patch(H, W)
        ps = self.patch

        # crop
        aqua_p = aqua[:, top:top+ps, left:left+ps]
        terra_p = terra[:, top:top+ps, left:left+ps]
        sza_p   = sza[top:top+ps, left:left+ps]
        vza_p   = vza[top:top+ps, left:left+ps]
        dem_p   = dem[top:top+ps, left:left+ps]

        # optional nodata mask
        if self.nodata_value is not None:
            mask = np.all(aqua_p != self.nodata_value, axis=0) & np.all(terra_p != self.nodata_value, axis=0)
        else:
            mask = np.isfinite(aqua_p[0]) & np.isfinite(terra_p[0])

        # angles -> sin/cos
        sza_s, sza_c = ang_sincos(sza_p)
        vza_s, vza_c = ang_sincos(vza_p)

        # DOY -> scalar sin/cos broadcast
        doy = doy_from_yyyymmdd(s.date)
        doy_s, doy_c = sincos_doy(doy)
        doy_s_ch = np.full((ps, ps), doy_s, dtype=np.float32)
        doy_c_ch = np.full((ps, ps), doy_c, dtype=np.float32)

        # lat/lon channels from affine transform (in CRS units; if lat/lon is required, use EPSG:4326 export)
        # Here we create normalized x/y coordinates in pixel space as a stable positional encoding
        yy, xx = np.mgrid[top:top+ps, left:left+ps].astype(np.float32)
        # normalize to [-1,1] in scene coordinates
        x_norm = (xx / max(W-1, 1)) * 2.0 - 1.0
        y_norm = (yy / max(H-1, 1)) * 2.0 - 1.0

        # stack inputs
        x = np.concatenate([
            aqua_p,                                # 7
            sza_s[None, ...], sza_c[None, ...],     # 2
            vza_s[None, ...], vza_c[None, ...],     # 2
            doy_s_ch[None, ...], doy_c_ch[None, ...],  # 2
            x_norm[None, ...], y_norm[None, ...],   # 2 (positional)
            dem_p[None, ...],                       # 1
        ], axis=0).astype(np.float32)              # total = 16 channels

        y = terra_p.astype(np.float32)

        # simple scaling suggestion (TOA often in scaled int; adapt to your products)
        # If values look like 0..10000, normalize to 0..1
        x[:7] = x[:7] / 10000.0
        y     = y     / 10000.0

        # apply mask to y (optional): keep masked pixels as 0; also return mask for loss
        mask_t = mask.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask_t)




from torch.utils.data import DataLoader
from tqdm import tqdm

def masked_l1(pred, target, mask):
    # pred/target: [B,C,H,W], mask: [B,H,W]
    m = mask.unsqueeze(1)  # [B,1,H,W]
    return (torch.abs(pred - target) * m).sum() / (m.sum() * pred.shape[1] + 1e-6)


def sam_loss(pred, target, mask):
    # Spectral Angle Mapper over channels
    # pred/target: [B,C,H,W], mask: [B,H,W]
    m = mask.unsqueeze(1)  # [B,1,H,W]
    # flatten
    p = pred * m
    t = target * m
    dot = (p * t).sum(dim=1)  # [B,H,W]
    pn = torch.sqrt((p * p).sum(dim=1) + 1e-6)
    tn = torch.sqrt((t * t).sum(dim=1) + 1e-6)
    cos = dot / (pn * tn + 1e-6)
    cos = torch.clamp(cos, -1.0, 1.0)
    ang = torch.acos(cos)  # radians
    # average only valid pixels
    return (ang * mask).sum() / (mask.sum() + 1e-6)


def train_one_epoch(model, loader, opt, device, lam_sam=0.1):
    model.train()
    total = 0.0
    for x, y, mask in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        pred = model(x)
        l1 = masked_l1(pred, y, mask)
        sam = sam_loss(pred, y, mask)
        loss = l1 + lam_sam * sam

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


@torch.no_grad()
def valid_one_epoch(model, loader, device, lam_sam=0.1):
    model.eval()
    total = 0.0
    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        pred = model(x)
        l1 = masked_l1(pred, y, mask)
        sam = sam_loss(pred, y, mask)
        loss = l1 + lam_sam * sam
        total += float(loss.item())
    return total / max(len(loader), 1)





class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class WindowAttention(nn.Module):
    """
    Simple window-based multi-head self-attention (Swin-style idea).
    Not a full Swin implementation (no relative position bias tables), but strong enough for your paper + efficient.
    """
    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ws = window_size
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        ws = self.ws

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        # partition windows
        xw = x.unfold(2, ws, ws).unfold(3, ws, ws)  # [B,C,Hp/ws,Wp/ws,ws,ws]
        xw = xw.permute(0, 2, 3, 4, 5, 1).contiguous()  # [B,nh,nw,ws,ws,C]
        nH = xw.shape[1]
        nW = xw.shape[2]
        xw = xw.view(B * nH * nW, ws * ws, C)  # [B*nwin, tokens, C]

        # attention + residual
        h = self.ln1(xw)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        xw = xw + attn_out

        # mlp + residual
        h2 = self.ln2(xw)
        xw = xw + self.mlp(h2)

        # merge windows back
        xw = xw.view(B, nH, nW, ws, ws, C)
        xw = xw.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B,C,nH,ws,nW,ws]
        x = xw.view(B, C, Hp, Wp)

        # remove pad
        x = x[:, :, :H, :W]
        return x


class SwinUNetRegressor(nn.Module):
    def __init__(self, in_ch=16, out_ch=7, base=64, window=8):
        super().__init__()

        self.enc1 = ConvBlock(in_ch, base)
        self.att1 = WindowAttention(base, num_heads=4, window_size=window)
        self.down1 = nn.Conv2d(base, base*2, 2, stride=2)

        self.enc2 = ConvBlock(base*2, base*2)
        self.att2 = WindowAttention(base*2, num_heads=4, window_size=window)
        self.down2 = nn.Conv2d(base*2, base*4, 2, stride=2)

        self.enc3 = ConvBlock(base*4, base*4)
        self.att3 = WindowAttention(base*4, num_heads=8, window_size=window)
        self.down3 = nn.Conv2d(base*4, base*8, 2, stride=2)

        self.bottleneck = ConvBlock(base*8, base*8)
        self.attb = WindowAttention(base*8, num_heads=8, window_size=window)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = ConvBlock(base*8, base*4)  # concat skip

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock(base*2, base)

        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.att1(self.enc1(x))
        x = self.down1(e1)

        e2 = self.att2(self.enc2(x))
        x = self.down2(e2)

        e3 = self.att3(self.enc3(x))
        x = self.down3(e3)

        x = self.attb(self.bottleneck(x))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, e3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, e2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, e1], dim=1))

        y = self.head(x)  # [B,7,H,W]
        return y




    
    
 # 7-token Transformer with FiLM conditionin   FiLM (Feature-wise Linear Modulation) 



class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(cond_dim, 2 * hidden_dim)
    
    def forward(self, x, cond):
        """
        x: (B, L, hidden_dim)
        cond: (B, cond_dim)
        """
        gamma_beta = self.fc(cond)  # (B, 2*hidden_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each (B, hidden_dim)
        gamma = gamma.unsqueeze(1)  # (B, 1, hidden_dim)
        beta = beta.unsqueeze(1)    # (B, 1, hidden_dim)
        return gamma * x + beta


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden, cond_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        self.film_after_attn = FiLM(cond_dim, embed_dim)
        self.film_after_ffn = FiLM(cond_dim, embed_dim)

    def forward(self, x, cond):
        # Self-attention + residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.film_after_attn(x, cond)  # FiLM after attention

        # FFN + residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        x = self.film_after_ffn(x, cond)   # FiLM after FFN
        return x
    
    

# example usage:
"""# Dummy data
B = 32
aqua = torch.rand(B, 7)          # normalized reflectance (e.g., 0–1)
cond = torch.rand(B, 16)         # e.g., [solar_zenith, sensor_zenith, rel_azimuth,
                                 #         sin(doy), cos(doy), am_pm_flag,
                                 #         land_cover_embedding (8-dim), ...]

model = SpectralTransformer(
    input_bands=7,
    output_bands=7,
    embed_dim=64,
    num_heads=4,
    num_layers=3,
    ff_hidden=128,
    cond_dim=16
)

terra_pred = model(aqua, cond)   # (B, 7)
print(terra_pred.shape)          # torch.Size([32, 7])"""


class SpectralTransformer(nn.Module):
    def __init__(
        self,
        input_bands=7,
        output_bands=7,
        embed_dim=64,
        num_heads=4,
        num_layers=3,
        ff_hidden=128,
        cond_dim=16,        # dimension of auxiliary conditioning vector
        dropout=0.1
    ):
        super().__init__()
        self.input_bands = input_bands
        self.embed_dim = embed_dim

        # Input projection: from scalar band value to embed_dim
        self.input_proj = nn.Linear(1, embed_dim)  # applied per band

        # Learnable positional encoding for 7 bands
        self.pos_embed = nn.Parameter(torch.randn(1, input_bands, embed_dim))

        # Conditioning encoder (optional: could be identity if cond already embedded)
        self.cond_encoder = nn.Linear(cond_dim, cond_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden, cond_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output head: per-token linear map back to scalar
        self.output_proj = nn.Linear(embed_dim, 1)

    def forward(self, aqua_vec, cond):
        """
        aqua_vec: (B, 7) — Aqua reflectance bands 1–7
        cond:     (B, cond_dim) — auxiliary conditions (geometry, land cover, etc.)
        Returns:  (B, 7) — predicted Terra reflectance
        """
        B = aqua_vec.shape[0]
        
        # Expand each band to (B, 7, 1), then embed to (B, 7, embed_dim)
        x = aqua_vec.unsqueeze(-1)  # (B, 7, 1)
        x = self.input_proj(x)      # (B, 7, embed_dim)

        # Add positional encoding
        x = x + self.pos_embed  # broadcasting over batch

        # Encode condition
        cond = self.cond_encoder(cond)  # (B, cond_dim)

        # Pass through Transformer layers with FiLM
        for layer in self.layers:
            x = layer(x, cond)

        # Project each token back to scalar
        out = self.output_proj(x).squeeze(-1)  # (B, 7)

        return out
 
 
 


class TokenizerOneToken(nn.Module):
    def __init__(self, num_features, dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.embedding = nn.Linear(num_features, dim)

    def forward(self, x):
        x_embed = self.embedding(x).unsqueeze(1)  # [B, 1, dim]
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, dim]
        return torch.cat([cls_token, x_embed], dim=1)  # [B, 2, dim]

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.last_attn_weights = None  # to store attention weights

    def forward(self, x):
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        self.last_attn_weights = attn_weights  # shape: [B, num_heads, seq_len, seq_len]
        x = self.norm1(x + self.drop1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out))
        return x

class FTTransformer(nn.Module):
    def __init__(self, num_features, dim=64, depth=4, heads=8, ff_dim=256, dropout=0.1):
        super().__init__()
        self.tokenizer = TokenizerOneToken(num_features, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, ff_dim, dropout) for _ in range(depth)
        ])
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.tokenizer(x)  # [B, 2, dim]
        for block in self.blocks:
            x = block(x)
        cls_embedding = x[:, 0, :]  # [B, dim]
        return self.head(cls_embedding)  # [B, 1]

    def get_first_layer_attention(self):
        return self.blocks[0].last_attn_weights  # [B, num_heads, 2, 2]




class ParquetIterableDS(IterableDataset):
    """
    Top-level, picklable IterableDataset for streaming a single Parquet file.
    - No ParquetFile handle is stored on the instance (opened per-worker in __iter__).
    - Works with num_workers > 0 on Windows (spawn).
    """
    def __init__(
        self,
        parquet_path: str,
        feature_cols: list[str],
        target_col: str = "y",
        batch_size: int = 256,
        indices=None,
        shuffle: bool = True,
        seed: int = 42,
        name: str = "IO-Loader",
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.columns = self.feature_cols + [self.target_col]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.name = name

        # Precompute row-group metadata and selected indices (store only Python lists/arrays)
        pf = pq.ParquetFile(self.parquet_path, memory_map=True)
        self.num_row_groups = pf.num_row_groups
        self.rg_offsets = []
        rg_sizes = []
        running = 0
        for rg in range(self.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            self.rg_offsets.append((running, running + n))
            rg_sizes.append(n)
            running += n
        self.total_rows = running

        import numpy as np
        if indices is None:
            self.selected_per_rg = {rg: None for rg in range(self.num_row_groups)}
            per_rg_counts = rg_sizes[:]
        else:
            idx = np.asarray(indices, dtype=np.int32)
            idx.sort()
            self.selected_per_rg = {}
            per_rg_counts = []
            for rg, (start, end) in enumerate(self.rg_offsets):
                left = np.searchsorted(idx, start, side="left")
                right = np.searchsorted(idx, end, side="left")
                local = idx[left:right] - start
                if local.size > 0:
                    self.selected_per_rg[rg] = local
                    per_rg_counts.append(int(local.size))
                else:
                    self.selected_per_rg[rg] = np.empty((0,), dtype=np.int32)
                    per_rg_counts.append(0)

        self.per_rg_counts = per_rg_counts
        self.total_selected = int(sum(per_rg_counts))
        self.total_batches = int(sum((c + self.batch_size - 1) // self.batch_size for c in per_rg_counts if c > 0)) or 1

    def __len__(self):
        return self.total_batches

    def _table_to_numpy(self, table: pa.Table):
        import numpy as np
        try:
            X_cols = [table[c].to_numpy(zero_copy_only=False) for c in self.feature_cols]
            X = np.column_stack(X_cols).astype(np.float32, copy=False)
            y = table[self.target_col].to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1, 1)
            return X, y
        except Exception:
            pdf = table.to_pandas()
            X = pdf[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
            y = pdf[[self.target_col]].to_numpy(dtype=np.float32, copy=False)
            return X, y

    def __iter__(self):
        import numpy as np, random, torch
        pf = pq.ParquetFile(self.parquet_path, memory_map=True)  # open per-worker
        order = list(range(self.num_row_groups))

        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            wid = worker.id
            wnum = worker.num_workers
        else:
            wid = 0
            wnum = 1

        # Worker-specific RNG for shuffle
        seed = self.seed + wid
        if self.shuffle:
            random.seed(seed)
            random.shuffle(order)

        # Shard: each worker processes a disjoint slice of row groups
        order = order[wid::wnum]

        for rg in order:
            local_idx = self.selected_per_rg.get(rg, None)
            if local_idx is not None and isinstance(local_idx, np.ndarray) and local_idx.size == 0:
                continue

            tbl = pf.read_row_group(rg, columns=self.columns)

            if local_idx is not None:
                if self.shuffle and local_idx.size > 0:
                    rng = np.random.default_rng(seed + rg)
                    local_idx = local_idx.copy()
                    rng.shuffle(local_idx)
                if local_idx.size > 0:
                    tbl = tbl.take(pa.array(local_idx, type=pa.int64()))
                else:
                    continue

            n = tbl.num_rows
            if n == 0:
                continue

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_tbl = tbl.slice(start, end - start)
                X, y = self._table_to_numpy(batch_tbl)
                yield torch.from_numpy(X), torch.from_numpy(y)



class Model:
    def __init__(
        self, 
        data_x=None, 
        data_y=None, 
        model_name='catboost', 
        task='r',
        training_percent=0.8, 
        batch_size=32, 
        generator=None,
        validation_percentage=0.2,
        scaler_x_name=None,
        scaler_y_name=None,
        parquet_batches_data=None,
        output_metrics_dir="./results_monitor",
        max_gpus='all', 
        use_gpu: bool = False,
        single_parquet_path: str | None = None,
        reading_mode: str | None = 'dataframe',
        workers_data_loaders: int = 7,
        prefetch_factor: int = 2,
        streaming: bool = False,
        exclude_columns: list[str] | None = None,
        n_workers: int | str = 'auto',
        multi_gpu: bool = False,
        all_gpu: bool = True,
        tabpfn_version='v2',
        **kwargs
        ):
        """_summary_

        Args:
            data_x (_type_, optional): _description_. Defaults to None.
            data_y (_type_, optional): _description_. Defaults to None.
            model_name (str, optional): _description_. Defaults to 'catboost'.
            task (str, optional): _description_. Defaults to 'r'.
            training_percent (float, optional): _description_. Defaults to 0.8.
            epochs (int, optional): _description_. Defaults to 50.
            batch_size (int, optional): _description_. Defaults to 32.
            generator (_type_, optional): _description_. Defaults to None.
            validation_percentage (float, optional): _description_. Defaults to 0.2.
            scaler_x_name (_type_, optional): _description_. Defaults to None.
            scaler_y_name (_type_, optional): _description_. Defaults to None.
            gpu_ids=None, explicit list of GPU indices to use (e.g. [0,1,2])
            max_gpus=None, 
            parquet_batches_data:
            - Either a dict with any of the keys:
              {"train_loader": <iterable>, "val_loader": <iterable>, "test_loader": <iterable>}
            - Or a single iterable yielding (Xb, yb) batches to be used as train_loader.
            reading_mode: str | None = 'loaders'. can be 'loaders', 'dataframe', 'file_path' or 'input_folder'.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        
        
        
        self.n_workers = n_workers
        if self.n_workers == 'auto':
            self.n_workers = Model.get_optimal_dask_config()["n_workers"]
        
        self.multi_gpu = multi_gpu
        self.all_gpu = all_gpu
        self.prefetch_factor = prefetch_factor
        self.workers_data_loaders = workers_data_loaders
        self.use_gpu = use_gpu
        self._parquet_mode = parquet_batches_data is not None
        self.single_parquet_path = single_parquet_path
        self._single_parquet_mode = single_parquet_path is not None and not self._parquet_mode
        self.reading_mode = reading_mode
        self.output_metrics_dir = output_metrics_dir
        self.train_percent = training_percent
        self.__y_pred = None
        self.__batch_size = batch_size
        self.model_name  = model_name 
        self.__boosted_model = None
        self.__generator = generator
        self.history = None
        self.__task = task
        self.validation_percentage = validation_percentage
        self.max_gpus = max_gpus
        self._used_gpu_ids = None   # will be filled
        self.multi_gpu = False
        self.primary_gpu = 0
        self.x = data_x
        self.y = data_y
        self.__x_train = None
        self.__x_test = None
        self.__y_train = None
        self.__y_test = None
        
        if scaler_x_name == 'minmax':
            self.scaler_x = MinMaxScaler()
        elif scaler_x_name == 'standard':
            self.scaler_x = StandardScaler()
        elif scaler_x_name == 'robust':
            self.scaler_x = RobustScaler()
        elif scaler_x_name == 'quantile':
            self.scaler_x = QuantileTransformer()
        elif scaler_x_name == 'maxabs':
            self.scaler_x = MaxAbsScaler()
        elif scaler_x_name == None:
            self.scaler_x = None
        else:
            raise ValueError(f"Unsupported scaler: {scaler_x_name}")

        if scaler_y_name == 'minmax':
            self.scaler_y = MinMaxScaler()
        elif scaler_y_name == 'standard':
            self.scaler_y = StandardScaler()
        elif scaler_y_name == 'robust':
            self.scaler_y = RobustScaler()
        elif scaler_y_name == 'quantile':
            self.scaler_y = QuantileTransformer()
        elif scaler_y_name == 'maxabs':
            self.scaler_y = MaxAbsScaler()
        elif scaler_y_name == None:
            self.scaler_y = None
        else:
            raise ValueError(f"Unsupported scaler: {scaler_y_name}")


        if self.scaler_x is not None:
            self.scale_x()
            
        if self.scaler_y is not None:
            self.scale_y()
        
        # if not exist create the output directory
        if not os.path.exists(self.output_metrics_dir):
            os.makedirs(self.output_metrics_dir)

        if self.reading_mode == "loaders":
            if streaming:
                print("Streaming mode enabled.")
            elif isinstance(parquet_batches_data, dict):
                self.train_loader = parquet_batches_data.get("train_loader", None)
                self.val_loader   = parquet_batches_data.get("val_loader", None)
                self.test_loader  = parquet_batches_data.get("test_loader", None)
            elif self.train_loader is not None:
                # single iterable -> treat as train_loader
                self.train_loader = parquet_batches_data

            else:
                raise ValueError("parquet_batches_data provided but no 'train_loader' found or iterable given.")
        elif self.reading_mode == "dataframe":
            if training_percent != 1:
                self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(self.x, 
                                                                                            self.y,
                                                                                            train_size=training_percent,
                                                                                            test_size=1-training_percent)
                
        elif self.reading_mode == "file_path":
            if training_percent != 1:
                self._load_single_parquet_into_memory()
        elif self.reading_mode == "empty_model":
            # load empty model
            pass

        
        if model_name  == 'decision_tree':
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            
            if task == 'c':
                self.__model = DecisionTreeClassifier()
            else:
                self.__model = DecisionTreeRegressor()

        elif model_name  == 'svm':
            from sklearn.svm import SVC, SVR
            
            if task == 'c':
                self.__model = SVC()
            else:
                self.__model = SVR()
                
        elif model_name  == 'linear_regression':
            from sklearn.linear_model import LogisticRegression, LinearRegression
            
            if task == 'c':
                self.__model = LogisticRegression(random_state=2)
            else:
                self.__model = LinearRegression()

        elif model_name  == 'naive_bayes':
            from sklearn.naive_bayes import GaussianNB, MultinomialNB
            
            if task == 'c':
                self.__model = MultinomialNB()
            else:
                self.__model = GaussianNB()
        
        elif model_name  == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            if task == 'c':
                self.__model = RandomForestClassifier()
            else:
                self.__model = RandomForestRegressor()
                
        elif model_name  == 'adaboost':
            from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
            
            if task == 'c':
                self.__model = AdaBoostClassifier()
            else:
                self.__model = AdaBoostRegressor()
                
        elif model_name  == 'xgboost':
            from xgboost import XGBClassifier, XGBRegressor
            
            if task == 'c':
                self.__model = XGBClassifier(**kwargs)
            else:
                self.__model = XGBRegressor(**kwargs)
                
        elif model_name  == 'catboost':
            
            
            if task == 'c':
                self.__model = CatBoostClassifier(**kwargs)
            else:
                self.__model = CatBoostRegressor(**kwargs)

        elif model_name  == 'deep_learning':
            import tensorflow.keras.optimizers
            from tensorflow.keras import backend as K
            from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, MaxPool1D, Conv1D, Reshape, LSTM
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.losses import mse as tf_mse
            from tensorflow.keras.optimizers import Adam
            
            self.__model = Sequential()

        elif model_name  == 'knn':
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            
            if task == 'c':
                self.__model = KNeighborsClassifier(**kwargs)
            else:
                self.__model = KNeighborsRegressor(**kwargs)
                
        elif model_name  == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
            
            if task == 'c':
                self.__model = GradientBoostingClassifier()
            else:
                self.__model = GradientBoostingRegressor() 
            
        elif model_name  == 'tabformer':
            # Initialize TabFormer model
            if self.x is not None:
                # In-memory data: infer input dim from numpy/array
                in_dim = self.x.shape[1]
                self.__model = FTTransformer(num_features=in_dim)
            elif streaming and self.single_parquet_path is not None:
                # Direct parquet file streaming mode - extract dimensions from file schema
                import pyarrow.parquet as pq
                
                print(f"🔍 Extracting input dimension from Parquet schema for streaming mode")
                pf = pq.ParquetFile(self.single_parquet_path)
                schema = pf.schema_arrow
                col_names = list(schema.names)
                
                # Exclude non-feature columns
                drop_meta = {"date", "year", "month", "y"}
                feature_cols = [c for c in col_names if c not in drop_meta]
                if exclude_columns is not None:
                    feature_cols = [c for c in feature_cols if c not in exclude_columns]
                
                in_dim = len(feature_cols)
                # Store feature names for later use
                self.feature_names_ = feature_cols
                self.in_dim_ = in_dim
                self.__model = FTTransformer(num_features=in_dim)
                print(f"📊 Input dimension from Parquet schema: {in_dim} features")
            elif parquet_batches_data is not None:
                # Standard streaming mode: parquet_batches_data may be dict of loaders
                if isinstance(parquet_batches_data, dict) and parquet_batches_data.get("train_loader") is not None:
                    # Peek one batch to infer feature dimension
                    Xb, _ = next(iter(parquet_batches_data["train_loader"]))
                    in_dim = Xb.shape[1]
                else:
                    # If it's just a loader, peek directly
                    Xb, _ = next(iter(parquet_batches_data))
                    in_dim = Xb.shape[1]
            else:
                raise ValueError("TabFormer init requires either data_x, parquet_batches_data, or streaming=True with single_parquet_path to infer input dim.")
        
        
        elif model_name  == 'tabpfn':
            """
            TabPFN models (few‑shot prior transformer for tabular data).
            task='c' -> TabPFNClassifier (probabilities via predict_proba)
            task!='c' -> TabPFNRegressor
            """
            try:
                from tabpfn import TabPFNClassifier, TabPFNRegressor
                from tabpfn.constants import ModelVersion
            except ImportError:
                raise ImportError("tabpfn not installed. Run: pip install tabpfn")

            # Optional: user can pass tabpfn_version='v2' in kwargs
            tabpfn_version = kwargs.pop("tabpfn_version", None)
            if task == 'c':
                if tabpfn_version == 'v2':
                    self.__model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, **kwargs)
                else:
                    self.__model = TabPFNClassifier(**kwargs)  # default (2.5 weights)
            else:
                if tabpfn_version == 'v2':
                    self.__model = TabPFNRegressor.create_default_for_version(ModelVersion.V2, **kwargs)
                else:
                    self.__model = TabPFNRegressor(**kwargs)   # default (2.5 weights)
        
        else:
            self.__model = None

        
        self.set_device(use_gpu=self.use_gpu, multi_gpu=self.multi_gpu)


    def get_scaler(self):
        return self.scaler_x

    @staticmethod
    def get_optimal_dask_config(max_ram_per_worker_gb=2):
        """
        Auto-configure optimal Dask worker/thread settings based on available system resources.

        Args:
            max_ram_per_worker_gb (int): Approx. RAM (in GB) to assign per worker.

        Returns:
            dict: {'n_workers': int, 'threads_per_worker': int}
        """
        total_cores = os.cpu_count()
        total_ram_gb = psutil.virtual_memory().total / 1e9

        # Estimate number of workers based on RAM
        n_workers_by_ram = int(total_ram_gb // max_ram_per_worker_gb)
        n_workers = min(total_cores, max(1, n_workers_by_ram))

        # Use at least 1 thread per worker; divide evenly
        threads_per_worker = max(1, total_cores // n_workers)

        print(f"🔧 Optimal config: {n_workers} workers, {threads_per_worker} threads each")

        return {
            "n_workers": n_workers,
            "threads_per_worker": threads_per_worker
        }

    
    
    def set_device(self, use_gpu=False, multi_gpu=False):
        """Set base device (single GPU or CPU). Multi-GPU wrapping done later in configure_multi_gpu()."""
        if multi_gpu:
            self.configure_multi_gpu()
            return
        if use_gpu and torch.cuda.is_available():
            print("CUDA is available.")
            # pick a temporary base device (will adjust if multi-GPU requested)
            self.device = torch.device("cuda:0")
            print(self.gpu_info())
        else:
            print("CUDA is not available. Using CPU.")
            self.device = torch.device("cpu")
    
    def get_generator(self):
        return self.__generator
    
    def set_generator(self, generator):
        self.__generator = generator
        
    def get_model(self):
        return self.__model
    
    def set_model(self, model):
        self.__model = model
    
    def add_layer(self, connections_number=2, activation_function='relu', input_dim=None):
        """Add a dense layer to the model architecture

        Args:
            connections_number (int, optional): number of neurons to add. Defaults to 2.
            activation_function (str, optional): function to apply on sum of wi.xi. examples: ['linear', 'relu', 'softmax']. Defaults to 'relu'.
            input_dim (int, optional): number of features in X matrix. Defaults to None.
        """
        if input_dim:
            self.__model.add(Dense(connections_number, activation=activation_function, input_dim=input_dim))
        else:
            self.__model.add(Dense(connections_number, activation=activation_function))
            
    def add_lstm_layer(self, connections_number=2, activation_function='relu', input_shape=None, return_sequences=True):
        """Add a lstm layer

        Args:
            connections_number (int, optional): [description]. Defaults to 2.
            activation_function (str, optional): [description]. Defaults to 'relu'.
            input_shape ([type], optional): example: (weather_window,1). Defaults to None.
            return_sequences: This hyper parameter should be set to False for the last layer and true for the other previous layers.
        """
        if input_shape is not None:
            self.__model.add(LSTM(units=connections_number, activation=activation_function, input_shape=input_shape, return_sequences=return_sequences))
        else:
            self.__model.add(LSTM(units=connections_number, activation=activation_function, return_sequences=return_sequences))

    def add_conv_2d_layer(self, filter_nbr=1, filter_shape_tuple=(3,3), input_shape=None, activation_function='relu'):
        if input_shape:
            self.__model.add(Conv2D(filters=filter_nbr, kernel_size=filter_shape_tuple, input_shape=input_shape,
                                    activation=activation_function))
        else:
            self.__model.add(Conv2D(filters=filter_nbr, kernel_size=filter_shape_tuple,
                                    activation=activation_function))
            
    def add_conv_1d_layer(self, filter_nbr=1, filter_shape_int=3, input_shape=None, activation_function='relu', strides=10):
        if input_shape:
            #Input size should be (n_features, 1) == (data_x.shape[1], 1)
            self.__model.add(Conv1D(filters=filter_nbr, kernel_size=filter_shape_int, input_shape=input_shape,
                                    activation=activation_function))
        else:
            self.__model.add(Conv1D(filters=filter_nbr, kernel_size=filter_shape_int,
                                    activation=activation_function))

    def add_pooling_2d_layer(self, pool_size_tuple=(2, 2)):
        self.__model.add(MaxPooling2D(pool_size=pool_size_tuple))

    def add_pooling_1d_layer(self, pool_size_int=2):
        self.__model.add(MaxPool1D(pool_size=pool_size_int))

    def add_flatten_layer(self):
        self.__model.add(Flatten())
        
    def add_reshape_layer(self, input_dim):
        """
        for 1dcnn and 2dcnn use this layer as first layer 
        """
        self.__model.add(Reshape((input_dim, 1), input_shape=(input_dim, )))

    """def add_reshape_layer(self, target_shape=None, input_shape=None):
        self.__model.add(Reshape(target_shape=target_shape, input_shape=input_shape))"""

    def add_dropout_layer(self, rate_to_keep_output_value=0.2):
        """ dropout default initial value """
        self.__model.add(Dropout(rate_to_keep_output_value))

    def configure_multi_gpu(self):
        """
        Auto-configure GPUs using ONLY self.max_gpus.
        Rules:
          - No CUDA -> CPU.
          - max_gpus is None  -> use 1 GPU (cuda:0) if available.
          - max_gpus == 'all' or 0 -> use all available GPUs.
          - max_gpus is int N -> use min(N, total_available) GPUs [0..N-1].
          - Wrap with DataParallel if >1 GPU selected.
        """
        import torch
        if self.__model is None:
            return
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print("[GPU] CUDA not available -> using CPU.")
            return

        total = torch.cuda.device_count()
        if isinstance(self.max_gpus, str):
            if self.max_gpus.lower() == 'all':
                desired = total
            else:
                try:
                    desired = int(self.max_gpus)
                except ValueError:
                    desired = 1
        elif self.max_gpus in (0, None):
            # 0 or None -> interpret 0 as 'all', None as 'single'
            desired = total if self.max_gpus == 0 else 1
        else:
            desired = int(self.max_gpus)

        use_n = max(1, min(desired, total))
        selected = list(range(use_n))
        self._used_gpu_ids = selected
        self.primary_gpu = selected[0]
        self.device = torch.device(f"cuda:{self.primary_gpu}")

        if use_n > 1:
            print(f"[GPU] Using {use_n} GPUs: {selected}")
            self.__model = torch.nn.DataParallel(self.__model, device_ids=selected)
            self.multi_gpu = True
        else:
            print(f"[GPU] Using single GPU: {self.primary_gpu}")
            self.multi_gpu = False

        self.__model.to(self.device)
    
    
    def configure_multi_gpu_old(self):
        """
        Wrap model in DataParallel if multiple GPUs requested.
        Priority:
          1. Explicit gpu_ids list
          2. max_gpus (takes first N device indices)
          3. Fallback single GPU (cuda:0) or CPU
        """

        if self.__model is None:
            return  # nothing to wrap yet
        if not torch.cuda.is_available():
            return
        
        total = torch.cuda.device_count()
        if total <= 1:
            print(f"{total} GPU(s) available, but multi-GPU not requested or possible.")
            print("[GPU] Using single GPU: cuda:0")
            self._used_gpu_ids = [0]
            self.device = torch.device("cuda:0")
            self.__model.to(self.device)
            return

        # Resolve desired GPUs
        if self.gpu_ids is not None:
            selected = [g for g in self.gpu_ids if isinstance(g, int) and 0 <= g < total]
            if not selected:
                print(f"[GPU] Provided gpu_ids {self.gpu_ids} invalid; falling back to [0].")
                selected = [0]
        elif self.max_gpus is not None:
            try:
                mg = int(self.max_gpus)
                if mg < 1:
                    mg = 1
            except Exception:
                mg = 1
            selected = list(range(min(mg, total)))
        else:
            # default single GPU
            selected = [0]

        self._used_gpu_ids = selected
        self.primary_gpu = selected[0]
        self.device = torch.device(f"cuda:{self.primary_gpu}")

        if len(selected) > 1:
            print(f"[GPU] Using multiple GPUs: {selected}")
            self.__model = torch.nn.DataParallel(self.__model, device_ids=selected)
            self.multi_gpu = True
        else:
            self.multi_gpu = False

        self.__model.to(self.device)

    def gpu_info(self):
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "total_devices": torch.cuda.device_count(),
            "using": self._used_gpu_ids,
            "multi_gpu": self.multi_gpu,
            "device": str(self.device)
        }

    def train(self, n_epochs=50, metrics_list=['accuracy'], loss=None, optimizer=None, checkpoint_path='training_results'):
        """
        Metrics_list: ['r2'], ['accuracy'], ['mse', r2] or functions
        
        losses and metrics for regresion:
        tensorflow.keras.losses.mse
        r2_keras
        
        losses and metrics for classification:
        multi classes: tensorflow.keras.losses.categorical_crossentropy
        two classes: tensorflow.keras.losses.binary_crossentropy
                
        Optimizers:
        tensorflow.keras.optimizers.SGD(learning_rate=0.01)
        tensorflow.keras.optimizers.Adam(learning_rate=0.01)
        ...
        
        if you pass y as integers use loss='sparse_categorical_crossentropy'
        class Adadelta: Optimizer that implements the Adadelta algorithm.
        class Adagrad: Optimizer that implements the Adagrad algorithm.
        class Adam: Optimizer that implements the Adam algorithm.
        class Adamax: Optimizer that implements the Adamax algorithm.
        class Ftrl: Optimizer that implements the FTRL algorithm.
        class Nadam: Optimizer that implements the NAdam algorithm.
        class Optimizer: Base class for Keras optimizers.
        class RMSprop: Optimizer that implements the RMSprop algorithm.
        class SGD: Gradient descent (with momentum) optimizer.
        """
        
         # NEW: full-load single parquet path
        if getattr(self, "_single_parquet_mode", False) and not getattr(self, "_parquet_mode", False):
            # Delegate to dedicated method
            return self.train_single_parquet(n_epochs)
        
        
        # If parquet/streaming mode was provided, use streaming path and return.
        if getattr(self, "_parquet_mode", False):

            if self.model_name == 'tabformer':
                return self.train_streaming(
                    epochs=n_epochs,
                    output_metrics_dir=self.output_metrics_dir
                )

            return self.train_streaming_ml(
                epochs=n_epochs,   
            )
        
        if self.model_name  == 'dl':
            if loss is None:
                loss = tf_mse
            
            if optimizer is None:
                optimizer = Adam(learning_rate=0.0001)
            
            if 'r2' in metrics_list:
                metrics_list.remove('r2')
                metrics_list.append(self.r2_keras)
            self.__model.compile(loss=loss, optimizer=optimizer, metrics=metrics_list)
            if self.__generator is not None:
                self.history = self.__model.fit(self.get_generator(), epochs=self.__epochs, batch_size=self.__batch_size)
                print(self.history.history)
            else:
                self.history = self.__model.fit(self.x, self.y, epochs=self.__epochs,
                                        batch_size=self.__batch_size, validation_split=self.__validation_percentage)
                print(self.history.history)
                self.__y_pred = self.__model.predict(self.__x_test)
        
        elif self.model_name == 'tabformer':
            import numpy as np
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            import pandas as pd
            from tqdm.auto import tqdm

            
            self.__model.to(self.device)
            # Split ONLY the training set into train/val
            x_train, x_val, y_train, y_val = train_test_split(
                self.__x_train, self.__y_train,
                test_size=self.validation_percentage,
                random_state=42
            )

            

            x_train = self._to_float32_ndarray(x_train)
            x_val   = self._to_float32_ndarray(x_val)
            y_train = self._to_float32_1d(y_train)
            y_val   = self._to_float32_1d(y_val)

            # Tensors
            X_train_tensor = torch.as_tensor(x_train, dtype=torch.float32)
            y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1)
            X_val_tensor   = torch.as_tensor(x_val,   dtype=torch.float32)
            y_val_tensor   = torch.as_tensor(y_val,   dtype=torch.float32).unsqueeze(1)

            # Datasets & loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset   = TensorDataset(X_val_tensor,   y_val_tensor)

            train_loader = DataLoader(train_dataset, batch_size=self.__batch_size, shuffle=True)
            val_loader   = DataLoader(val_dataset,   batch_size=self.__batch_size, shuffle=False)

            optimizer = torch.optim.Adam(self.__model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            training_history = []

            # Epoch-level tqdm
            epoch_bar = tqdm(range(n_epochs), desc="Epochs", position=0)
            for epoch in epoch_bar:
                # ---- Train ----
                self.__model.train()
                batch_losses = []

                # Batch-level tqdm (training)
                train_batch_bar = tqdm(train_loader, desc=f"Train {epoch+1}/{n_epochs}", leave=False, position=1)
                running = 0.0
                total_train_batches = len(train_loader)
                for i, (batch_X, batch_y) in enumerate(train_batch_bar, 1):
                    
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.__model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    val_loss_item = loss.item()
                    batch_losses.append(val_loss_item)
                    running += val_loss_item
                    # Update batch bar postfix every few steps
                    if i % 10 == 0 or i == len(train_loader):
                        pct = 100.0 * i / total_train_batches
                        train_batch_bar.set_postfix(avg_batch_loss=running / i, progress=f"{pct:5.1f}%")

                epoch_train_mse_from_batches = float(np.mean(batch_losses))

                # ---- Evaluate (TRAIN & VAL) with batch-level tqdm ----
                self.__model.eval()
                with torch.no_grad():
                    # TRAIN predictions
                    train_preds_list, train_targets_list = [], []
                    eval_train_bar = tqdm(train_loader, desc="Eval-Train", leave=False, position=1)
                    total_eval_train = len(train_loader)
                    for j, (Xb, yb) in enumerate(eval_train_bar, 1):
                        Xb = Xb.to(self.device)
                        yb = yb.to(self.device)
                        pb = self.__model(Xb)
                        train_preds_list.append(pb.detach().cpu())
                        train_targets_list.append(yb.detach().cpu())
                        if j % 10 == 0 or j == total_eval_train:
                            eval_train_bar.set_postfix(progress=f"{100.0 * j / total_eval_train:5.1f}%")

                    y_train_pred = torch.cat(train_preds_list, dim=0).numpy().reshape(-1)
                    y_train_true = torch.cat(train_targets_list, dim=0).numpy().reshape(-1)

                    # VAL predictions
                    val_preds_list, val_targets_list = [], []
                    eval_val_bar = tqdm(val_loader, desc="Eval-Val", leave=False, position=1)
                    total_eval_val = len(val_loader)
                    for k, (Xb, yb) in enumerate(eval_val_bar, 1):
                        Xb = Xb.to(self.device)
                        yb = yb.to(self.device)
                        pb = self.__model(Xb)
                        val_preds_list.append(pb.detach().cpu())
                        val_targets_list.append(yb.detach().cpu())
                        if k % 10 == 0 or k == total_eval_val:
                            eval_val_bar.set_postfix(progress=f"{100.0 * k / total_eval_val:5.1f}%")

                    y_val_pred = torch.cat(val_preds_list, dim=0).numpy().reshape(-1)
                    y_val_true = torch.cat(val_targets_list, dim=0).numpy().reshape(-1)

                # ---- Metrics ----
                train_mse = mean_squared_error(y_train_true, y_train_pred)
                train_rmse = float(np.sqrt(train_mse))
                train_r2 = r2_score(y_train_true, y_train_pred)

                val_mse = mean_squared_error(y_val_true, y_val_pred)
                val_rmse = float(np.sqrt(val_mse))
                val_r2 = r2_score(y_val_true, y_val_pred)

                training_history.append({
                    "epoch": epoch + 1,
                    "batch_train_mse": epoch_train_mse_from_batches,
                    "train_rmse": train_rmse,
                    "train_r2": train_r2,
                    "val_rmse": val_rmse,
                    "val_r2": val_r2
                })

                # Update epoch-level bar with live metrics
                epoch_bar.set_postfix(
                    train_rmse=f"{train_rmse:.4f}",
                    train_r2=f"{train_r2:.4f}",
                    val_rmse=f"{val_rmse:.4f}",
                    val_r2=f"{val_r2:.4f}"
                )

            # Save history
            history_df = pd.DataFrame(training_history)
            history_df.to_csv("training_history.csv", index=False)
            print("Training history exported to training_history.csv")
            
            # ==== Evaluate on TEST set ====
            
            
            print("Evaluating on TEST set...")


            x_test_np = self._to_float32_ndarray(self.__x_test)
            y_test_np = self._to_float32_1d(self.__y_test)

            X_test_tensor = torch.as_tensor(x_test_np, dtype=torch.float32)
            y_test_tensor = torch.as_tensor(y_test_np, dtype=torch.float32).unsqueeze(1)


            

            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

            self.__model.eval()
            test_preds_list, test_targets_list = [], []

            from tqdm.auto import tqdm
            with torch.no_grad():
                test_bar = tqdm(test_loader, desc="Eval-Test", leave=False, position=1)
                total_test_batches = len(test_loader)
                for t, (Xb, yb) in enumerate(test_bar, 1):
                    Xb = Xb.to(self.device)
                    yb = yb.to(self.device)
                    pb = self.__model(Xb)
                    test_preds_list.append(pb.detach().cpu())
                    test_targets_list.append(yb.detach().cpu())
                    if t % 10 == 0 or t == total_test_batches:
                        test_bar.set_postfix(progress=f"{100.0 * t / total_test_batches:5.1f}%")

            y_test_pred = torch.cat(test_preds_list, dim=0).numpy().reshape(-1)
            y_test_true = torch.cat(test_targets_list, dim=0).numpy().reshape(-1)

            test_mse = mean_squared_error(y_test_true, y_test_pred)
            test_rmse = float(np.sqrt(test_mse))
            test_r2 = r2_score(y_test_true, y_test_pred)
            
            self.y_test = y_test_true
            self.y_test_pred = y_test_pred
            

            print(f"Performance on Test set -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")


        else:
            if self.train_percent == 1:
                self.__model.fit(self.x, self.y)
            else:
                self.__model.fit(self.__x_train, self.__y_train)
                self.__y_pred = self.__model.predict(self.__x_test)
        

    def train_on_single_parquet_file_streaming_v4(
        self,
        epochs: int = 50,
        data_loading_batch_size: int = 65536,  # I/O batch (rows read from disk)
        model_batch_size: int | None = None,   # Model mini-batch (forward/backward)
        target_col: str = "y",
        exclude_cols: list | None = None,
        log_every: int = 50,
        early_stopping_patience: int = 10,
        output_metrics_dir: str | None = None,
        checkpoint_path: str = "best_model",
        enable_tensorboard: bool = True,
        lr: float = 1e-3,
        show_io_progress: bool = False,  # kept for compatibility but ignored
    ):
        """
        Streaming training on a single Parquet file with separate loader/model batch sizes.
        - No IO progress bars
        - Single epoch tqdm bar shows loss and percentage
        - CSV columns: epoch,batch_train_mse,train_rmse,train_r2,val_rmse,val_r2
        """
        import os
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        from tqdm.auto import tqdm
        from sklearn.metrics import mean_squared_error, r2_score
        import pyarrow.parquet as pq
        import pyarrow as pa

        if not getattr(self, "single_parquet_path", None):
            raise ValueError("single_parquet_path not set. Required for streaming a single parquet file.")

        # Resolve batch sizes
        if model_batch_size is None or int(model_batch_size) <= 0:
            model_batch_size = int(getattr(self, "_Model__batch_size", 64) or 64)
        model_batch_size = int(model_batch_size)
        data_loading_batch_size = max(1, int(data_loading_batch_size))

        # Output dir
        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./results_monitor")
        os.makedirs(output_metrics_dir, exist_ok=True)

        # Robust train split param
        train_ratio = getattr(self, "train_percent", None)
        if train_ratio is None:
            train_ratio = getattr(self, "training_percent", 0.8)
        val_ratio = getattr(self, "validation_percentage", 0.2)

        # -------- ParquetLoader (no tqdm inside) --------
        class ParquetLoader:
            def __init__(
                self,
                parquet_file,
                feature_cols,
                target_col="y",
                batch_size=65536,
                indices=None,
                shuffle=True,
                seed=42,
            ):
                self.pf = pq.ParquetFile(parquet_file, memory_map=True)
                self.feature_cols = feature_cols
                self.target_col = target_col
                self.columns = feature_cols + [target_col]
                self.batch_size = int(batch_size)
                self.shuffle = shuffle
                self.seed = seed

                # Row-group metadata
                self.num_row_groups = self.pf.num_row_groups
                self.rg_offsets = []
                rg_sizes = []
                running = 0
                for rg in range(self.num_row_groups):
                    n = self.pf.metadata.row_group(rg).num_rows
                    self.rg_offsets.append((running, running + n))
                    rg_sizes.append(n)
                    running += n
                self.total_rows = running

                # Resolve selected indices per RG
                if indices is None:
                    self.selected_per_rg = {rg: None for rg in range(self.num_row_groups)}
                    per_rg_counts = rg_sizes[:]
                else:
                    idx = np.asarray(indices, dtype=np.int64)
                    idx.sort()
                    self.selected_per_rg = {}
                    per_rg_counts = []
                    for rg, (start, end) in enumerate(self.rg_offsets):
                        left = np.searchsorted(idx, start, side="left")
                        right = np.searchsorted(idx, end, side="left")
                        local = idx[left:right] - start
                        if local.size > 0:
                            self.selected_per_rg[rg] = local
                            per_rg_counts.append(int(local.size))
                        else:
                            self.selected_per_rg[rg] = np.empty((0,), dtype=np.int64)
                            per_rg_counts.append(0)

                self.per_rg_counts = per_rg_counts
                self.total_selected = int(sum(per_rg_counts))
                # Accurate total batches (sum ceil per RG)
                self.total_batches = int(sum((c + self.batch_size - 1) // self.batch_size for c in per_rg_counts if c > 0))
                if self.total_batches <= 0:
                    self.total_batches = 1

            def __len__(self):
                return self.total_batches

            def _table_to_numpy(self, table: pa.Table):
                # Arrow -> NumPy fast path (avoid full to_pandas when possible)
                try:
                    X_cols = [table[c].to_numpy(zero_copy_only=False) for c in self.feature_cols]
                    X = np.column_stack(X_cols).astype(np.float32, copy=False)
                    y = table[self.target_col].to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1, 1)
                    return X, y
                except Exception:
                    pdf = table.to_pandas()
                    X = pdf[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
                    y = pdf[[self.target_col]].to_numpy(dtype=np.float32, copy=False)
                    return X, y

            def __iter__(self):
                import random
                order = list(range(self.num_row_groups))
                if self.shuffle:
                    random.seed(self.seed)
                    random.shuffle(order)

                for rg in order:
                    local_idx = self.selected_per_rg.get(rg, None)
                    if local_idx is not None and isinstance(local_idx, np.ndarray) and local_idx.size == 0:
                        continue

                    tbl = self.pf.read_row_group(rg, columns=self.columns)

                    if local_idx is not None:
                        if self.shuffle and local_idx.size > 0:
                            rng = np.random.default_rng(self.seed + rg)
                            local_idx = local_idx.copy()
                            rng.shuffle(local_idx)
                        if local_idx.size == 0:
                            continue
                        tbl = tbl.take(pa.array(local_idx, type=pa.int64()))

                    n = tbl.num_rows
                    if n == 0:
                        continue

                    for start in range(0, n, self.batch_size):
                        end = min(start + self.batch_size, n)
                        batch_tbl = tbl.slice(start, end - start)
                        X, y = self._table_to_numpy(batch_tbl)
                        yield torch.from_numpy(X), torch.from_numpy(y)

        # -------- Discover columns from schema --------
        pf = pq.ParquetFile(self.single_parquet_path)
        schema = pf.schema_arrow
        col_names = list(schema.names)
        if target_col not in col_names:
            raise ValueError(f"Target column '{target_col}' not found in parquet file.")
        drop_meta = {"date", "year", "month"}
        feature_cols = [c for c in col_names if c not in {target_col, *drop_meta}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        self.feature_names_ = feature_cols

        total_rows = sum(pf.metadata.row_group(rg).num_rows for rg in range(pf.num_row_groups))
        print(f"Rows: {total_rows:,} | Features: {len(feature_cols)}")

        # -------- Split row indices --------
        from sklearn.model_selection import train_test_split
        all_indices = np.arange(total_rows)
        train_indices, remaining_indices = train_test_split(
            all_indices,
            train_size=train_ratio,
            random_state=42
        )
        if len(remaining_indices) > 0 and val_ratio > 0:
            denom = max(1e-12, (1 - train_ratio))
            val_size_relative = min(0.999, val_ratio / denom)
            val_indices, test_indices = train_test_split(
                remaining_indices,
                train_size=val_size_relative,
                random_state=42
            )
        else:
            val_indices = []
            test_indices = remaining_indices

        # -------- Build loaders --------
        train_loader = ParquetLoader(
            self.single_parquet_path,
            feature_cols,
            target_col=target_col,
            batch_size=data_loading_batch_size,
            indices=train_indices,
            shuffle=True,
        )
        val_loader = ParquetLoader(
            self.single_parquet_path,
            feature_cols,
            target_col=target_col,
            batch_size=data_loading_batch_size,
            indices=val_indices,
            shuffle=False,
        ) if len(val_indices) > 0 else None
        test_loader = ParquetLoader(
            self.single_parquet_path,
            feature_cols,
            target_col=target_col,
            batch_size=data_loading_batch_size,
            indices=test_indices,
            shuffle=False,
        ) if len(test_indices) > 0 else None

        # -------- TensorBoard --------
        writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tblog = os.path.join(output_metrics_dir, "tensorboard")
                os.makedirs(tblog, exist_ok=True)
                writer = SummaryWriter(log_dir=tblog)
            except Exception:
                writer = None

        history = []

        # -------- Train: TabFormer (PyTorch) --------
        if self.model_name == "tabformer":
            # Ensure model has correct input dim
            in_dim = len(feature_cols)
            needs_build = False
            if getattr(self, "_Model__model", None) is None:
                needs_build = True
            else:
                try:
                    cur_dim = self.__model.tokenizer.embedding.in_features
                    if cur_dim != in_dim:
                        needs_build = True
                except Exception:
                    needs_build = True
            if needs_build:
                self.__model = FTTransformer(num_features=in_dim)
            self.__model.to(self.device)

            optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            best_val_rmse = float("inf")
            patience_counter = 0
            best_model_state = None
            global_step = 0

            for epoch in range(epochs):
                # ---- TRAIN ----
                self.__model.train()
                running_loss_items = []  # per-mini-batch MSE
                total_batches = max(1, getattr(train_loader, "total_batches", None) or len(train_loader))
                epoch_bar = tqdm(enumerate(train_loader), total=total_batches,
                                 desc=f"Epoch {epoch+1}/{epochs} (Train)", leave=False)

                for bi, (X_batch, y_batch) in epoch_bar:
                    # epoch-level progress (batches)
                    progress_pct = min(100.0, 100.0 * (bi + 1) / total_batches)

                    # Process in model-sized mini-batches (update epoch bar per mini-batch)
                    num_samples = int(len(X_batch))
                    total_minis = max(1, (num_samples + model_batch_size - 1) // model_batch_size)

                    for m_idx, start in enumerate(range(0, num_samples, model_batch_size), 1):
                        end = min(start + model_batch_size, num_samples)
                        mini_X = X_batch[start:end].to(self.device, non_blocking=True)
                        mini_y = y_batch[start:end].to(self.device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)
                        out = self.__model(mini_X)
                        loss = criterion(out, mini_y)
                        loss.backward()
                        optimizer.step()

                        li = float(loss.detach().item())
                        running_loss_items.append(li)
                        global_step += 1

                        if writer and (global_step % log_every == 0):
                            writer.add_scalar("Batch/train_mse", li, global_step)

                        # mini-batch progress & display (clamped)
                        mb_pct = min(100.0, 100.0 * m_idx / total_minis)
                        avg_mse = float(np.mean(running_loss_items)) if running_loss_items else float("nan")
                        epoch_bar.set_postfix(
                            batch=f"{min(bi+1, total_batches)}/{total_batches}",
                            mb=f"{m_idx}/{total_minis}",
                            progress=f"{progress_pct:5.1f}%",
                            mb_progress=f"{mb_pct:5.1f}%",
                            loss=f"{avg_mse:.6f}",
                            mb_loss=f"{li:.6f}",
                        )

                # Epoch train batch MSE (mean over mini-batches)
                epoch_batch_mse = float(np.mean(running_loss_items)) if running_loss_items else float("nan")

                # ---- EVAL TRAIN METRICS (RMSE, R²) ----
                self.__model.eval()
                preds_tr, trues_tr = [], []
                total_train_eval = max(1, getattr(train_loader, "total_batches", None) or len(train_loader))
                with torch.no_grad():
                    for tbi, (Xb, Yb) in enumerate(train_loader):
                        ns = int(len(Xb))
                        for s in range(0, ns, model_batch_size):
                            e = min(s + model_batch_size, ns)
                            mini_X = Xb[s:e].to(self.device, non_blocking=True)
                            mini_y = Yb[s:e].to(self.device, non_blocking=True)
                            out = self.__model(mini_X)
                            preds_tr.append(out.detach().cpu().numpy())
                            trues_tr.append(mini_y.detach().cpu().numpy())
                if preds_tr:
                    y_pred_tr = np.vstack(preds_tr).reshape(-1)
                    y_true_tr = np.vstack(trues_tr).reshape(-1)
                    tr_mse = mean_squared_error(y_true_tr, y_pred_tr)
                    train_rmse = float(np.sqrt(tr_mse))
                    train_r2 = float(r2_score(y_true_tr, y_pred_tr))
                else:
                    train_rmse = train_r2 = None

                # ---- EVAL VAL (optional) ----
                val_rmse = val_r2 = None
                if val_loader is not None:
                    preds_v, trues_v = [], []
                    with torch.no_grad():
                        for Xb, Yb in val_loader:
                            ns = int(len(Xb))
                            for s in range(0, ns, model_batch_size):
                                e = min(s + model_batch_size, ns)
                                mini_X = Xb[s:e].to(self.device, non_blocking=True)
                                mini_y = Yb[s:e].to(self.device, non_blocking=True)
                                out = self.__model(mini_X)
                                preds_v.append(out.detach().cpu().numpy())
                                trues_v.append(mini_y.detach().cpu().numpy())
                    if preds_v:
                        y_pred_v = np.vstack(preds_v).reshape(-1)
                        y_true_v = np.vstack(trues_v).reshape(-1)
                        vmse = mean_squared_error(y_true_v, y_pred_v)
                        val_rmse = float(np.sqrt(vmse))
                        val_r2 = float(r2_score(y_true_v, y_pred_v))

                        # Early stopping
                        if val_rmse < best_val_rmse:
                            best_val_rmse = val_rmse
                            patience_counter = 0
                            best_model_state = {
                                "epoch": epoch,
                                "model_state_dict": (self.__model.module.state_dict()
                                                     if hasattr(self.__model, "module") else self.__model.state_dict()),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "val_rmse": val_rmse,
                                "val_r2": val_r2,
                            }
                            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                            torch.save(best_model_state, f"{checkpoint_path}.pth")
                        else:
                            patience_counter += 1
                            if patience_counter >= early_stopping_patience:
                                print(f"Early stopping at epoch {epoch+1} (no improvement {early_stopping_patience} epochs)")
                                break

                # Log & CSV row
                if writer:
                    writer.add_scalar("Epoch/batch_train_mse", float(epoch_batch_mse), epoch)
                    if train_rmse is not None:
                        writer.add_scalar("Epoch/train_rmse", train_rmse, epoch)
                        writer.add_scalar("Epoch/train_r2", train_r2, epoch)
                    if val_rmse is not None:
                        writer.add_scalar("Epoch/val_rmse", val_rmse, epoch)
                        writer.add_scalar("Epoch/val_r2", val_r2, epoch)

                history.append({
                    "epoch": epoch + 1,
                    "batch_train_mse": float(epoch_batch_mse),
                    "train_rmse": float(train_rmse) if train_rmse is not None else None,
                    "train_r2": float(train_r2) if train_r2 is not None else None,
                    "val_rmse": float(val_rmse) if val_rmse is not None else None,
                    "val_r2": float(val_r2) if val_r2 is not None else None,
                })

            # Load best model
            if best_model_state is not None:
                target = self.__model.module if hasattr(self.__model, "module") else self.__model
                target.load_state_dict(best_model_state["model_state_dict"])

            # ---- TEST (optional) ----
            if test_loader is not None:
                self.__model.eval()
                preds_te, trues_te = [], []
                with torch.no_grad():
                    for Xb, Yb in test_loader:
                        ns = int(len(Xb))
                        for s in range(0, ns, model_batch_size):
                            e = min(s + model_batch_size, ns)
                            mini_X = Xb[s:e].to(self.device, non_blocking=True)
                            mini_y = Yb[s:e].to(self.device, non_blocking=True)
                            out = self.__model(mini_X)
                            preds_te.append(out.detach().cpu().numpy())
                            trues_te.append(mini_y.detach().cpu().numpy())
                if preds_te:
                    y_pred_te = np.vstack(preds_te).reshape(-1)
                    y_true_te = np.vstack(trues_te).reshape(-1)
                    test_rmse = float(np.sqrt(mean_squared_error(y_true_te, y_pred_te)))
                    test_r2 = float(r2_score(y_true_te, y_pred_te))
                    if writer:
                        writer.add_scalar("Test/rmse", test_rmse, 0)
                        writer.add_scalar("Test/r2", test_r2, 0)
                    self.y_test = y_true_te
                    self.y_test_pred = y_pred_te

        # -------- Save CSV --------
        csv_path = os.path.join(output_metrics_dir, "streaming_training_history.csv")
        pd.DataFrame(history).to_csv(csv_path, index=False)

        if writer:
            writer.close()

        return history
    
    
    def train_on_single_parquet_file_fullram(
        self,
        epochs: int = 50,
        model_batch_size: int | None = None,   # only training batch size
        target_col: str = "y",
        exclude_cols: list[str] | None = None,
        log_every: int = 50,
        early_stopping_patience: int = 10,
        output_metrics_dir: str | None = None,
        checkpoint_path: str = "best_model",
        enable_tensorboard: bool = True,
        lr: float = 1e-3,
        shuffle: bool = True,
    ):
        """
        Full-RAM training on a single parquet file (no I/O batching).
        - Loads entire parquet into memory once.
        - Uses model_batch_size only for model training.
        - Early stopping on validation RMSE with checkpointing.
        """
        import os
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        from tqdm.auto import tqdm

        if not getattr(self, "single_parquet_path", None):
            raise ValueError("single_parquet_path not set. Required for full-RAM training.")

        if model_batch_size is None or int(model_batch_size) <= 0:
            model_batch_size = int(getattr(self, "_Model__batch_size", 64) or 64)
        model_batch_size = int(model_batch_size)

        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./results_monitor")
        os.makedirs(output_metrics_dir, exist_ok=True)

        # Load entire parquet into RAM (uses your existing helper)
        self._load_single_parquet_into_memory(target_col=target_col, exclude_cols=exclude_cols)
        # self.x, self.y set, plus __x_train/__x_test if train_percent!=1

        # Build TabFormer for correct input dim if needed
        if self.model_name == 'tabformer':
            in_dim = self.x.shape[1]
            if not isinstance(self.__model, FTTransformer) or getattr(self.__model, "head", None) is None:
                self.__model = FTTransformer(num_features=in_dim)
            else:
                # re-init if input dim changed
                try:
                    if self.__model.tokenizer.embedding.in_features != in_dim:
                        self.__model = FTTransformer(num_features=in_dim)
                except Exception:
                    self.__model = FTTransformer(num_features=in_dim)
            self.__model.to(self.device)

        # Split: train/val from training portion
        X_train_full = self.__x_train if self.__x_train is not None else self.x
        y_train_full = self.__y_train if self.__y_train is not None else self.y
        if self.validation_percentage and self.validation_percentage > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=self.validation_percentage, random_state=42, shuffle=True
            )
        else:
            X_train, y_train = X_train_full, y_train_full
            X_val = y_val = None

        # Datasets & loaders (batch_size used only here)
        Xtr_t = torch.as_tensor(X_train, dtype=torch.float32)
        ytr_t = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        train_loader = DataLoader(
            TensorDataset(Xtr_t, ytr_t),
            batch_size=model_batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.n_workers,
            persistent_workers= (int(self.n_workers) > 0),
            prefetch_factor= (getattr(self, "prefetch_factor", 2) if int(self.n_workers) > 0 else None),
            drop_last=False,
        )

        if X_val is not None:
            Xval_t = torch.as_tensor(X_val, dtype=torch.float32)
            yval_t = torch.as_tensor(y_val, dtype=torch.float32).reshape(-1, 1)
            val_loader = DataLoader(
                TensorDataset(Xval_t, yval_t),
                batch_size=model_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=int(getattr(self, "workers_data_loaders", 0)),
                persistent_workers=(int(getattr(self, "workers_data_loaders", 0)) > 0),
                prefetch_factor=(getattr(self, "prefetch_factor", 2) if int(getattr(self, "workers_data_loaders", 0)) > 0 else None),
                drop_last=False,
            )
        else:
            val_loader = None

        # Optional test set from earlier split
        test_loader = None
        if self.__x_test is not None and self.__x_test.size > 0:
            Xte_t = torch.as_tensor(self.__x_test, dtype=torch.float32)
            yte_t = torch.as_tensor(self.__y_test, dtype=torch.float32).reshape(-1, 1)
            test_loader = DataLoader(
                TensorDataset(Xte_t, yte_t),
                batch_size=model_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0
            )

        # TensorBoard
        writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tblog = os.path.join(output_metrics_dir, "tensorboard")
                os.makedirs(tblog, exist_ok=True)
                writer = SummaryWriter(log_dir=tblog)
                print(f"✅ TensorBoard -> {tblog}")
            except Exception as e:
                print(f"⚠️ TensorBoard disabled ({e})")

        history = []

        if self.model_name == 'tabformer':
            optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            best_val_rmse = float("inf")
            patience_counter = 0
            best_model_state = None
            global_step = 0

            for epoch in range(epochs):
                # Train
                self.__model.train()
                running_loss = 0.0
                seen = 0
                train_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                                 desc=f"🟢 Epoch {epoch+1}/{epochs} (Train)", leave=False)
                for bi, (bx, by) in train_bar:
                    bx = bx.to(self.device, non_blocking=True)
                    by = by.to(self.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    out = self.__model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()

                    li = loss.item()
                    running_loss += li * bx.size(0)
                    seen += bx.size(0)
                    global_step += 1

                    if (global_step % log_every) == 0 and writer:
                        writer.add_scalar("Batch/train_mse", li, global_step)

                    train_bar.set_postfix(loss=f"{(running_loss/max(1,seen)):.6f}")

                avg_train_loss = running_loss / max(1, seen)

                # Validation
                val_rmse = val_r2 = None
                if val_loader is not None:
                    self.__model.eval()
                    preds, trues = [], []
                    with torch.no_grad():
                        for bx, by in val_loader:
                            bx = bx.to(self.device, non_blocking=True)
                            by = by.to(self.device, non_blocking=True)
                            pb = self.__model(bx)
                            preds.append(pb.detach().cpu().numpy())
                            trues.append(by.detach().cpu().numpy())
                    if preds:
                        y_pred = np.vstack(preds).reshape(-1)
                        y_true = np.vstack(trues).reshape(-1)
                        mse = mean_squared_error(y_true, y_pred)
                        val_rmse = float(np.sqrt(mse))
                        val_r2 = float(r2_score(y_true, y_pred))

                        # Early stopping
                        if val_rmse < best_val_rmse:
                            best_val_rmse = val_rmse
                            patience_counter = 0
                            best_model_state = {
                                "epoch": epoch,
                                "model_state_dict": (self.__model.module.state_dict()
                                                     if hasattr(self.__model, "module") else self.__model.state_dict()),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "val_rmse": val_rmse,
                                "val_r2": val_r2,
                            }
                            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                            torch.save(best_model_state, f"{checkpoint_path}.pth")
                            print(f"✓ Saved best model at epoch {epoch+1} (val_rmse={val_rmse:.4f})")
                        else:
                            patience_counter += 1
                            if patience_counter >= early_stopping_patience:
                                print(f"⛔ Early stopping after epoch {epoch+1} (no improvement {early_stopping_patience} epochs)")
                                break

                # Log epoch
                if writer:
                    writer.add_scalar("Epoch/train_loss", float(avg_train_loss), epoch)
                    if val_rmse is not None:
                        writer.add_scalar("Epoch/val_rmse", float(val_rmse), epoch)
                        writer.add_scalar("Epoch/val_r2", float(val_r2), epoch)

                history.append({
                    "epoch": epoch + 1,
                    "train_loss": float(avg_train_loss),
                    "val_rmse": float(val_rmse) if val_rmse is not None else None,
                    "val_r2": float(val_r2) if val_r2 is not None else None,
                })

            # Load best
            if best_model_state is not None:
                target = self.__model.module if hasattr(self.__model, "module") else self.__model
                target.load_state_dict(best_model_state["model_state_dict"])
                print(f"ℹ️ Loaded best model (epoch {best_model_state['epoch']+1}, val_rmse={best_model_state['val_rmse']:.4f})")

            # Test
            if test_loader is not None:
                self.__model.eval()
                preds, trues = [], []
                with torch.no_grad():
                    for bx, by in test_loader:
                        bx = bx.to(self.device, non_blocking=True)
                        by = by.to(self.device, non_blocking=True)
                        pb = self.__model(bx)
                        preds.append(pb.detach().cpu().numpy())
                        trues.append(by.detach().cpu().numpy())
                if preds:
                    y_pred = np.vstack(preds).reshape(-1)
                    y_true = np.vstack(trues).reshape(-1)
                    test_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    test_r2 = float(r2_score(y_true, y_pred))
                    print(f"📊 Test -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
                    if writer:
                        writer.add_scalar("Test/rmse", test_rmse, 0)
                        writer.add_scalar("Test/r2", test_r2, 0)
                    self.y_test = y_true
                    self.y_test_pred = y_pred

        else:
            # Classic ML: fit on full train, evaluate on val/test
            history = []
            self.__model.fit(X_train, y_train)
            train_pred = self.__model.predict(X_train)
            train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
            train_r2 = float(r2_score(y_train, train_pred))
            val_rmse = val_r2 = None
            if X_val is not None:
                val_pred = self.__model.predict(X_val)
                val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
                val_r2 = float(r2_score(y_val, val_pred))
            history.append({
                "epoch": 1, "train_rmse": train_rmse,
                "val_rmse": val_rmse, "val_r2": val_r2
            })
            if test_loader is not None and self.__x_test is not None:
                te_pred = self.__model.predict(self.__x_test)
                test_rmse = float(np.sqrt(mean_squared_error(self.__y_test, te_pred)))
                test_r2 = float(r2_score(self.__y_test, te_pred))
                print(f"📊 Test -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

        # Save history
        csv_path = os.path.join(output_metrics_dir, "fullram_training_history.csv")
        pd.DataFrame(history).to_csv(csv_path, index=False)
        print(f"✅ History saved -> {csv_path}")

        if writer:
            writer.close()
            print(f"✅ TensorBoard logs saved to {os.path.join(output_metrics_dir, 'tensorboard')}")

        return history
    
    
    def train_on_single_parquet_file_streaming_v5(
        self,
        epochs: int = 50,
        data_loading_batch_size: int = 65536,  # I/O batch (rows read from disk)
        model_batch_size: int | None = None,   # Model mini-batch (forward/backward)
        target_col: str = "y",
        exclude_cols: list | None = None,
        log_every: int = 50,
        early_stopping_patience: int = 10,
        output_metrics_dir: str | None = None,
        checkpoint_path: str = "best_model",
        enable_tensorboard: bool = True,
        lr: float = 1e-3,
        show_io_progress: bool = True,
    ):
        """
        Streaming training on a single Parquet file with separate loader/model batch sizes.
        - Accurate tqdm percentages (clamped to [0, 100])
        - Row-level I/O progress
        - Mini-batch progress inside each loader batch
        - CSV columns: epoch,batch_train_mse,train_rmse,train_r2,val_rmse,val_r2
        """
        import os
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        from tqdm.auto import tqdm
        from sklearn.metrics import mean_squared_error, r2_score
        import pyarrow.parquet as pq
        import pyarrow as pa

        if not getattr(self, "single_parquet_path", None):
            raise ValueError("single_parquet_path not set. Required for streaming a single parquet file.")

        # Resolve batch sizes
        if model_batch_size is None or int(model_batch_size) <= 0:
            model_batch_size = int(getattr(self, "_Model__batch_size", 64) or 64)
        model_batch_size = int(model_batch_size)
        data_loading_batch_size = max(1, int(data_loading_batch_size))

        # Output dir
        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./results_monitor")
        os.makedirs(output_metrics_dir, exist_ok=True)

        # -------- ParquetLoader with accurate total_batches and IO progress --------
        class ParquetLoader:
            def __init__(
                self,
                parquet_file,
                feature_cols,
                target_col="y",
                batch_size=65536,
                indices=None,
                shuffle=True,
                seed=42,
                name="IO-Loader",
                show_progress=True,
                progress_position=2,
            ):
                self.pf = pq.ParquetFile(parquet_file, memory_map=True)
                self.feature_cols = feature_cols
                self.target_col = target_col
                self.columns = feature_cols + [target_col]
                self.batch_size = int(batch_size)
                self.shuffle = shuffle
                self.seed = seed

                # IO progress config
                self.io_name = name
                self.show_progress = show_progress
                self.progress_position = progress_position

                # Row-group metadata
                self.num_row_groups = self.pf.num_row_groups
                self.rg_offsets = []
                rg_sizes = []
                running = 0
                for rg in range(self.num_row_groups):
                    n = self.pf.metadata.row_group(rg).num_rows
                    self.rg_offsets.append((running, running + n))
                    rg_sizes.append(n)
                    running += n
                self.total_rows = running

                # Resolve selected indices per RG
                if indices is None:
                    self.selected_per_rg = {rg: None for rg in range(self.num_row_groups)}
                    per_rg_counts = rg_sizes[:]
                else:
                    idx = np.asarray(indices, dtype=np.int64)
                    idx.sort()
                    self.selected_per_rg = {}
                    per_rg_counts = []
                    for rg, (start, end) in enumerate(self.rg_offsets):
                        left = np.searchsorted(idx, start, side="left")
                        right = np.searchsorted(idx, end, side="left")
                        local = idx[left:right] - start
                        if local.size > 0:
                            self.selected_per_rg[rg] = local
                            per_rg_counts.append(int(local.size))
                        else:
                            self.selected_per_rg[rg] = np.empty((0,), dtype=np.int64)
                            per_rg_counts.append(0)

                self.per_rg_counts = per_rg_counts
                self.total_selected = int(sum(per_rg_counts))
                # Accurate total batches (sum ceil per RG)
                self.total_batches = int(sum((c + self.batch_size - 1) // self.batch_size for c in per_rg_counts if c > 0))
                if self.total_batches <= 0:
                    self.total_batches = 1

            def __len__(self):
                return self.total_batches

            def _table_to_numpy(self, table: pa.Table):
                # Arrow -> NumPy fast path (avoid full to_pandas when possible)
                try:
                    X_cols = [table[c].to_numpy(zero_copy_only=False) for c in self.feature_cols]
                    X = np.column_stack(X_cols).astype(np.float32, copy=False)
                    y = table[self.target_col].to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1, 1)
                    return X, y
                except Exception:
                    pdf = table.to_pandas()
                    X = pdf[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
                    y = pdf[[self.target_col]].to_numpy(dtype=np.float32, copy=False)
                    return X, y

            def __iter__(self):
                import random
                order = list(range(self.num_row_groups))
                if self.shuffle:
                    random.seed(self.seed)
                    random.shuffle(order)

                io_pbar = None
                if self.show_progress:
                    io_pbar = tqdm(
                        total=self.total_selected,
                        desc=self.io_name,
                        position=self.progress_position,
                        leave=False,
                        unit="rows",
                        dynamic_ncols=True,
                    )

                try:
                    for rg in order:
                        local_idx = self.selected_per_rg.get(rg, None)
                        if local_idx is not None and isinstance(local_idx, np.ndarray) and local_idx.size == 0:
                            continue

                        tbl = self.pf.read_row_group(rg, columns=self.columns)

                        if local_idx is not None:
                            if self.shuffle and local_idx.size > 0:
                                rng = np.random.default_rng(self.seed + rg)
                                local_idx = local_idx.copy()
                                rng.shuffle(local_idx)
                            if local_idx.size == 0:
                                continue
                            tbl = tbl.take(pa.array(local_idx, type=pa.int64()))

                        n = tbl.num_rows
                        if n == 0:
                            continue

                        for start in range(0, n, self.batch_size):
                            end = min(start + self.batch_size, n)
                            batch_tbl = tbl.slice(start, end - start)
                            X, y = self._table_to_numpy(batch_tbl)
                            if io_pbar is not None:
                                io_pbar.update(end - start)
                            yield torch.from_numpy(X), torch.from_numpy(y)
                finally:
                    if io_pbar is not None:
                        io_pbar.close()

        # -------- Discover columns from schema --------
        pf = pq.ParquetFile(self.single_parquet_path)
        schema = pf.schema_arrow
        col_names = list(schema.names)
        if target_col not in col_names:
            raise ValueError(f"Target column '{target_col}' not found in parquet file.")
        drop_meta = {"date", "year", "month"}
        feature_cols = [c for c in col_names if c not in {target_col, *drop_meta}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        self.feature_names_ = feature_cols

        total_rows = sum(pf.metadata.row_group(rg).num_rows for rg in range(pf.num_row_groups))
        print(f"🗂️ Rows: {total_rows:,} | 🧩 Features: {len(feature_cols)}")

        # -------- Split row indices --------
        from sklearn.model_selection import train_test_split
        all_indices = np.arange(total_rows)
        train_indices, remaining_indices = train_test_split(
            all_indices,
            train_size=self.train_percent,
            random_state=42
        )
        if len(remaining_indices) > 0 and self.validation_percentage > 0:
            # clamp to avoid 1.0000000000000002
            denom = max(1e-12, (1 - self.train_percent))
            val_size_relative = min(0.999, self.validation_percentage / denom)
            val_indices, test_indices = train_test_split(
                remaining_indices,
                train_size=val_size_relative,
                random_state=42
            )
        else:
            val_indices = []
            test_indices = remaining_indices

        print(f"🔀 Split -> train={len(train_indices):,}, val={len(val_indices):,}, test={len(test_indices):,}")

        # -------- Build loaders --------
        train_loader = ParquetLoader(
            self.single_parquet_path,
            feature_cols,
            target_col=target_col,
            batch_size=data_loading_batch_size,
            indices=train_indices,
            shuffle=True,
            name="🧰 IO-Train (rows)",
            show_progress=show_io_progress,
            progress_position=2
        )
        val_loader = ParquetLoader(
            self.single_parquet_path,
            feature_cols,
            target_col=target_col,
            batch_size=data_loading_batch_size,
            indices=val_indices,
            shuffle=False,
            name="🧪 IO-Val (rows)",
            show_progress=show_io_progress,
            progress_position=2
        ) if len(val_indices) > 0 else None
        test_loader = ParquetLoader(
            self.single_parquet_path,
            feature_cols,
            target_col=target_col,
            batch_size=data_loading_batch_size,
            indices=test_indices,
            shuffle=False,
            name="🧷 IO-Test (rows)",
            show_progress=show_io_progress,
            progress_position=2
        ) if len(test_indices) > 0 else None

        # -------- TensorBoard --------
        writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tblog = os.path.join(output_metrics_dir, "tensorboard")
                os.makedirs(tblog, exist_ok=True)
                writer = SummaryWriter(log_dir=tblog)
                print(f"✅ TensorBoard -> {tblog}")
            except Exception as e:
                print(f"⚠️ TensorBoard disabled ({e})")

        history = []

        # -------- Train: TabFormer (PyTorch) --------
        if self.model_name == "tabformer":
            # Ensure model has correct input dim
            in_dim = len(feature_cols)
            needs_build = False
            if getattr(self, "_Model__model", None) is None:
                needs_build = True
            else:
                try:
                    cur_dim = self.__model.tokenizer.embedding.in_features
                    if cur_dim != in_dim:
                        needs_build = True
                except Exception:
                    needs_build = True
            if needs_build:
                self.__model = FTTransformer(num_features=in_dim)
            self.__model.to(self.device)

            optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            best_val_rmse = float("inf")
            patience_counter = 0
            best_model_state = None
            global_step = 0

            for epoch in range(epochs):
                # ---- TRAIN ----
                self.__model.train()
                running_loss_items = []  # store per-mini-batch loss.item() (MSE) for epoch avg
                total_batches = max(1, getattr(train_loader, "total_batches", None) or len(train_loader))
                train_bar = tqdm(enumerate(train_loader), total=total_batches,
                                 desc=f"🟢 Epoch {epoch+1}/{epochs} (Train)", leave=False)
                for bi, (X_batch, y_batch) in train_bar:
                    progress_pct = min(100.0, 100.0 * (bi + 1) / total_batches)

                    num_samples = int(len(X_batch))
                    num_mini_batches = max(1, (num_samples + model_batch_size - 1) // model_batch_size)
                    mini_bar = tqdm(range(0, num_samples, model_batch_size),
                                    total=num_mini_batches,
                                    desc=f"   🔹 Mini ({min(bi+1, total_batches)}/{total_batches})",
                                    leave=False)
                    for start in mini_bar:
                        end = min(start + model_batch_size, num_samples)
                        disp_start = min(start + 1, num_samples)
                        disp_end = min(end, num_samples)

                        mini_X = X_batch[start:end].to(self.device, non_blocking=True)
                        mini_y = y_batch[start:end].to(self.device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)
                        out = self.__model(mini_X)
                        loss = criterion(out, mini_y)
                        loss.backward()
                        optimizer.step()

                        li = float(loss.detach().item())
                        running_loss_items.append(li)
                        global_step += 1

                        mini_bar.set_postfix(
                            loss=f"{li:.6f}",
                            samples=f"{disp_start}-{disp_end}/{num_samples}"
                        )
                        if writer and (global_step % log_every == 0):
                            writer.add_scalar("Batch/train_mse", li, global_step)

                    train_bar.set_postfix(progress=f"{progress_pct:5.1f}%")

                # Epoch train batch MSE (mean over mini-batches)
                epoch_batch_mse = float(np.mean(running_loss_items)) if running_loss_items else float("nan")

                # ---- EVAL TRAIN METRICS (RMSE, R²) ----
                self.__model.eval()
                preds_tr, trues_tr = [], []
                total_train_eval = max(1, getattr(train_loader, "total_batches", None) or len(train_loader))
                eval_tr_bar = tqdm(enumerate(train_loader), total=total_train_eval,
                                   desc=f"🧮 Epoch {epoch+1}/{epochs} (Eval-Train)", leave=False)
                with torch.no_grad():
                    for tbi, (Xb, yb) in eval_tr_bar:
                        ns = int(len(Xb))
                        nmb = max(1, (ns + model_batch_size - 1) // model_batch_size)
                        mini_bar = tqdm(range(0, ns, model_batch_size), total=nmb,
                                        desc=f"   🔹 Train-eval mini ({min(tbi+1, total_train_eval)}/{total_train_eval})",
                                        leave=False)
                        for s in mini_bar:
                            e = min(s + model_batch_size, ns)
                            mini_X = Xb[s:e].to(self.device, non_blocking=True)
                            mini_y = yb[s:e].to(self.device, non_blocking=True)
                            out = self.__model(mini_X)
                            preds_tr.append(out.detach().cpu().numpy())
                            trues_tr.append(mini_y.detach().cpu().numpy())
                if preds_tr:
                    y_pred_tr = np.vstack(preds_tr).reshape(-1)
                    y_true_tr = np.vstack(trues_tr).reshape(-1)
                    tr_mse = mean_squared_error(y_true_tr, y_pred_tr)
                    train_rmse = float(np.sqrt(tr_mse))
                    train_r2 = float(r2_score(y_true_tr, y_pred_tr))
                else:
                    train_rmse, train_r2 = None, None

                # ---- EVAL VAL (optional) ----
                val_rmse = None
                val_r2 = None
                if val_loader is not None:
                    preds_v, trues_v = [], []
                    total_val = max(1, getattr(val_loader, "total_batches", None) or len(val_loader))
                    val_bar = tqdm(enumerate(val_loader), total=total_val,
                                   desc=f"🧪 Epoch {epoch+1}/{epochs} (Val)", leave=False)
                    with torch.no_grad():
                        for vbi, (Xb, yb) in val_bar:
                            progress_pct = min(100.0, 100.0 * (vbi + 1) / total_val)
                            ns = int(len(Xb))
                            nmb = max(1, (ns + model_batch_size - 1) // model_batch_size)
                            mini_bar = tqdm(range(0, ns, model_batch_size), total=nmb,
                                            desc=f"   🔹 Val mini ({min(vbi+1, total_val)}/{total_val})",
                                            leave=False)
                            for s in mini_bar:
                                e = min(s + model_batch_size, ns)
                                mini_X = Xb[s:e].to(self.device, non_blocking=True)
                                mini_y = yb[s:e].to(self.device, non_blocking=True)
                                out = self.__model(mini_X)
                                preds_v.append(out.detach().cpu().numpy())
                                trues_v.append(mini_y.detach().cpu().numpy())
                            val_bar.set_postfix(progress=f"{progress_pct:5.1f}%")
                    if preds_v:
                        y_pred_v = np.vstack(preds_v).reshape(-1)
                        y_true_v = np.vstack(trues_v).reshape(-1)
                        vmse = mean_squared_error(y_true_v, y_pred_v)
                        val_rmse = float(np.sqrt(vmse))
                        val_r2 = float(r2_score(y_true_v, y_pred_v))

                        # Early stopping
                        if val_rmse < best_val_rmse:
                            best_val_rmse = val_rmse
                            patience_counter = 0
                            best_model_state = {
                                "epoch": epoch,
                                "model_state_dict": (self.__model.module.state_dict()
                                                     if hasattr(self.__model, "module") else self.__model.state_dict()),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "val_rmse": val_rmse,
                                "val_r2": val_r2,
                            }
                            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                            torch.save(best_model_state, f"{checkpoint_path}.pth")
                            print(f"✅ Saved best model at epoch {epoch+1} • 🧪 Val RMSE={val_rmse:.4f} • 📈 Val R²={val_r2:.4f}")
                        else:
                            patience_counter += 1
                            if patience_counter >= early_stopping_patience:
                                print(f"⛔ Early stopping after epoch {epoch+1} (no improvement {early_stopping_patience} epochs)")
                                break

                # Log & CSV row
                if writer:
                    writer.add_scalar("Epoch/batch_train_mse", float(epoch_batch_mse), epoch)
                    if train_rmse is not None:
                        writer.add_scalar("Epoch/train_rmse", train_rmse, epoch)
                        writer.add_scalar("Epoch/train_r2", train_r2, epoch)
                    if val_rmse is not None:
                        writer.add_scalar("Epoch/val_rmse", val_rmse, epoch)
                        writer.add_scalar("Epoch/val_r2", val_r2, epoch)

                history.append({
                    "epoch": epoch + 1,
                    "batch_train_mse": float(epoch_batch_mse),
                    "train_rmse": float(train_rmse) if train_rmse is not None else None,
                    "train_r2": float(train_r2) if train_r2 is not None else None,
                    "val_rmse": float(val_rmse) if val_rmse is not None else None,
                    "val_r2": float(val_r2) if val_r2 is not None else None,
                })

                # Colorful line
                msg = f"🟢 Epoch {epoch+1}/{epochs} • 🧮 batch_mse={epoch_batch_mse:.6f}"
                if train_rmse is not None and train_r2 is not None:
                    msg += f" • 🎯 train RMSE={train_rmse:.4f} R²={train_r2:.4f}"
                if val_rmse is not None and val_r2 is not None:
                    msg += f" • 🧪 val RMSE={val_rmse:.4f} R²={val_r2:.4f}"
                print(msg)

            # Load best model
            if best_model_state is not None:
                target = self.__model.module if hasattr(self.__model, "module") else self.__model
                target.load_state_dict(best_model_state["model_state_dict"])
                print(f"ℹ️ Loaded best model (epoch {best_model_state['epoch']+1}, val_rmse={best_model_state['val_rmse']:.4f})")

            # ---- TEST (optional) ----
            if test_loader is not None:
                self.__model.eval()
                preds_te, trues_te = [], []
                total_test = max(1, getattr(test_loader, "total_batches", None) or len(test_loader))
                test_bar = tqdm(enumerate(test_loader), total=total_test, desc="🧷 Testing", leave=False)
                with torch.no_grad():
                    for tbi, (Xb, yb) in test_bar:
                        progress_pct = min(100.0, 100.0 * (tbi + 1) / total_test)
                        ns = int(len(Xb))
                        nmb = max(1, (ns + model_batch_size - 1) // model_batch_size)
                        mini_bar = tqdm(range(0, ns, model_batch_size), total=nmb,
                                        desc=f"   🔹 Test mini ({min(tbi+1, total_test)}/{total_test})",
                                        leave=False)
                        for s in mini_bar:
                            e = min(s + model_batch_size, ns)
                            mini_X = Xb[s:e].to(self.device, non_blocking=True)
                            mini_y = yb[s:e].to(self.device, non_blocking=True)
                            out = self.__model(mini_X)
                            preds_te.append(out.detach().cpu().numpy())
                            trues_te.append(mini_y.detach().cpu().numpy())
                        test_bar.set_postfix(progress=f"{progress_pct:5.1f}%")
                if preds_te:
                    y_pred_te = np.vstack(preds_te).reshape(-1)
                    y_true_te = np.vstack(trues_te).reshape(-1)
                    test_rmse = float(np.sqrt(mean_squared_error(y_true_te, y_pred_te)))
                    test_r2 = float(r2_score(y_true_te, y_pred_te))
                    print(f"📊 Test -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
                    if writer:
                        writer.add_scalar("Test/rmse", test_rmse, 0)
                        writer.add_scalar("Test/r2", test_r2, 0)
                    self.y_test = y_true_te
                    self.y_test_pred = y_pred_te

        # -------- Train: Classic ML (CatBoost/XGBoost/RF/DT/GB) --------
        elif self.model_name in ["catboost", "xgboost", "random_forest", "decision_tree", "gradient_boosting"]:
            # One-pass fit over streaming batches with metrics
            batch_mse_list = []
            total_batches = max(1, getattr(train_loader, "total_batches", None) or len(train_loader))
            train_bar = tqdm(enumerate(train_loader), total=total_batches, desc="🟢 Train 1/1 (ML)", leave=False)
            for bi, (Xb, yb) in train_bar:
                progress_pct = min(100.0, 100.0 * (bi + 1) / total_batches)
                X_np = Xb.numpy()
                y_np = yb.numpy().reshape(-1)
                # Fit/update on this batch
                if self.model_name == "catboost":
                    from catboost import Pool
                    self.__model.set_params(verbose=False)
                    self.__model.fit(Pool(X_np, y_np), verbose=False)
                else:
                    self.__model.fit(X_np, y_np)
                # batch MSE on this batch
                pred_np = self.__model.predict(X_np)
                mse = mean_squared_error(y_np, pred_np)
                batch_mse_list.append(mse)
                train_bar.set_postfix(progress=f"{progress_pct:5.1f}%", mse=f"{mse:.6f}")

            epoch_batch_mse = float(np.mean(batch_mse_list)) if batch_mse_list else float("nan")

            # Train metrics over full train loader
            preds_tr, trues_tr = [], []
            total_train_eval = max(1, getattr(train_loader, "total_batches", None) or len(train_loader))
            eval_tr_bar = tqdm(enumerate(train_loader), total=total_train_eval, desc="🧮 Eval-Train (ML)", leave=False)
            for tbi, (Xb, yb) in eval_tr_bar:
                X_np = Xb.numpy(); y_np = yb.numpy().reshape(-1)
                preds_tr.append(self.__model.predict(X_np))
                trues_tr.append(y_np)
            if preds_tr:
                y_pred_tr = np.concatenate(preds_tr, axis=0)
                y_true_tr = np.concatenate(trues_tr, axis=0)
                train_rmse = float(np.sqrt(mean_squared_error(y_true_tr, y_pred_tr)))
                train_r2 = float(r2_score(y_true_tr, y_pred_tr))
            else:
                train_rmse = train_r2 = None

            # Validation metrics (optional)
            val_rmse = val_r2 = None
            if val_loader is not None:
                preds_v, trues_v = [], []
                total_val = max(1, getattr(val_loader, "total_batches", None) or len(val_loader))
                val_bar = tqdm(enumerate(val_loader), total=total_val, desc="🧪 Eval-Val (ML)", leave=False)
                for vbi, (Xb, yb) in val_bar:
                    progress_pct = min(100.0, 100.0 * (vbi + 1) / total_val)
                    X_np = Xb.numpy(); y_np = yb.numpy().reshape(-1)
                    preds_v.append(self.__model.predict(X_np))
                    trues_v.append(y_np)
                    val_bar.set_postfix(progress=f"{progress_pct:5.1f}%")
                if preds_v:
                    y_pred_v = np.concatenate(preds_v, axis=0)
                    y_true_v = np.concatenate(trues_v, axis=0)
                    val_rmse = float(np.sqrt(mean_squared_error(y_true_v, y_pred_v)))
                    val_r2 = float(r2_score(y_true_v, y_pred_v))

            history.append({
                "epoch": 1,
                "batch_train_mse": float(epoch_batch_mse),
                "train_rmse": float(train_rmse) if train_rmse is not None else None,
                "train_r2": float(train_r2) if train_r2 is not None else None,
                "val_rmse": float(val_rmse) if val_rmse is not None else None,
                "val_r2": float(val_r2) if val_r2 is not None else None,
            })
            msg = f"🟢 Epoch 1/1 • 🧮 batch_mse={epoch_batch_mse:.6f}"
            if train_rmse is not None and train_r2 is not None:
                msg += f" • 🎯 train RMSE={train_rmse:.4f} R²={train_r2:.4f}"
            if val_rmse is not None and val_r2 is not None:
                msg += f" • 🧪 val RMSE={val_rmse:.4f} R²={val_r2:.4f}"
            print(msg)

            # Optional: Test
            if test_loader is not None:
                preds_te, trues_te = [], []
                total_test = max(1, getattr(test_loader, "total_batches", None) or len(test_loader))
                test_bar = tqdm(enumerate(test_loader), total=total_test, desc="🧷 Testing (ML)", leave=False)
                for tbi, (Xb, yb) in test_bar:
                    progress_pct = min(100.0, 100.0 * (tbi + 1) / total_test)
                    X_np = Xb.numpy(); y_np = yb.numpy().reshape(-1)
                    preds_te.append(self.__model.predict(X_np))
                    trues_te.append(y_np)
                    test_bar.set_postfix(progress=f"{progress_pct:5.1f}%")
                if preds_te:
                    y_pred_te = np.concatenate(preds_te, axis=0)
                    y_true_te = np.concatenate(trues_te, axis=0)
                    test_rmse = float(np.sqrt(mean_squared_error(y_true_te, y_pred_te)))
                    test_r2 = float(r2_score(y_true_te, y_pred_te))
                    print(f"📊 Test -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

        else:
            raise ValueError(f"Model '{self.model_name}' not supported for single parquet streaming.")

        # -------- Save CSV --------
        csv_path = os.path.join(output_metrics_dir, "streaming_training_history.csv")
        pd.DataFrame(history).to_csv(csv_path, index=False)
        print(f"✅ History saved -> {csv_path}")

        if writer:
            writer.close()
            print(f"✅ TensorBoard logs saved to {os.path.join(output_metrics_dir, 'tensorboard')}")

        return history
    
    
    
    def train_on_single_parquet_file_streaming_catboost(
        self,
        *,
        target_col: str = "y",
        exclude_cols: list[str] | None = None,
        train_ratio: float | None = None,           # defaults to self.train_percent
        val_ratio: float | None = None,             # defaults to self.validation_percentage
        iterations: int = 2000,
        depth: int = 8,
        lr: float = 0.03,
        loss_function: str | None = None,           # e.g. 'RMSE' for regression
        early_stopping_rounds: int = 100,
        output_metrics_dir: str | None = None,
        show_progress: bool = True,
    ):
        """
        Train CatBoost on a single Parquet file with tqdm progress during fit.
        """
        import os
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
        from tqdm.auto import tqdm

        if self.model_name not in ("catboost",):
            raise ValueError("This method is only for CatBoost models.")
        if not getattr(self, "single_parquet_path", None):
            raise ValueError("single_parquet_path not set.")

        # Resolve output directory
        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./results_monitor")
        os.makedirs(output_metrics_dir, exist_ok=True)

        # Infer schema/feature columns
        pf = pq.ParquetFile(self.single_parquet_path)
        schema = pf.schema_arrow
        names = list(schema.names)
        if target_col not in names:
            raise ValueError(f"Target column '{target_col}' not found in parquet.")

        meta_drop = {"date", "year", "month"}
        feature_cols = [c for c in names if c not in {target_col, *meta_drop}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        self.feature_names_ = feature_cols

        # Split indices (train/val/test) to match TabPFN subsampling behavior
        total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
        from sklearn.model_selection import train_test_split

        tr_ratio = float(train_ratio) if train_ratio is not None else float(getattr(self, "train_percent", 0.8))

        # If val_ratio is None → disable validation; else use provided value directly on the train portion
        use_val = (val_ratio is not None) and (float(val_ratio) > 0)
        if use_val:
            vl_ratio = float(val_ratio)
            print(f"🔀 Splitting data with train_ratio={tr_ratio}, val_ratio={vl_ratio}")
        else:
            print(f"🔀 Splitting data with train_ratio={tr_ratio}, val_ratio=None")

        all_idx = np.arange(total_rows, dtype=np.int64)

        # First split: train+val vs test
        trainval_idx, test_idx = train_test_split(
            all_idx,
            train_size=tr_ratio,
            random_state=42,
            shuffle=True,
        )

        # Second split: only if validation explicitly requested (taken from the train portion)
        if use_val:
            train_idx, val_idx = train_test_split(
                trainval_idx,
                test_size=vl_ratio,
                random_state=42,
                shuffle=True,
            )
        else:
            train_idx = trainval_idx
            val_idx = np.array([], dtype=np.int64)

        print(
            f"🔄 Split -> train: {len(train_idx):,}, "
            f"val: {len(val_idx):,}, test: {len(test_idx):,}"
        )
        
        print("🗂️  Parquet rows: {:,} | 🧩 Features: {}".format(total_rows, len(feature_cols)))
        print(f"Features: {feature_cols}")

        print(f"Split: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")

        # Helper: materialize (X,y) by iterating row-groups and taking needed rows
        def _gather_by_indices(indices: np.ndarray, desc: str):
            import pyarrow as pa
            feat_parts, tgt_parts = [], []
            if indices.size == 0:
                return None, None
            indices_sorted = np.sort(indices)
            # Precompute RG offsets to map global->local
            rg_bounds = []
            acc = 0
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                rg_bounds.append((acc, acc + n))
                acc += n
            bar = tqdm(range(pf.num_row_groups), desc=desc, leave=False, disable=not show_progress)
            for rg in bar:
                start, end = rg_bounds[rg]
                left = np.searchsorted(indices_sorted, start, side="left")
                right = np.searchsorted(indices_sorted, end, side="left")
                local = indices_sorted[left:right] - start
                if local.size == 0:
                    continue
                tbl = pf.read_row_group(rg, columns=feature_cols + [target_col])
                tbl_sel = tbl.take(pa.array(local, type=pa.int64()))
                try:
                    X_cols = [tbl_sel[c].to_numpy(zero_copy_only=False) for c in feature_cols]
                    X = np.column_stack(X_cols).astype(np.float32, copy=False)
                    y = tbl_sel[target_col].to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1)
                except Exception:
                    pdf = tbl_sel.to_pandas()
                    X = pdf[feature_cols].to_numpy(dtype=np.float32, copy=False)
                    y = pdf[target_col].to_numpy(dtype=np.float32, copy=False).reshape(-1)
                feat_parts.append(X)
                tgt_parts.append(y)
            if not feat_parts:
                return None, None
            X_all = np.vstack(feat_parts)
            y_all = np.concatenate(tgt_parts)
            return X_all, y_all

        # Gather train/val/test
        X_train, y_train = _gather_by_indices(train_idx, "Load train")
        if X_train is None:
            raise RuntimeError("No training rows collected.")
        X_val, y_val = _gather_by_indices(val_idx, "Load val") if val_idx.size > 0 else (None, None)
        X_test, y_test = _gather_by_indices(test_idx, "Load test") if test_idx.size > 0 else (None, None)

        # Build CatBoost model (GPU if requested)
        from catboost import CatBoostRegressor, CatBoostClassifier, Pool
        is_regression = (self.__task != 'c')

        if loss_function is None:
            loss_function = "RMSE" if is_regression else "Logloss"

        # Decide GPU usage for CatBoost
        gpu_params = {}
        if self.use_gpu == True:
            try:
                import torch
                has_cuda = torch.cuda.is_available()
                print(f"CatBoost GPU requested, torch reports CUDA available: {has_cuda}")
            except Exception:
                has_cuda = True  # assume available if torch missing
                torch = None
            if has_cuda:
                gpu_params["task_type"] = "GPU"
                # Resolve device list
                devices_str = None

                # If user requests all GPUs via 'all_gpu' flag or max_gpus='all'
                if bool(getattr(self, "all_gpu", False)) or str(getattr(self, "max_gpus", "")).lower() == "all":
                    try:
                        total = torch.cuda.device_count() if torch is not None else 0
                        if total and total > 0:
                            devices_str = ",".join(str(i) for i in range(total))
                    except Exception:
                        pass

                # Else use explicitly selected GPU ids (from multi-GPU config)
                if devices_str is None and getattr(self, "_used_gpu_ids", None):
                    devices_str = ",".join(str(i) for i in self._used_gpu_ids)

                # Fallback to single GPU 0
                if devices_str is None:
                    devices_str = "0"

                gpu_params["devices"] = devices_str
                print(f"Using GPU ids: {devices_str}")

        if is_regression:
            if not isinstance(self.__model, CatBoostRegressor):
                self.__model = CatBoostRegressor(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=lr,
                    loss_function=loss_function,      # "RMSE" for regression
                    eval_metric="RMSE",
                    custom_metric=["R2"],             # compute R2 each iteration without staged_predict
                    random_seed=42,
                    verbose=1,                      # print every 100 iters (adjust)
                    metric_period=1,
                    use_best_model=True,
                    #bootstrap_type="No",        # better for small data
                    bootstrap_type='Bernoulli',   # But set sampling frequency to 1.0
                    subsample=1.0,
                    
                    **gpu_params
                )
            else:
                self.__model.set_params(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=lr,
                    loss_function=loss_function,
                    eval_metric="RMSE",
                    custom_metric=["R2"],
                    random_seed=42,
                    verbose=1,
                    metric_period=1,
                    use_best_model=True,
                    #bootstrap_type="No",        # better for small data
                    bootstrap_type='Bernoulli',   # But set sampling frequency to 1.0
                    subsample=1.0,
                    **gpu_params
                )
        else:
            if not isinstance(self.__model, CatBoostClassifier):
                self.__model = CatBoostClassifier(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=lr,
                    loss_function=loss_function,
                    random_seed=42,
                    verbose=True,
                    **gpu_params
                )
            else:
                self.__model.set_params(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=lr,
                    loss_function=loss_function,
                    random_seed=42,
                    verbose=True,
                    **gpu_params
                )

        # Pools
        train_pool = Pool(X_train, y_train)
        eval_set = Pool(X_val, y_val) if (X_val is not None and y_val is not None and X_val.size and y_val.size) else None

        # tqdm redirection context
        from contextlib import contextmanager
        import sys

        # tqdm redirection context (parses CatBoost stdout to show current RMSEs)
        from contextlib import contextmanager
        import sys, re

        @contextmanager
        def _tqdm_redirect(total_iters: int, enabled: bool = True):
            if not enabled:
                yield None
                return

            class _TqdmStream:
                def __init__(self, total):
                    self.pbar = tqdm(
                        total=total if total and total > 0 else 1,
                        desc="CatBoost train",
                        leave=False,
                        dynamic_ncols=True,
                    )
                    self._last_iter = -1
                    self._learn_rmse = None
                    self._val_rmse = None

                def write(self, s):
                    try:
                        line = str(s).strip()
                        if not line:
                            return

                        # 1) Update iteration progress (e.g., "123: learn: ... test: ...")
                        head = line.split(":", 1)[0].strip()
                        if head.isdigit():
                            it = int(head)
                            if it > self._last_iter:
                                self.pbar.update(it - self._last_iter)
                                self._last_iter = it

                        # 2) Parse current RMSE values from the line
                        # CatBoost prints "learn: <rmse>" and "test: <rmse>" (or "validation: <rmse>")
                        m_learn = re.search(r'learn:\s*([0-9]+(?:\.[0-9]+)?)', line)
                        if m_learn:
                            self._learn_rmse = float(m_learn.group(1))

                        m_val = re.search(r'(?:test|validation):\s*([0-9]+(?:\.[0-9]+)?)', line)
                        if m_val:
                            self._val_rmse = float(m_val.group(1))

                        # 3) Show in tqdm postfix
                        postfix = {}
                        if self._learn_rmse is not None:
                            postfix["learn_rmse"] = f"{self._learn_rmse:.5f}"
                        if self._val_rmse is not None:
                            postfix["val_rmse"] = f"{self._val_rmse:.5f}"
                        if postfix:
                            self.pbar.set_postfix(postfix)
                    except Exception:
                        pass

                def flush(self):
                    pass

                def close(self):
                    try:
                        self.pbar.close()
                    except Exception:
                        pass

            logger = _TqdmStream(total_iters)
            old_out, old_err = sys.stdout, sys.stderr
            try:
                sys.stdout = logger
                sys.stderr = logger
                yield logger
            finally:
                sys.stdout = old_out
                sys.stderr = old_err
                logger.close()

        # Fit with early stopping wrapped by tqdm
        fit_params = dict(
            use_best_model=True if eval_set is not None else False,
            early_stopping_rounds=early_stopping_rounds if eval_set is not None else None,
        )
        with _tqdm_redirect(iterations, enabled=show_progress):
            if eval_set is not None:
                self.__model.fit(train_pool, eval_set=eval_set, **fit_params)
            else:
                self.__model.fit(train_pool, **fit_params)

        # Optional: save per-iteration eval metrics
        try:
            evals = self.__model.get_evals_result()  # no extra passes; taken from training log
            # evals structure: {'learn': {'RMSE': [...], 'R2': [...]}, 'validation': {'RMSE': [...], 'R2': [...]}}
            import pandas as pd, numpy as np, os
            rows = []
            # pick keys robustly
            learn = evals.get("learn", {})
            valid_key = next((k for k in evals.keys() if k != "learn"), None)
            valid = evals.get(valid_key, {}) if valid_key else {}
            # number of recorded iters
            n = max((len(next(iter(learn.values()))) if learn else 0),
                    (len(next(iter(valid.values()))) if valid else 0))
            for i in range(n):
                row = {"iteration": i}
                if "RMSE" in learn and i < len(learn["RMSE"]):
                    row["train_rmse"] = learn["RMSE"][i]
                if "R2" in learn and i < len(learn["R2"]):
                    row["train_r2"] = learn["R2"][i]
                if "RMSE" in valid and i < len(valid["RMSE"]):
                    row["val_rmse"] = valid["RMSE"][i]
                if "R2" in valid and i < len(valid["R2"]):
                    row["val_r2"] = valid["R2"][i]
                rows.append(row)
            if rows:
                df_iter = pd.DataFrame(rows)
                os.makedirs(output_metrics_dir, exist_ok=True)
                df_iter.to_csv(os.path.join(output_metrics_dir, "catboost_iteration_metrics.csv"), index=False)
                print(f"✅ CatBoost iteration metrics saved to {os.path.join(output_metrics_dir, 'catboost_iteration_metrics.csv')}")
        except Exception:
            pass
        print("✅ CatBoost training complete.")
        # Keep test arrays for report() if regression
        if is_regression and X_test is not None and y_test is not None:
            self.y_test = y_test
            self.y_test_pred = self.__model.predict(X_test)

        return True
    
    
    def train_on_single_parquet_file_streaming_catboost_rso(
        self,
        *,
        target_col: str = "y",
        exclude_cols: list[str] | None = None,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        iterations: int = 2000,
        depth: int = 8,
        lr: float = 0.03,
        loss_function: str | None = None,
        early_stopping_rounds: int = 100,
        output_metrics_dir: str | None = None,
        show_progress: bool = True,
        enforce_rso_cap: bool = True,          # NEW: clip predictions AFTER training (post-processing)
        enforce_rso_cap_in_train: bool = True, # NEW: also cap targets DURING training
        rso_feature_name: str = "RSO",         # Name (case-insensitive) of RSO feature column
    ):
        """
        Train CatBoost on a single Parquet file with tqdm progress during fit.
        If enforce_rso_cap_in_train=True, the training (and validation) target values are
        first replaced with min(original_target, RSO_feature_value). This does NOT change
        CatBoost’s internal loss, but constrains what it sees as the label so the model
        learns not to exceed RSO. Final predictions can still be (rarely) above RSO, so
        enforce_rso_cap (post-training) will clip them.
        """
        import os
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
        from tqdm.auto import tqdm

        if self.model_name not in ("catboost",):
            raise ValueError("This method is only for CatBoost models.")
        if not getattr(self, "single_parquet_path", None):
            raise ValueError("single_parquet_path not set.")

        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./results_monitor")
        os.makedirs(output_metrics_dir, exist_ok=True)

        pf = pq.ParquetFile(self.single_parquet_path)
        schema = pf.schema_arrow
        names = list(schema.names)
        if target_col not in names:
            raise ValueError(f"Target column '{target_col}' not found in parquet.")

        meta_drop = {"date", "year", "month"}
        feature_cols = [c for c in names if c not in {target_col, *meta_drop}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        self.feature_names_ = feature_cols

        # Locate RSO feature index
        rso_idx = None
        for i, f in enumerate(feature_cols):
            if f.lower() == rso_feature_name.lower():
                rso_idx = i
                break
        if enforce_rso_cap_in_train and rso_idx is None:
            print(f"⚠️  RSO feature '{rso_feature_name}' not found. Training-time cap disabled.")
            enforce_rso_cap_in_train = False

        total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
        from sklearn.model_selection import train_test_split

        if train_ratio is None:
            train_ratio = getattr(self, "train_percent", 0.8)
        if val_ratio is None:
            val_ratio = getattr(self, "validation_percentage", 0.2)

        all_idx = np.arange(total_rows, dtype=np.int64)
        train_idx, rest_idx = train_test_split(all_idx, train_size=train_ratio, random_state=42)
        if len(rest_idx) > 0 and val_ratio and val_ratio > 0:
            denom = max(1e-12, (1 - train_ratio))
            val_rel = min(0.999, val_ratio / denom)
            val_idx, test_idx = train_test_split(rest_idx, train_size=val_rel, random_state=42)
        else:
            val_idx = np.array([], dtype=np.int64)
            test_idx = rest_idx

        print("🗂️  Parquet rows: {:,} | 🧩 Features: {}".format(total_rows, len(feature_cols)))
        print(f"Features: {feature_cols}")
        print(f"Split: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")

        def _gather_by_indices(indices: np.ndarray, desc: str):
            import pyarrow as pa
            feat_parts, tgt_parts = [], []
            if indices.size == 0:
                return None, None
            indices_sorted = np.sort(indices)
            rg_bounds = []
            acc = 0
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                rg_bounds.append((acc, acc + n))
                acc += n
            bar = tqdm(range(pf.num_row_groups), desc=desc, leave=False, disable=not show_progress)
            for rg in bar:
                start, end = rg_bounds[rg]
                left = np.searchsorted(indices_sorted, start, side="left")
                right = np.searchsorted(indices_sorted, end, side="left")
                local = indices_sorted[left:right] - start
                if local.size == 0:
                    continue
                tbl = pf.read_row_group(rg, columns=feature_cols + [target_col])
                tbl_sel = tbl.take(pa.array(local, type=pa.int64()))
                try:
                    X_cols = [tbl_sel[c].to_numpy(zero_copy_only=False) for c in feature_cols]
                    X = np.column_stack(X_cols).astype(np.float32, copy=False)
                    y = tbl_sel[target_col].to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1)
                except Exception:
                    pdf_local = tbl_sel.to_pandas()
                    X = pdf_local[feature_cols].to_numpy(dtype=np.float32, copy=False)
                    y = pdf_local[target_col].to_numpy(dtype=np.float32, copy=False).reshape(-1)
                feat_parts.append(X)
                tgt_parts.append(y)
            if not feat_parts:
                return None, None
            return np.vstack(feat_parts), np.concatenate(tgt_parts)

        X_train, y_train = _gather_by_indices(train_idx, "Load train")
        if X_train is None:
            raise RuntimeError("No training rows collected.")
        X_val, y_val = _gather_by_indices(val_idx, "Load val") if val_idx.size > 0 else (None, None)
        X_test, y_test = _gather_by_indices(test_idx, "Load test") if test_idx.size > 0 else (None, None)

        # ================== NEW: TRAIN-TIME RSO TARGET CAPPING ==================
        if enforce_rso_cap_in_train and rso_idx is not None:
            original_violations = (y_train > X_train[:, rso_idx]).sum()
            y_train = np.minimum(y_train, X_train[:, rso_idx])
            if X_val is not None and y_val is not None:
                y_val = np.minimum(y_val, X_val[:, rso_idx])
            print(f"🔒 Applied in-training RSO cap to targets (train violations clipped: {original_violations:,}).")
        # ========================================================================

        from catboost import CatBoostRegressor, CatBoostClassifier, Pool
        is_regression = (self.__task != 'c')
        if loss_function is None:
            loss_function = "RMSE" if is_regression else "Logloss"

        gpu_params = {}
        if self.use_gpu is True:
            try:
                import torch
                has_cuda = torch.cuda.is_available()
                print(f"CatBoost GPU requested, torch reports CUDA available: {has_cuda}")
            except Exception:
                has_cuda = True
                torch = None
            if has_cuda:
                gpu_params["task_type"] = "GPU"
                devices_str = None
                if bool(getattr(self, "all_gpu", False)) or str(getattr(self, "max_gpus", "")).lower() == "all":
                    try:
                        total = torch.cuda.device_count() if torch is not None else 0
                        if total and total > 0:
                            devices_str = ",".join(str(i) for i in range(total))
                    except Exception:
                        pass
                if devices_str is None and getattr(self, "_used_gpu_ids", None):
                    devices_str = ",".join(str(i) for i in self._used_gpu_ids)
                if devices_str is None:
                    devices_str = "0"
                gpu_params["devices"] = devices_str
                print(f"Using GPU ids: {devices_str}")

        if is_regression:
            if not isinstance(self.__model, CatBoostRegressor):
                self.__model = CatBoostRegressor(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=lr,
                    loss_function=loss_function,
                    eval_metric="RMSE",
                    custom_metric=["R2"],
                    random_seed=42,
                    verbose=1,
                    metric_period=1,
                    use_best_model=True,
                    bootstrap_type='Bernoulli',
                    subsample=1.0,
                    **gpu_params
                )
            else:
                self.__model.set_params(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=lr,
                    loss_function=loss_function,
                    eval_metric="RMSE",
                    custom_metric=["R2"],
                    random_seed=42,
                    verbose=1,
                    metric_period=1,
                    use_best_model=True,
                    bootstrap_type='Bernoulli',
                    subsample=1.0,
                    **gpu_params
                )
        else:
            if not isinstance(self.__model, CatBoostClassifier):
                self.__model = CatBoostClassifier(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=lr,
                    loss_function=loss_function,
                    random_seed=42,
                    verbose=1,
                    **gpu_params
                )
            else:
                self.__model.set_params(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=lr,
                    loss_function=loss_function,
                    random_seed=42,
                    verbose=1,
                    **gpu_params
                )

        train_pool = Pool(X_train, y_train)
        eval_set = Pool(X_val, y_val) if (X_val is not None and y_val is not None and X_val.size and y_val.size) else None

        from contextlib import contextmanager
        import sys, re
        @contextmanager
        def _tqdm_redirect(total_iters: int, enabled: bool = True):
            if not enabled:
                yield None
                return
            class _TqdmStream:
                def __init__(self, total):
                    self.pbar = tqdm(
                        total=total if total and total > 0 else 1,
                        desc="CatBoost train",
                        leave=False,
                        dynamic_ncols=True,
                    )
                    self._last_iter = -1
                    self._learn_rmse = None
                    self._val_rmse = None
                def write(self, s):
                    try:
                        line = str(s).strip()
                        if not line:
                            return
                        head = line.split(":", 1)[0].strip()
                        if head.isdigit():
                            it = int(head)
                            if it > self._last_iter:
                                self.pbar.update(it - self._last_iter)
                                self._last_iter = it
                        m_learn = re.search(r'learn:\s*([0-9]+(?:\.[0-9]+)?)', line)
                        if m_learn:
                            self._learn_rmse = float(m_learn.group(1))
                        m_val = re.search(r'(?:test|validation):\s*([0-9]+(?:\.[0-9]+)?)', line)
                        if m_val:
                            self._val_rmse = float(m_val.group(1))
                        postfix = {}
                        if self._learn_rmse is not None:
                            postfix["learn_rmse"] = f"{self._learn_rmse:.5f}"
                        if self._val_rmse is not None:
                            postfix["val_rmse"] = f"{self._val_rmse:.5f}"
                        if postfix:
                            self.pbar.set_postfix(postfix)
                    except Exception:
                        pass
                def flush(self): pass
                def close(self):
                    try: self.pbar.close()
                    except Exception: pass
            logger = _TqdmStream(total_iters)
            old_out, old_err = sys.stdout, sys.stderr
            try:
                sys.stdout = logger
                sys.stderr = logger
                yield logger
            finally:
                sys.stdout = old_out
                sys.stderr = old_err
                logger.close()

        fit_params = dict(
            use_best_model=True if eval_set is not None else False,
            early_stopping_rounds=early_stopping_rounds if eval_set is not None else None,
        )
        with _tqdm_redirect(iterations, enabled=show_progress):
            if eval_set is not None:
                self.__model.fit(train_pool, eval_set=eval_set, **fit_params)
            else:
                self.__model.fit(train_pool, **fit_params)

        # Save iteration metrics
        try:
            evals = self.__model.get_evals_result()
            rows = []
            learn = evals.get("learn", {})
            valid_key = next((k for k in evals.keys() if k != "learn"), None)
            valid = evals.get(valid_key, {}) if valid_key else {}
            n = max(
                (len(next(iter(learn.values()))) if learn else 0),
                (len(next(iter(valid.values()))) if valid else 0)
            )
            for i in range(n):
                row = {"iteration": i}
                if "RMSE" in learn and i < len(learn["RMSE"]):
                    row["train_rmse"] = learn["RMSE"][i]
                if "R2" in learn and i < len(learn["R2"]):
                    row["train_r2"] = learn["R2"][i]
                if "RMSE" in valid and i < len(valid["RMSE"]):
                    row["val_rmse"] = valid["RMSE"][i]
                if "R2" in valid and i < len(valid["R2"]):
                    row["val_r2"] = valid["R2"][i]
                rows.append(row)
            if rows:
                df_iter = pd.DataFrame(rows)
                df_iter.to_csv(os.path.join(output_metrics_dir, "catboost_iteration_metrics.csv"), index=False)
                print(f"✅ CatBoost iteration metrics saved to {os.path.join(output_metrics_dir, 'catboost_iteration_metrics.csv')}")
        except Exception:
            pass

        print("✅ CatBoost training complete.")

        # Post-training prediction + optional cap
        if is_regression and X_test is not None and y_test is not None:
            raw_pred = self.__model.predict(X_test)
            if enforce_rso_cap and rso_idx is not None:
                capped = np.minimum(raw_pred, X_test[:, rso_idx])
                overs = (raw_pred > X_test[:, rso_idx]).sum()
                print(f"🔒 Post-training RSO cap applied (clipped {overs} predictions).")
                self.y_test_pred = capped
            else:
                self.y_test_pred = raw_pred
            self.y_test = y_test

        return True
    
    
    def cross_validation_no_retrain(
        self,
        *,
        target_col: str = "y",
        date_col: str | None = None,
        mode: str = "kfold",          # 'kfold' | 'yearly' | 'monthly'
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        show_progress: bool = True,
    ):
        """
        Evaluate an already TRAINED model over folds WITHOUT retraining.
        Adds full regression metrics (R2,R,MSE,RMSE,NRMSE_mean,NRMSE_range,MAE,MEDAE,MAPE,
        Bias,PBIAS,NSE,KGE,MSLE(if>0),SMAPE,MPE,CV_RMSE_%,RAE,RSE) and
        classification metrics (Accuracy,BalancedAccuracy,Precision/Recall/F1 macro & weighted,
        MCC, ROC_AUC (binary / OVR), LogLoss) when applicable.

        Warning: optimistic (data leakage) because the model was trained on all data already.
        """
        import numpy as np, pandas as pd, pyarrow.parquet as pq
        from math import sqrt
        from sklearn.model_selection import KFold
        from sklearn.metrics import (
            mean_squared_error, r2_score, mean_absolute_error, median_absolute_error,
            accuracy_score, precision_score, recall_score, f1_score,
            balanced_accuracy_score, matthews_corrcoef, roc_auc_score, log_loss
        )
        try:
            from tqdm.auto import tqdm
        except Exception:
            # Fallback dummy tqdm
            def tqdm(x, **k): return x

        if self.__model is None:
            raise ValueError("No trained model available.")
        is_regression = (self.__task != 'c')

        # ---------------- Regression metric helpers ----------------
        def _mape(y, p):
            mask = y != 0
            if not np.any(mask): return np.nan
            return 100.0 * np.mean(np.abs((y[mask] - p[mask]) / y[mask]))
        def _smape(y, p):
            denom = np.abs(y) + np.abs(p)
            mask = denom != 0
            if not np.any(mask): return np.nan
            return 100.0 * np.mean(2.0 * np.abs(p[mask] - y[mask]) / denom[mask])
        def _mpe(y, p):
            mask = y != 0
            if not np.any(mask): return np.nan
            return 100.0 * np.mean((p[mask] - y[mask]) / y[mask])
        def _nrmse_mean(y, p):
            m = np.mean(y)
            if m == 0: return np.nan
            return sqrt(mean_squared_error(y, p)) / m
        def _nrmse_range(y, p):
            r = np.max(y) - np.min(y)
            if r <= 0: return np.nan
            return sqrt(mean_squared_error(y, p)) / r
        def _bias(y, p):
            return float(np.mean(p - y))
        def _pbias(y, p):
            denom = np.sum(y)
            if denom == 0: return np.nan
            return 100.0 * np.sum(p - y) / denom
        def _nse(y, p):
            denom = np.sum((y - np.mean(y))**2)
            if denom <= 0: return np.nan
            return 1.0 - np.sum((p - y)**2) / denom
        def _kge(y, p):
            sy, sp = np.std(y), np.std(p)
            if sy == 0 or sp == 0: return np.nan
            r = np.corrcoef(y, p)[0, 1]
            alpha = sp / sy
            mu_y, mu_p = np.mean(y), np.mean(p)
            beta = (mu_p / mu_y) if mu_y != 0 else np.nan
            if np.isnan(r) or np.isnan(beta): return np.nan
            return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        def _cvrmse_percent(y, p):
            m = np.mean(y)
            if m == 0: return np.nan
            return 100.0 * sqrt(mean_squared_error(y, p)) / m
        def _rae(y, p):
            denom = np.sum(np.abs(y - np.mean(y)))
            if denom == 0: return np.nan
            return np.sum(np.abs(y - p)) / denom
        def _rse(y, p):
            denom = np.sum((y - np.mean(y))**2)
            if denom == 0: return np.nan
            return np.sum((y - p)**2) / denom

        # ------------ In-memory simple path ------------
        if getattr(self, "x", None) is not None and getattr(self, "y", None) is not None and self.single_parquet_path is None:
            X_all = self.x
            y_all = self.y
            n = X_all.shape[0]
            if mode.lower() != "kfold":
                raise ValueError("Temporal modes require parquet; only kfold supported in pure in-memory arrays.")
            kf = KFold(n_splits=min(n_folds, n), shuffle=shuffle, random_state=random_state)
            rows = []
            iterator = kf.split(np.arange(n))
            if show_progress:
                iterator = tqdm(list(iterator), desc="Folds", unit="fold", leave=False)
            for fold_i, split in enumerate(iterator, start=1):
                _, test_idx = split
                Xf = X_all[test_idx]
                yf = y_all[test_idx]
                yp = self.__model.predict(Xf)
                yp = np.asarray(yp).reshape(-1)
                if is_regression:
                    use_msle = np.all(yf > 0) and np.all(yp > 0)
                    mse = mean_squared_error(yf, yp)
                    rmse = sqrt(mse)
                    rec = {
                        "Fold": f"Fold_{fold_i}",
                        "N": len(test_idx),
                        "R2": float(r2_score(yf, yp)),
                        "R": float(np.corrcoef(yf, yp)[0, 1]) if (np.std(yf) > 0 and np.std(yp) > 0) else np.nan,
                        "MSE": float(mse),
                        "RMSE": float(rmse),
                        "NRMSE_mean": _nrmse_mean(yf, yp),
                        "NRMSE_range": _nrmse_range(yf, yp),
                        "MAE": mean_absolute_error(yf, yp),
                        "MEDAE": median_absolute_error(yf, yp),
                        "MAPE": _mape(yf, yp),
                        "SMAPE": _smape(yf, yp),
                        "MPE": _mpe(yf, yp),
                        "Bias": _bias(yf, yp),
                        "PBIAS": _pbias(yf, yp),
                        "CV_RMSE_%": _cvrmse_percent(yf, yp),
                        "RAE": _rae(yf, yp),
                        "RSE": _rse(yf, yp),
                        "NSE": _nse(yf, yp),
                        "KGE": _kge(yf, yp),
                    }
                    if use_msle:
                        from sklearn.metrics import mean_squared_log_error
                        rec["MSLE"] = mean_squared_log_error(yf, yp)
                    rows.append(rec)
                else:
                    ypc = np.asarray(yp)
                    rec = {
                        "Fold": f"Fold_{fold_i}",
                        "N": len(test_idx),
                        "Accuracy": accuracy_score(yf, ypc),
                        "BalancedAccuracy": balanced_accuracy_score(yf, ypc),
                        "Precision_macro": precision_score(yf, ypc, average="macro", zero_division=0),
                        "Precision_weighted": precision_score(yf, ypc, average="weighted", zero_division=0),
                        "Recall_macro": recall_score(yf, ypc, average="macro", zero_division=0),
                        "Recall_weighted": recall_score(yf, ypc, average="weighted", zero_division=0),
                        "F1_macro": f1_score(yf, ypc, average="macro", zero_division=0),
                        "F1_weighted": f1_score(yf, ypc, average="weighted", zero_division=0),
                        "MCC": matthews_corrcoef(yf, ypc),
                    }
                    try:
                        if hasattr(self.__model, "predict_proba"):
                            proba = self.__model.predict_proba(Xf)
                            if proba.ndim == 2 and proba.shape[1] == 2:
                                rec["ROC_AUC"] = roc_auc_score(yf, proba[:, 1])
                            elif proba.ndim == 2 and proba.shape[1] > 2:
                                rec["ROC_AUC_OVR_weighted"] = roc_auc_score(yf, proba, multi_class="ovr", average="weighted")
                            rec["LogLoss"] = log_loss(yf, proba)
                    except Exception:
                        pass
                    rows.append(rec)
            df = pd.DataFrame(rows)
            if not df.empty:
                mean_row = {"Fold": "Mean", "N": df["N"].sum()}
                for c in df.columns:
                    if c not in ("Fold", "N"):
                        mean_row[c] = df[c].mean()
                df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
            if output_csv:
                df.to_csv(output_csv, index=False)
            print(df)
            return df

        # ------------ Parquet streaming path ------------
        if not getattr(self, "single_parquet_path", None):
            raise ValueError("single_parquet_path required for parquet modes.")

        pf = pq.ParquetFile(self.single_parquet_path, memory_map=True)
        total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))

        # Build folds
        if mode.lower() == "kfold":
            idx_all = np.arange(total_rows, dtype=np.int64)
            kf = KFold(n_splits=min(n_folds, total_rows), shuffle=shuffle, random_state=random_state)
            fold_indices = [test for _, test in kf.split(idx_all)]
            fold_labels = [f"Fold_{i+1}" for i in range(len(fold_indices))]
        else:
            if date_col is None:
                raise ValueError("date_col required for yearly/monthly modes.")
            group_map = {}
            offset = 0
            rg_iter = range(pf.num_row_groups)
            if show_progress:
                rg_iter = tqdm(rg_iter, desc="Scan RG dates", leave=False)
            for rg in rg_iter:
                col = pf.read_row_group(rg, columns=[date_col])[date_col].to_pandas()
                col = pd.to_datetime(col, errors="coerce")
                if mode.lower() == "yearly":
                    groups = col.dt.year
                elif mode.lower() == "monthly":
                    groups = col.dt.to_period("M").astype(str)
                else:
                    raise ValueError("mode must be 'kfold','yearly','monthly'")
                for j, g in enumerate(groups):
                    if pd.isna(g): continue
                    group_map.setdefault(g, []).append(offset + j)
                offset += len(col)
            fold_labels, fold_indices = [], []
            for g, lst in sorted(group_map.items()):
                if lst:
                    fold_labels.append(str(g))
                    fold_indices.append(np.asarray(lst, dtype=np.int64))

        # Precompute RG boundaries
        rg_bounds = []
        acc_rows = 0
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            rg_bounds.append((acc_rows, acc_rows + n))
            acc_rows += n

        results = []
        import pyarrow as pa

        fold_iter = zip(fold_labels, fold_indices)
        if show_progress:
            fold_iter = tqdm(list(fold_iter), desc="Evaluate folds", unit="fold", leave=False)

        for label, test_idx in fold_iter:
            test_idx.sort()
            if test_idx.size == 0:
                continue
            if is_regression:
                y_chunks = []
                p_chunks = []
            else:
                y_chunks = []
                p_chunks = []
                proba_chunks = []

            rg_loop = enumerate(rg_bounds)
            if show_progress:
                rg_loop = tqdm(rg_loop, total=len(rg_bounds), desc=f"{label} RG", leave=False)

            for rg, (start, end) in rg_loop:
                left = np.searchsorted(test_idx, start, side="left")
                right = np.searchsorted(test_idx, end, side="left")
                local = test_idx[left:right] - start
                if local.size == 0:
                    continue

                tbl = pf.read_row_group(rg)
                if hasattr(self, "feature_names_") and self.feature_names_:
                    feat_cols = [c for c in self.feature_names_ if c in tbl.column_names]
                else:
                    exclude = {target_col, "date", "year", "month"}
                    feat_cols = [c for c in tbl.column_names if c not in exclude]

                tbl_sel = tbl.take(pa.array(local, type=pa.int64()))
                df_local = tbl_sel.select(feat_cols + [target_col]).to_pandas()

                Xb = df_local[feat_cols].to_numpy(dtype=np.float32, copy=False)
                yb = df_local[target_col].to_numpy(dtype=np.float32, copy=False)
                preds = self.__model.predict(Xb)
                preds = np.asarray(preds).reshape(-1)

                if is_regression:
                    y_chunks.append(yb)
                    p_chunks.append(preds)
                else:
                    y_chunks.append(yb)
                    p_chunks.append(preds)
                    if hasattr(self.__model, "predict_proba"):
                        try:
                            proba_chunks.append(self.__model.predict_proba(Xb))
                        except Exception:
                            pass

            if is_regression:
                if not p_chunks:
                    continue
                y_all = np.concatenate(y_chunks)
                p_all = np.concatenate(p_chunks)
                mse = mean_squared_error(y_all, p_all)
                rmse = sqrt(mse)
                use_msle = np.all(y_all > 0) and np.all(p_all > 0)
                rec = {
                    "Fold": label,
                    "N": y_all.size,
                    "R2": float(r2_score(y_all, p_all)),
                    "R": float(np.corrcoef(y_all, p_all)[0, 1]) if (np.std(y_all) > 0 and np.std(p_all) > 0) else np.nan,
                    "MSE": float(mse),
                    "RMSE": float(rmse),
                    "NRMSE_mean": _nrmse_mean(y_all, p_all),
                    "NRMSE_range": _nrmse_range(y_all, p_all),
                    "MAE": mean_absolute_error(y_all, p_all),
                    "MEDAE": median_absolute_error(y_all, p_all),
                    "MAPE": _mape(y_all, p_all),
                    "SMAPE": _smape(y_all, p_all),
                    "MPE": _mpe(y_all, p_all),
                    "Bias": _bias(y_all, p_all),
                    "PBIAS": _pbias(y_all, p_all),
                    "CV_RMSE_%": _cvrmse_percent(y_all, p_all),
                    "RAE": _rae(y_all, p_all),
                    "RSE": _rse(y_all, p_all),
                    "NSE": _nse(y_all, p_all),
                    "KGE": _kge(y_all, p_all),
                }
                if use_msle:
                    from sklearn.metrics import mean_squared_log_error
                    rec["MSLE"] = mean_squared_log_error(y_all, p_all)
                results.append(rec)
            else:
                if not p_chunks:
                    continue
                y_all = np.concatenate(y_chunks)
                p_all = np.concatenate(p_chunks)
                rec = {
                    "Fold": label,
                    "N": y_all.size,
                    "Accuracy": accuracy_score(y_all, p_all),
                    "BalancedAccuracy": balanced_accuracy_score(y_all, p_all),
                    "Precision_macro": precision_score(y_all, p_all, average="macro", zero_division=0),
                    "Precision_weighted": precision_score(y_all, p_all, average="weighted", zero_division=0),
                    "Recall_macro": recall_score(y_all, p_all, average="macro", zero_division=0),
                    "Recall_weighted": recall_score(y_all, p_all, average="weighted", zero_division=0),
                    "F1_macro": f1_score(y_all, p_all, average="macro", zero_division=0),
                    "F1_weighted": f1_score(y_all, p_all, average="weighted", zero_division=0),
                    "MCC": matthews_corrcoef(y_all, p_all),
                }
                try:
                    if proba_chunks:
                        proba_all = np.concatenate(proba_chunks, axis=0)
                        if proba_all.ndim == 2 and proba_all.shape[1] == 2:
                            rec["ROC_AUC"] = roc_auc_score(y_all, proba_all[:, 1])
                            rec["LogLoss"] = log_loss(y_all, proba_all)
                        elif proba_all.ndim == 2 and proba_all.shape[1] > 2:
                            rec["ROC_AUC_OVR_weighted"] = roc_auc_score(y_all, proba_all, multi_class="ovr", average="weighted")
                            rec["LogLoss"] = log_loss(y_all, proba_all)
                except Exception:
                    pass
                results.append(rec)

        df = pd.DataFrame(results)
        if not df.empty:
            mean_row = {"Fold": "Mean", "N": df["N"].sum()}
            for c in df.columns:
                if c not in ("Fold", "N"):
                    mean_row[c] = df[c].mean()
            df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
            
            
        output_csv = f"{self.output_metrics_dir}/cross_validation_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved CV (no-retrain) metrics -> {output_csv}")

        print(df)
        return df
    
    
    
    def cross_validation_no_retrain_v1(
        self,
        *,
        target_col: str = "y",
        date_col: str | None = None,
        mode: str = "kfold",          # 'kfold' | 'yearly' | 'monthly'
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        metrics: tuple[str, ...] = ("rmse", "r2"),
        output_csv: str | None = None,
    ):
        """
        Evaluate an already TRAINED model over folds WITHOUT retraining.
        Supports in‑memory (self.x/self.y) and huge single parquet file (self.single_parquet_path).

        Modes:
          kfold   : classic KFold over rows (in‑memory or parquet index space)
          yearly  : group by year extracted from date_col (parquet only unless data provided)
          monthly : group by year-month from date_col

        Notes:
          - For big parquet we stream only needed rows per fold (no full load).
          - Works for regression (RMSE,R2,MAE) and classification (Accuracy,F1).
          - Using one trained model for all folds is optimistic (data leakage); use for diagnostics only.
        """
        import numpy as np, pandas as pd, pyarrow.parquet as pq
        from sklearn.metrics import (
            mean_squared_error, r2_score, mean_absolute_error,
            accuracy_score, f1_score
        )
        from math import sqrt
        from sklearn.model_selection import KFold

        if self.__model is None:
            raise ValueError("No trained model available.")

        is_regression = (self.__task != 'c')

        # ------------ Helper incremental metric accumulation (regression) ------------
        class RegAccum:
            def __init__(self):
                self.n = 0
                self.mean_y = 0.0
                self.M2_y = 0.0   # for SST
                self.sse = 0.0    # Σ (y - yhat)^2
                self.mae_sum = 0.0
            def update(self, y_batch, pred_batch):
                y = y_batch.astype(float)
                p = pred_batch.astype(float)
                # Welford for y variance (SST)
                for val in y:
                    self.n += 1
                    delta = val - self.mean_y
                    self.mean_y += delta / self.n
                    self.M2_y += delta * (val - self.mean_y)
                self.sse += np.sum((y - p)**2)
                self.mae_sum += np.sum(np.abs(y - p))
            def finalize(self):
                if self.n == 0:
                    return {m: np.nan for m in ("RMSE","R2","MAE")}
                rmse = sqrt(self.sse / self.n)
                sst = self.M2_y if self.M2_y > 0 else np.nan
                r2  = 1 - self.sse / sst if sst and sst > 0 else np.nan
                mae = self.mae_sum / self.n
                return {"RMSE": rmse, "R2": r2, "MAE": mae}

        # ------------ In-memory path ------------
        if getattr(self, "x", None) is not None and getattr(self, "y", None) is not None and self.single_parquet_path is None:
            X_all = self.x
            y_all = self.y
            n = X_all.shape[0]
            if mode != "kfold":
                raise ValueError("Temporal modes require parquet or a DataFrame with date_col; only kfold supported in pure in-memory arrays.")
            kf = KFold(n_splits=min(n_folds, n), shuffle=shuffle, random_state=random_state)
            rows = []
            fold_id = 1
            for _, test_idx in kf.split(np.arange(n)):
                X_fold = X_all[test_idx]
                y_fold = y_all[test_idx]
                y_pred = self.__model.predict(X_fold)
                y_pred = np.asarray(y_pred).reshape(-1)
                if is_regression:
                    rmse = sqrt(mean_squared_error(y_fold, y_pred))
                    r2   = r2_score(y_fold, y_pred)
                    mae  = mean_absolute_error(y_fold, y_pred)
                    rec = {"Fold": fold_id, "N": len(test_idx), "RMSE": rmse, "R2": r2, "MAE": mae}
                else:
                    y_pred_cls = y_pred
                    acc = accuracy_score(y_fold, y_pred_cls)
                    f1  = f1_score(y_fold, y_pred_cls, average="weighted")
                    rec = {"Fold": fold_id, "N": len(test_idx), "Accuracy": acc, "F1": f1}
                rows.append(rec)
                fold_id += 1
            df = pd.DataFrame(rows)
            if not df.empty:
                mean_row = {"Fold": "Mean", "N": df["N"].sum()}
                for c in df.columns:
                    if c not in ("Fold","N"):
                        mean_row[c] = df[c].mean()
                df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
            if output_csv:
                df.to_csv(output_csv, index=False)
            print(df)
            return df

        # ------------ Big single parquet path ------------
        if not getattr(self, "single_parquet_path", None):
            raise ValueError("For big parquet evaluation set single_parquet_path or provide in-memory data.")

        pf = pq.ParquetFile(self.single_parquet_path, memory_map=True)
        total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))

        # Build fold index lists
        if mode == "kfold":
            idx_all = np.arange(total_rows, dtype=np.int64)
            kf = KFold(n_splits=min(n_folds, total_rows), shuffle=shuffle, random_state=random_state)
            fold_indices = [test for _, test in kf.split(idx_all)]
            fold_labels = [f"Fold_{i+1}" for i in range(len(fold_indices))]
        else:
            if date_col is None:
                raise ValueError("date_col required for yearly/monthly modes.")
            # Pass 1: collect groups (year or year-month)
            group_map = {}  # group -> list of global row indices
            offset = 0
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=[date_col])
                # Convert to numpy str -> datetime
                col = tbl[date_col].to_pandas()
                # coerce to datetime
                col = pd.to_datetime(col, errors="coerce")
                if mode == "yearly":
                    groups = col.dt.year
                elif mode == "monthly":
                    groups = col.dt.to_period("M").astype(str)
                else:
                    raise ValueError("mode must be one of kfold/yearly/monthly")
                for local_pos, g in enumerate(groups):
                    if pd.isna(g):
                        continue
                    global_idx = offset + local_pos
                    group_map.setdefault(g, []).append(global_idx)
                offset += len(col)
            # Each group is a fold
            fold_labels = []
            fold_indices = []
            for g, idx_list in sorted(group_map.items()):
                if len(idx_list) == 0:
                    continue
                fold_labels.append(str(g))
                fold_indices.append(np.asarray(idx_list, dtype=np.int64))

        # Precompute row-group boundaries for fast intersection
        rg_bounds = []
        acc = 0
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            rg_bounds.append((acc, acc + n))
            acc += n

        results = []
        for label, test_idx in zip(fold_labels, fold_indices):
            test_idx.sort()
            if test_idx.size == 0:
                continue
            if is_regression:
                accum = RegAccum()
            else:
                true_all = []
                pred_all = []

            # Stream only rows of this fold
            for rg, (start, end) in enumerate(rg_bounds):
                # slice indices belonging to this row-group
                left = np.searchsorted(test_idx, start, side="left")
                right = np.searchsorted(test_idx, end, side="left")
                local = test_idx[left:right] - start
                if local.size == 0:
                    continue
                tbl = pf.read_row_group(rg)
                # subset columns needed (all feature columns + target + possibly date)
                # Determine features from stored feature_names_ if available else infer now
                if hasattr(self, "feature_names_") and self.feature_names_:
                    feat_cols = [c for c in self.feature_names_ if c in tbl.column_names]
                else:
                    # infer (exclude target/date/year/month)
                    exclude = {target_col, "date", "year", "month"}
                    feat_cols = [c for c in tbl.column_names if c not in exclude]
                # Take only local rows
                import pyarrow as pa
                tbl_sel = tbl.take(pa.array(local, type=pa.int64()))
                # Convert to numpy
                df_local = tbl_sel.select(feat_cols + [target_col]).to_pandas()
                X_batch = df_local[feat_cols].to_numpy(dtype=np.float32, copy=False)
                y_batch = df_local[target_col].to_numpy(dtype=np.float32, copy=False)
                # Predict
                preds = self.__model.predict(X_batch)
                preds = np.asarray(preds).reshape(-1)
                if is_regression:
                    accum.update(y_batch, preds)
                else:
                    true_all.append(y_batch)
                    pred_all.append(preds)

            if is_regression:
                fold_metrics = accum.finalize()
                rec = {"Fold": label, "N": accum.n}
                for m in metrics:
                    ml = m.lower()
                    if ml == "rmse": rec["RMSE"] = fold_metrics["RMSE"]
                    elif ml == "r2": rec["R2"] = fold_metrics["R2"]
                    elif ml == "mae": rec["MAE"] = fold_metrics["MAE"]
                results.append(rec)
            else:
                if not pred_all:
                    continue
                y_true = np.concatenate(true_all)
                y_pred = np.concatenate(pred_all)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average="weighted")
                results.append({"Fold": label, "N": y_true.size, "Accuracy": acc, "F1": f1})

        df = pd.DataFrame(results)
        if not df.empty:
            mean_row = {"Fold": "Mean", "N": df["N"].sum()}
            for c in df.columns:
                if c not in ("Fold","N"):
                    mean_row[c] = df[c].mean()
            df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Saved CV (no-retrain) metrics -> {output_csv}")

        print(df)
        return df
    
    
    def train_on_single_parquet_file_streaming_catboost_v1(
        self,
        *,
        target_col: str = "y",
        exclude_cols: list[str] | None = None,
        train_ratio: float | None = None,           # defaults to self.train_percent
        val_ratio: float | None = None,             # defaults to self.validation_percentage
        iterations: int = 2000,
        depth: int = 8,
        lr: float = 0.03,
        loss_function: str | None = None,           # e.g. 'RMSE' for regression
        early_stopping_rounds: int = 100,
        output_metrics_dir: str | None = None,
        show_progress: bool = True,
    ):
        """
        Train CatBoost on a single Parquet file with tqdm progress during fit.
        """
        import os
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
        from tqdm.auto import tqdm

        if self.model_name not in ("catboost",):
            raise ValueError("This method is only for CatBoost models.")
        if not getattr(self, "single_parquet_path", None):
            raise ValueError("single_parquet_path not set.")

        # Resolve output directory
        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./results_monitor")
        os.makedirs(output_metrics_dir, exist_ok=True)

        # Infer schema/feature columns
        pf = pq.ParquetFile(self.single_parquet_path)
        schema = pf.schema_arrow
        names = list(schema.names)
        if target_col not in names:
            raise ValueError(f"Target column '{target_col}' not found in parquet.")

        meta_drop = {"date", "year", "month"}
        feature_cols = [c for c in names if c not in {target_col, *meta_drop}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        self.feature_names_ = feature_cols

        # Split indices (train/val/test) by row count
        total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
        from sklearn.model_selection import train_test_split

        if train_ratio is None:
            train_ratio = getattr(self, "train_percent", 0.8)
        if val_ratio is None:
            val_ratio = getattr(self, "validation_percentage", 0.2)

        all_idx = np.arange(total_rows, dtype=np.int64)
        train_idx, rest_idx = train_test_split(all_idx, train_size=train_ratio, random_state=42)
        if len(rest_idx) > 0 and val_ratio and val_ratio > 0:
            denom = max(1e-12, (1 - train_ratio))
            val_rel = min(0.999, val_ratio / denom)
            val_idx, test_idx = train_test_split(rest_idx, train_size=val_rel, random_state=42)
        else:
            val_idx = np.array([], dtype=np.int64)
            test_idx = rest_idx

        print(f"Split: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")

        # Helper: materialize (X,y) by iterating row-groups and taking needed rows
        def _gather_by_indices(indices: np.ndarray, desc: str):
            import pyarrow as pa
            feat_parts, tgt_parts = [], []
            if indices.size == 0:
                return None, None
            indices_sorted = np.sort(indices)
            # Precompute RG offsets to map global->local
            rg_bounds = []
            acc = 0
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                rg_bounds.append((acc, acc + n))
                acc += n
            bar = tqdm(range(pf.num_row_groups), desc=desc, leave=False, disable=not show_progress)
            for rg in bar:
                start, end = rg_bounds[rg]
                left = np.searchsorted(indices_sorted, start, side="left")
                right = np.searchsorted(indices_sorted, end, side="left")
                local = indices_sorted[left:right] - start
                if local.size == 0:
                    continue
                tbl = pf.read_row_group(rg, columns=feature_cols + [target_col])
                tbl_sel = tbl.take(pa.array(local, type=pa.int64()))
                try:
                    X_cols = [tbl_sel[c].to_numpy(zero_copy_only=False) for c in feature_cols]
                    X = np.column_stack(X_cols).astype(np.float32, copy=False)
                    y = tbl_sel[target_col].to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1)
                except Exception:
                    pdf = tbl_sel.to_pandas()
                    X = pdf[feature_cols].to_numpy(dtype=np.float32, copy=False)
                    y = pdf[target_col].to_numpy(dtype=np.float32, copy=False).reshape(-1)
                feat_parts.append(X)
                tgt_parts.append(y)
            if not feat_parts:
                return None, None
            X_all = np.vstack(feat_parts)
            y_all = np.concatenate(tgt_parts)
            return X_all, y_all

        # Gather train/val/test
        X_train, y_train = _gather_by_indices(train_idx, "Load train")
        if X_train is None:
            raise RuntimeError("No training rows collected.")
        X_val, y_val = _gather_by_indices(val_idx, "Load val") if val_idx.size > 0 else (None, None)
        X_test, y_test = _gather_by_indices(test_idx, "Load test") if test_idx.size > 0 else (None, None)

        # Build CatBoost model (GPU if requested)
        from catboost import CatBoostRegressor, CatBoostClassifier, Pool
        is_regression = (self.__task != 'c')

        if loss_function is None:
            loss_function = "RMSE" if is_regression else "Logloss"

        # Decide GPU usage for CatBoost
        gpu_params = {}
        if self.use_gpu == True:
            try:
                import torch
                has_cuda = torch.cuda.is_available()
                print(f"CatBoost GPU requested, torch reports CUDA available: {has_cuda}")
            except Exception:
                has_cuda = True  # assume available if torch missing
            if has_cuda:
                gpu_params["task_type"] = "GPU"
                # Use configured GPU ids if available, else default to 0
                if getattr(self, "_used_gpu_ids", None):
                    devices_str = ",".join(str(i) for i in self._used_gpu_ids)
                else:
                    devices_str = "0"
                
                gpu_params["devices"] = devices_str
                print(f"Using GPU ids: {devices_str}")

        if is_regression:
            if not isinstance(self.__model, CatBoostRegressor):
                self.__model = CatBoostRegressor(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=lr,
                    loss_function=loss_function,
                    random_seed=42,
                    verbose=True,  # keep verbose for tqdm parsing
                    **gpu_params
                )
            else:
                self.__model.set_params(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=lr,
                    loss_function=loss_function,
                    random_seed=42,
                    verbose=True,
                    **gpu_params
                )
        else:
            if not isinstance(self.__model, CatBoostClassifier):
                self.__model = CatBoostClassifier(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=lr,
                    loss_function=loss_function,
                    random_seed=42,
                    verbose=True,
                    **gpu_params
                )
            else:
                self.__model.set_params(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=lr,
                    loss_function=loss_function,
                    random_seed=42,
                    verbose=True,
                    **gpu_params
                )

        # Pools
        train_pool = Pool(X_train, y_train)
        eval_set = Pool(X_val, y_val) if (X_val is not None and y_val is not None and X_val.size and y_val.size) else None

        # tqdm redirection context
        from contextlib import contextmanager
        import sys

        @contextmanager
        def _tqdm_redirect(total_iters: int, enabled: bool = True):
            if not enabled:
                yield None
                return
            class _TqdmStream:
                def __init__(self, total):
                    self.pbar = tqdm(total=total, desc="CatBoost train", leave=False)
                    self._last_iter = -1
                def write(self, s):
                    try:
                        line = str(s).strip()
                        if not line:
                            return
                        # Lines look like: "0:\tlearn: 1.2345\ttest: 1.5678\tbest: ..."
                        head = line.split(":", 1)[0]
                        if head.isdigit():
                            it = int(head)
                            if it > self._last_iter:
                                self.pbar.update(it - self._last_iter)
                                self._last_iter = it
                    except Exception:
                        pass
                def flush(self): pass
                def close(self):
                    try:
                        self.pbar.close()
                    except Exception:
                        pass
            logger = _TqdmStream(total_iters if total_iters and total_iters > 0 else 1)
            old_out, old_err = sys.stdout, sys.stderr
            try:
                sys.stdout = logger
                sys.stderr = logger
                yield logger
            finally:
                sys.stdout = old_out
                sys.stderr = old_err
                logger.close()

        # Fit with early stopping wrapped by tqdm
        fit_params = dict(
            use_best_model=True if eval_set is not None else False,
            early_stopping_rounds=early_stopping_rounds if eval_set is not None else None,
        )
        with _tqdm_redirect(iterations, enabled=show_progress):
            if eval_set is not None:
                self.__model.fit(train_pool, eval_set=eval_set, **fit_params)
            else:
                self.__model.fit(train_pool, **fit_params)

        # Optional: save per-iteration eval metrics
        try:
            import numpy as np
            import pandas as pd
            from sklearn.metrics import mean_squared_error, r2_score

            def _rmse(y_true, y_pred):
                return float(np.sqrt(mean_squared_error(y_true, y_pred)))

            def _r2(y_true, y_pred):
                try:
                    return float(r2_score(y_true, y_pred))
                except Exception:
                    return float("nan")

            # staged_predict yields predictions after each iteration
            learn_rmse_list, learn_r2_list = [], []
            val_rmse_list, val_r2_list = [], []

            # Estimate total iterations for tqdm (best available)
            total_iters = None
            try:
                total_iters = int(getattr(self.__model, "tree_count_", None) or self.__model.get_tree_count())
            except Exception:
                pass
            if not total_iters:
                try:
                    bi = self.__model.get_best_iteration()
                    if bi is not None and bi >= 0:
                        total_iters = int(bi + 1)
                except Exception:
                    total_iters = None

            # Train iter metrics with tqdm
            for preds in tqdm(
                self.__model.staged_predict(train_pool),
                total=total_iters,
                desc="CatBoost iter (train)",
                leave=False,
                disable=not show_progress
            ):
                learn_rmse_list.append(_rmse(y_train, preds))
                learn_r2_list.append(_r2(y_train, preds))

            # Val iter metrics (if eval_set exists) with tqdm
            if eval_set is not None:
                for preds in tqdm(
                    self.__model.staged_predict(eval_set),
                    total=total_iters,
                    desc="CatBoost iter (val)",
                    leave=False,
                    disable=not show_progress
                ):
                    val_rmse_list.append(_rmse(y_val, preds))
                    val_r2_list.append(_r2(y_val, preds))

            iters = len(learn_rmse_list)
            df_rows = []
            for i in tqdm(
                range(iters),
                desc="Assemble iter metrics",
                leave=False,
                disable=not show_progress
            ):
                row = {
                    "iteration": i,
                    "train_rmse": learn_rmse_list[i],
                    "train_r2": learn_r2_list[i],
                }
                if val_rmse_list:
                    if i < len(val_rmse_list):
                        row["val_rmse"] = val_rmse_list[i]
                        row["val_r2"] = val_r2_list[i]
                    else:
                        row["val_rmse"] = np.nan
                        row["val_r2"] = np.nan
                df_rows.append(row)
            print(f"✅ Collected per-iteration CatBoost metrics for {iters} iterations.")
            if df_rows:
                df_iter = pd.DataFrame(df_rows)
                os.makedirs(output_metrics_dir, exist_ok=True)
                df_iter.to_csv(os.path.join(output_metrics_dir, "catboost_iteration_metrics.csv"), index=False)
                print(f"✅ Saved iteration metrics -> {os.path.join(output_metrics_dir, 'catboost_iteration_metrics.csv')}")
        except Exception:
            pass
        print("✅ CatBoost training complete.")
        # Keep test arrays for report() if regression
        if is_regression and X_test is not None and y_test is not None:
            self.y_test = y_test
            self.y_test_pred = self.__model.predict(X_test)

        return True
    
    
    def train_on_single_parquet_file_streaming_catboost_old(
        self,
        *,
        target_col: str = "y",
        exclude_cols: list[str] | None = None,
        data_loading_batch_size: int = 1_000_000,   # rows per IO batch (used only to chunk reads)
        train_ratio: float | None = None,           # defaults to self.train_percent
        val_ratio: float | None = None,             # defaults to self.validation_percentage
        iterations: int = 2000,
        depth: int = 8,
        lr: float = 0.03,
        loss_function: str | None = None,           # e.g. 'RMSE' for regression
        early_stopping_rounds: int = 100,
        output_metrics_dir: str | None = None,
        overwrite_history: bool = True,
        show_progress: bool = True,
    ):
        """
        Train CatBoost on a single Parquet file.
        Reads the Parquet in batches, accumulates numpy arrays, and fits CatBoost with eval_set.
        """
        import os
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
        from tqdm.auto import tqdm

        if self.model_name not in ("catboost",):
            raise ValueError("This method is only for CatBoost models.")
        if not getattr(self, "single_parquet_path", None):
            raise ValueError("single_parquet_path not set.")

        # Resolve output directory
        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./results_monitor")
        os.makedirs(output_metrics_dir, exist_ok=True)

        # Infer schema/feature columns
        pf = pq.ParquetFile(self.single_parquet_path)
        schema = pf.schema_arrow
        names = list(schema.names)
        if target_col not in names:
            raise ValueError(f"Target column '{target_col}' not found in parquet.")

        meta_drop = {"date", "year", "month"}
        feature_cols = [c for c in names if c not in {target_col, *meta_drop}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        self.feature_names_ = feature_cols

        # Split indices (train/val/test) by row count
        total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
        import numpy as np
        from sklearn.model_selection import train_test_split

        if train_ratio is None:
            train_ratio = getattr(self, "train_percent", 0.8)
        if val_ratio is None:
            val_ratio = getattr(self, "validation_percentage", 0.2)

        all_idx = np.arange(total_rows, dtype=np.int64)
        train_idx, rest_idx = train_test_split(all_idx, train_size=train_ratio, random_state=42)
        if len(rest_idx) > 0 and val_ratio and val_ratio > 0:
            denom = max(1e-12, (1 - train_ratio))
            val_rel = min(0.999, val_ratio / denom)
            val_idx, test_idx = train_test_split(rest_idx, train_size=val_rel, random_state=42)
        else:
            val_idx = np.array([], dtype=np.int64)
            test_idx = rest_idx

        # Helper: materialize (X,y) by iterating row-groups and taking needed rows
        def _gather_by_indices(indices: np.ndarray, desc: str):
            import pyarrow as pa
            feat_parts, tgt_parts = [], []
            if indices.size == 0:
                return None, None
            indices_sorted = np.sort(indices)
            # Precompute RG offsets to map global->local
            rg_bounds = []
            acc = 0
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                rg_bounds.append((acc, acc + n))
                acc += n
            bar = tqdm(range(pf.num_row_groups), desc=desc, leave=False, disable=not show_progress)
            for rg in bar:
                start, end = rg_bounds[rg]
                left = np.searchsorted(indices_sorted, start, side="left")
                right = np.searchsorted(indices_sorted, end, side="left")
                local = indices_sorted[left:right] - start
                if local.size == 0:
                    continue
                tbl = pf.read_row_group(rg, columns=feature_cols + [target_col])
                # subset
                tbl_sel = tbl.take(pa.array(local, type=pa.int64()))
                # Arrow -> NumPy fast path
                try:
                    X_cols = [tbl_sel[c].to_numpy(zero_copy_only=False) for c in feature_cols]
                    X = np.column_stack(X_cols).astype(np.float32, copy=False)
                    y = tbl_sel[target_col].to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1)
                except Exception:
                    pdf = tbl_sel.to_pandas()
                    X = pdf[feature_cols].to_numpy(dtype=np.float32, copy=False)
                    y = pdf[target_col].to_numpy(dtype=np.float32, copy=False).reshape(-1)
                feat_parts.append(X)
                tgt_parts.append(y)
            if not feat_parts:
                return None, None
            X_all = np.vstack(feat_parts)
            y_all = np.concatenate(tgt_parts)
            return X_all, y_all

        # Gather train/val/test
        X_train, y_train = _gather_by_indices(train_idx, "Load train")
        if X_train is None:
            raise RuntimeError("No training rows collected.")
        X_val, y_val = _gather_by_indices(val_idx, "Load val") if val_idx.size > 0 else (None, None)
        X_test, y_test = _gather_by_indices(test_idx, "Load test") if test_idx.size > 0 else (None, None)

        # Build CatBoost model (reuse existing if present)
        from catboost import CatBoostRegressor, CatBoostClassifier, Pool
        is_regression = (self.__task != 'c')

        if loss_function is None:
            loss_function = "RMSE" if is_regression else "Logloss"

        if is_regression:
            if not isinstance(self.__model, CatBoostRegressor):
                self.__model = CatBoostRegressor(
                    iterations=iterations, depth=depth, learning_rate=lr,
                    loss_function=loss_function, random_seed=42, verbose=False
                )
        else:
            if not isinstance(self.__model, CatBoostClassifier):
                self.__model = CatBoostClassifier(
                    iterations=iterations, depth=depth, learning_rate=lr,
                    loss_function=loss_function, random_seed=42, verbose=False
                )

        # Pools
        train_pool = Pool(X_train, y_train)
        eval_set = None
        if X_val is not None and y_val is not None and X_val.size and y_val.size:
            eval_set = Pool(X_val, y_val)

        # Fit with early stopping
        fit_params = dict(
            use_best_model=True if eval_set is not None else False,
            early_stopping_rounds=early_stopping_rounds if eval_set is not None else None,
            verbose=False
        )
        if eval_set is not None:
            self.__model.fit(train_pool, eval_set=eval_set, **fit_params)
        else:
            self.__model.fit(train_pool, **fit_params)

        # Metrics
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
        history = []

        if is_regression:
            y_pred_tr = self.__model.predict(X_train)
            tr_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_tr)))
            tr_r2 = float(r2_score(y_train, y_pred_tr))
            val_rmse = val_r2 = None
            if X_val is not None and y_val is not None:
                y_pred_val = self.__model.predict(X_val)
                val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
                val_r2 = float(r2_score(y_val, y_pred_val))
            test_rmse = test_r2 = None
            if X_test is not None and y_test is not None:
                y_pred_te = self.__model.predict(X_test)
                test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_te)))
                test_r2 = float(r2_score(y_test, y_pred_te))
                print(f"Test -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

            history.append({
                "epoch": 1,
                "train_rmse": tr_rmse,
                "train_r2": tr_r2,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "test_rmse": test_rmse,
                "test_r2": test_r2
            })
        else:
            y_pred_tr = self.__model.predict(X_train)
            tr_acc = float(accuracy_score(y_train, y_pred_tr))
            tr_f1 = float(f1_score(y_train, y_pred_tr, average="weighted"))
            val_acc = val_f1 = None
            if X_val is not None and y_val is not None:
                y_pred_val = self.__model.predict(X_val)
                val_acc = float(accuracy_score(y_val, y_pred_val))
                val_f1 = float(f1_score(y_val, y_pred_val, average="weighted"))
            test_acc = test_f1 = None
            if X_test is not None and y_test is not None:
                y_pred_te = self.__model.predict(X_test)
                test_acc = float(accuracy_score(y_test, y_pred_te))
                test_f1 = float(f1_score(y_test, y_pred_te, average="weighted"))
                print(f"Test -> Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

            history.append({
                "epoch": 1,
                "train_acc": tr_acc,
                "train_f1": tr_f1,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "test_acc": test_acc,
                "test_f1": test_f1
            })

        # Save CSV
        csv_path = os.path.join(output_metrics_dir, "catboost_streaming_history.csv")
        if (not overwrite_history) and os.path.exists(csv_path):
            # append
            prev = pd.read_csv(csv_path)
            pd.concat([prev, pd.DataFrame(history)], ignore_index=True).to_csv(csv_path, index=False)
        else:
            pd.DataFrame(history).to_csv(csv_path, index=False)
        print(f"Saved history -> {csv_path}")

        # Keep test arrays for report() if regression
        if is_regression and X_test is not None and y_test is not None:
            self.y_test = y_test
            self.y_test_pred = self.__model.predict(X_test)

        return history
    
    
    def train_on_single_parquet_file_streaming(
        self,
        epochs: int = 50,
        data_loading_batch_size: int = 8_388_608,  # Large batch for efficient disk I/O
        model_batch_size: int | None = 256,        # Smaller batch for model training
        target_col: str = 'y',
        exclude_cols: list | None = None,
        log_every: int = 10,
        early_stopping_patience: int = 10,
        output_metrics_dir: str | None = None,
        enable_tensorboard: bool = True,
        lr: float = 1e-5
    ):
        """
        Train a model on a single parquet file with memory-efficient streaming.
        Now uses torch.utils.data.IterableDataset for fast multi-worker I/O and GPU overlap.
        """
        import os
        import torch
        import torch.nn as nn
        from torch.utils.data import IterableDataset, DataLoader
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from sklearn.metrics import mean_squared_error, r2_score
        import pyarrow.parquet as pq
        import pyarrow as pa

        if not self.single_parquet_path:
            raise ValueError("single_parquet_path not set. This method requires a single parquet file.")

        # Validate/resolve batch sizes
        if model_batch_size is None or int(model_batch_size) <= 0:
            model_batch_size = int(getattr(self, "_Model__batch_size", 64) or 64)
        model_batch_size = int(model_batch_size)
        data_loading_batch_size = max(1, int(data_loading_batch_size))

        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./results_monitor")
        os.makedirs(output_metrics_dir, exist_ok=True)
      
        # Setup TensorBoard logging if enabled
        writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tensorboard_dir = os.path.join(output_metrics_dir, "tensorboard")
                os.makedirs(tensorboard_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=tensorboard_dir)
                print(f"TensorBoard logging -> {tensorboard_dir}")
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")

        # Infer features from Parquet schema
        pf = pq.ParquetFile(self.single_parquet_path)
        schema = pf.schema_arrow
        col_names = list(schema.names)

        if target_col not in col_names:
            raise ValueError(f"Target column '{target_col}' not found in parquet file.")

        drop_meta = {"date", "year", "month"}
        feature_cols = [c for c in col_names if c not in {target_col, *drop_meta}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        self.feature_names_ = feature_cols

        # Count total rows
        total_rows = sum(pf.metadata.row_group(rg).num_rows for rg in range(pf.num_row_groups))
        print(f"Rows: {total_rows:,} | Features: {len(feature_cols)}")
        print(f"Feature columns: {feature_cols}")
        # Train/Val/Test split over row indices
        from sklearn.model_selection import train_test_split
        all_indices = np.arange(total_rows)

        train_full_indices, test_indices = train_test_split(
            all_indices,
            train_size=self.train_percent,
            random_state=42
        )

        if self.validation_percentage and self.validation_percentage > 0:
            val_size = min(0.999, float(self.validation_percentage))
            train_indices, val_indices = train_test_split(
                train_full_indices,
                test_size=val_size,
                random_state=42
            )
        else:
            train_indices = train_full_indices
            val_indices = np.array([], dtype=int)

        print(f"Split: train={len(train_indices):,}, val={len(val_indices):,}, test={len(test_indices):,}")

        # Build IterableDatasets
        train_ds = ParquetIterableDS(
            self.single_parquet_path,
            feature_cols,
            target_col=target_col,
            batch_size=data_loading_batch_size,
            indices=train_indices,
            shuffle=False,
            seed=42,
            name="IO-Train"
        )
        val_ds = ParquetIterableDS(
            self.single_parquet_path,
            feature_cols,
            target_col=target_col,
            batch_size=data_loading_batch_size,
            indices=val_indices,
            shuffle=False,
            seed=43,
            name="IO-Val"
        ) if len(val_indices) > 0 else None
        test_ds = ParquetIterableDS(
            self.single_parquet_path,
            feature_cols,
            target_col=target_col,
            batch_size=data_loading_batch_size,
            indices=test_indices,
            shuffle=False,
            seed=44,
            name="IO-Test"
        ) if len(test_indices) > 0 else None

        # DataLoaders with multi-worker prefetch + pinned memory for fast GPU transfer
        nw = self.n_workers
        pfact = getattr(self, "prefetch_factor", 2)
        def make_loader(ds):
            return DataLoader(
                ds,
                batch_size=None,                  # dataset yields ready-made batches
                num_workers=int(nw),
                pin_memory=True,
                persistent_workers=(int(nw) > 0),
                prefetch_factor=(pfact if int(nw) > 0 else None)
            )

        train_loader = make_loader(train_ds)
        val_loader = make_loader(val_ds) if val_ds is not None else None
        test_loader = make_loader(test_ds) if test_ds is not None else None

        # Train
        if self.model_name == 'tabformer':
            # Ensure correct input dimension (if model not yet built)
            in_dim = len(feature_cols)
            if not isinstance(self.__model, FTTransformer) or self.__model.tokenizer.embedding.in_features != in_dim:
                self.__model = FTTransformer(num_features=in_dim)
            self.__model.to(self.device)

            optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            best_val_rmse = float('inf')
            patience_counter = 0
            best_model_state = None
            global_step = 0
            history = []
            
            batch_loss_csv = os.path.join(output_metrics_dir, "batch_losses.csv")
            if not os.path.exists(batch_loss_csv):
                with open(batch_loss_csv, "w", encoding="utf-8") as f:
                    f.write("epoch,batch_index,loss\n")
            

            for epoch in range(epochs):
                # Train pass
                self.__model.train()
                train_loss_sum = 0.0
                samples_processed = 0

                total_batches_est = max(1, getattr(train_loader.dataset, "total_batches", None) or len(train_loader))
                train_bar = tqdm(enumerate(train_loader), total=total_batches_est,
                                 desc=f"Epoch {epoch+1}/{epochs} (Train)", leave=False)

                for bi, (X_batch, y_batch) in train_bar:
                    progress_pct = min(100.0, 100.0 * (bi + 1) / max(1, total_batches_est))
                    num_samples = int(len(X_batch))
                    num_mini_batches = max(1, (num_samples + model_batch_size - 1) // model_batch_size)

                    batch_loss_sum = 0.0  # accumulate (sum of loss * mini_batch_size)
                    batch_samples = 0

                    mini_bar = tqdm(range(0, num_samples, model_batch_size),
                                    total=num_mini_batches,
                                    desc=f"  Mini-batches ({min(bi+1, total_batches_est)}/{total_batches_est})",
                                    leave=False)
                    for start in mini_bar:
                        end = min(start + model_batch_size, num_samples)
                        mini_X = X_batch[start:end].to(self.device, non_blocking=True)
                        mini_y = y_batch[start:end].to(self.device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)
                        outputs = self.__model(mini_X)
                        loss = criterion(outputs, mini_y)
                        loss.backward()
                        optimizer.step()

                        li = float(loss.detach().item())
                        batch_loss_sum += li * mini_X.size(0)
                        batch_samples += mini_X.size(0)
                        global_step += 1

                        if writer and (global_step % log_every == 0):
                            writer.add_scalar('Batch/train_loss', li, global_step)

                        mini_bar.set_postfix(
                            loss=f"{li:.6f}",
                            samples=f"{start+1}-{end}/{num_samples}"
                        )

                    # Finished this outer batch: compute average (per-sample) loss for the batch
                    batch_avg_loss = batch_loss_sum / max(1, batch_samples)
                    train_loss_sum += batch_loss_sum
                    samples_processed += batch_samples

                    # Write one line per outer batch
                    with open(batch_loss_csv, "a", encoding="utf-8") as f:
                        f.write(f"{epoch+1},{bi+1},{batch_avg_loss:.8f}\n")

                    avg_train_loss_running = train_loss_sum / max(1, samples_processed)
                    train_bar.set_postfix(progress=f"{progress_pct:5.1f}%", loss=f"{avg_train_loss_running:.6f}")

                avg_train_loss = train_loss_sum / max(1, samples_processed)

                # Validation
                val_rmse = None
                val_r2 = None
                if val_loader is not None:
                    self.__model.eval()
                    preds_all, trues_all = [], []

                    total_val_batches_est = max(1, getattr(val_loader.dataset, "total_batches", None) or len(val_loader))
                    val_bar = tqdm(enumerate(val_loader), total=total_val_batches_est,
                                   desc=f"Epoch {epoch+1}/{epochs} (Val)", leave=False)

                    with torch.no_grad():
                        for vbi, (Xb, yb) in val_bar:
                            progress_pct = min(100.0, 100.0 * (vbi + 1) / max(1, total_val_batches_est))

                            ns = int(len(Xb))
                            nmb = max(1, (ns + model_batch_size - 1) // model_batch_size)
                            mini_bar = tqdm(range(0, ns, model_batch_size), total=nmb,
                                            desc=f"  Val mini-batches ({min(vbi+1, total_val_batches_est)}/{total_val_batches_est})",
                                            leave=False)
                            for s in mini_bar:
                                e = min(s + model_batch_size, ns)
                                disp_start = min(s + 1, ns)
                                disp_end = min(e, ns)

                                mini_X = Xb[s:e].to(self.device, non_blocking=True)
                                mini_y = yb[s:e].to(self.device, non_blocking=True)
                                out = self.__model(mini_X)
                                preds_all.append(out.detach().cpu().numpy())
                                trues_all.append(mini_y.detach().cpu().numpy())

                                mini_bar.set_postfix(samples=f"{disp_start}-{disp_end}/{ns}")

                            val_bar.set_postfix(progress=f"{progress_pct:5.1f}%")

                    if preds_all:
                        y_pred = np.vstack(preds_all).reshape(-1)
                        y_true = np.vstack(trues_all).reshape(-1)
                        mse = mean_squared_error(y_true, y_pred)
                        val_rmse = float(np.sqrt(mse))
                        val_r2 = float(r2_score(y_true, y_pred))

                        # Early stopping
                        if val_rmse < best_val_rmse:
                            best_val_rmse = val_rmse
                            patience_counter = 0
                            best_model_state = {
                                'epoch': epoch,
                                'model_state_dict': self.__model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_rmse': val_rmse,
                                'val_r2': val_r2
                            }
                            best_model_name = f"best_model_epoch_{epoch}"
                            os.makedirs(os.path.dirname(output_metrics_dir) or ".", exist_ok=True)
                            torch.save(best_model_state, f"{output_metrics_dir}/{best_model_name}.pth")
                            print(f"✓ Saved best model at epoch {epoch+1} (val_rmse={val_rmse:.4f})")
                        else:
                            patience_counter += 1
                            if patience_counter >= early_stopping_patience:
                                print(f"Early stopping after epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                                break

                # Log epoch
                if writer:
                    writer.add_scalar('Epoch/train_loss', float(avg_train_loss), epoch)
                    if val_rmse is not None:
                        writer.add_scalar('Epoch/val_rmse', float(val_rmse), epoch)
                        writer.add_scalar('Epoch/val_r2', float(val_r2), epoch)

                history.append({
                    'epoch': epoch + 1,
                    'train_loss': float(avg_train_loss),
                    'val_loss': float(mse) if val_rmse is not None else None,
                    'val_rmse': float(val_rmse) if val_rmse is not None else None,
                    'val_r2': float(val_r2) if val_r2 is not None else None,
                })

            # Load best model if available
            if best_model_state is not None:
                print(f"Loading best model from epoch {best_model_state['epoch']+1}")
                self.__model.load_state_dict(best_model_state['model_state_dict'])

            # Test
            if test_loader is not None:
                self.__model.eval()
                preds_all, trues_all = [], []
                total_test_batches_est = max(1, getattr(test_loader.dataset, "total_batches", None) or len(test_loader))
                test_bar = tqdm(enumerate(test_loader), total=total_test_batches_est, desc="Testing", leave=False)
                with torch.no_grad():
                    for tbi, (Xb, yb) in test_bar:
                        progress_pct = min(100.0, 100.0 * (tbi + 1) / max(1, total_test_batches_est))
                        ns = int(len(Xb))
                        nmb = max(1, (ns + model_batch_size - 1) // model_batch_size)
                        mini_bar = tqdm(range(0, ns, model_batch_size), total=nmb,
                                        desc=f"  Test mini-batches ({min(tbi+1, total_test_batches_est)}/{total_test_batches_est})",
                                        leave=False)
                        for start in mini_bar:
                            end = min(start + model_batch_size, ns)
                            mini_X = Xb[start:end].to(self.device, non_blocking=True)
                            mini_y = yb[start:end].to(self.device, non_blocking=True)
                            out = self.__model(mini_X)
                            preds_all.append(out.detach().cpu().numpy())
                            trues_all.append(mini_y.detach().cpu().numpy())
                        test_bar.set_postfix(progress=f"{progress_pct:5.1f}%")

                if preds_all:
                    y_pred = np.vstack(preds_all).reshape(-1)
                    y_true = np.vstack(trues_all).reshape(-1)
                    test_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    test_r2 = float(r2_score(y_true, y_pred))
                    print(f"Test -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
                    if writer:
                        writer.add_scalar('Test/rmse', test_rmse, 0)
                        writer.add_scalar('Test/r2', test_r2, 0)
                    self.y_test = y_true
                    self.y_test_pred = y_pred

        elif self.model_name in ['catboost', 'xgboost', 'random_forest', 'decision_tree', 'gradient_boosting', 'tabpfn']:
            # Classic ML: one-pass fit over streaming batches
            from sklearn.metrics import mean_squared_error, r2_score
            history = []

            if self.model_name == 'catboost' and hasattr(self.__model, 'set_params'):
                self.__model.set_params(verbose=False)

            total_batches_est = max(1, getattr(train_loader.dataset, "total_batches", None) or len(train_loader))
            train_bar = tqdm(enumerate(train_loader), total=total_batches_est,
                             desc=f"Epoch 1/1 (Train)", leave=False)
            train_losses = []

            for bi, (Xb, yb) in train_bar:
                progress_pct = min(100.0, 100.0 * (bi + 1) / max(1, total_batches_est))
                X_np = Xb.numpy()
                y_np = yb.numpy().reshape(-1)

                if self.model_name == 'catboost':
                    from catboost import Pool
                    self.__model.fit(Pool(X_np, y_np), verbose=False)
                else:
                    self.__model.fit(X_np, y_np)

                y_pred = self.__model.predict(X_np)
                mse = mean_squared_error(y_np, y_pred)
                train_losses.append(mse)
                train_bar.set_postfix(progress=f"{progress_pct:5.1f}%", mse=f"{mse:.6f}")

            avg_train_rmse = float(np.sqrt(np.mean(train_losses))) if train_losses else None

            val_rmse = val_r2 = None
            if val_loader is not None:
                preds, trues = [], []
                total_val_batches_est = max(1, getattr(val_loader.dataset, "total_batches", None) or len(val_loader))
                val_bar = tqdm(enumerate(val_loader), total=total_val_batches_est, desc="Validating", leave=False)
                for vbi, (Xb, yb) in val_bar:
                    progress_pct = min(100.0, 100.0 * (vbi + 1) / max(1, total_val_batches_est))
                    X_np = Xb.numpy(); y_np = yb.numpy().reshape(-1)
                    preds.append(self.__model.predict(X_np))
                    trues.append(y_np)
                    val_bar.set_postfix(progress=f"{progress_pct:5.1f}%")
                if preds:
                    y_pred = np.concatenate(preds, axis=0)
                    y_true = np.concatenate(trues, axis=0)
                    val_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    val_r2 = float(r2_score(y_true, y_pred))

            history.append({
                "epoch": 1,
                "train_rmse": float(avg_train_rmse) if avg_train_rmse is not None else None,
                "val_rmse": float(val_rmse) if val_rmse is not None else None,
                "val_r2": float(val_r2) if val_r2 is not None else None,
            })
            msg = f"🟢 Epoch 1/1 • "
            if avg_train_rmse is not None:
                msg += f"🎯 train RMSE={avg_train_rmse:.4f}"
            if val_rmse is not None and val_r2 is not None:
                msg += f" • 🧪 val RMSE={val_rmse:.4f} R²={val_r2:.4f}"
            print(msg)

            if test_loader is not None:
                preds_te, trues_te = [], []
                total_test_batches_est = max(1, getattr(test_loader.dataset, "total_batches", None) or len(test_loader))
                test_bar = tqdm(enumerate(test_loader), total=total_test_batches_est, desc="Testing", leave=False)
                for tbi, (Xb, yb) in test_bar:
                    progress_pct = min(100.0, 100.0 * (tbi + 1) / max(1, total_test_batches_est))
                    X_np = Xb.numpy(); y_np = yb.numpy().reshape(-1)
                    preds_te.append(self.__model.predict(X_np))
                    trues_te.append(y_np)
                    test_bar.set_postfix(progress=f"{progress_pct:5.1f}%")
                if preds_te:
                    y_pred_te = np.concatenate(preds_te, axis=0)
                    y_true_te = np.concatenate(trues_te, axis=0)
                    test_rmse = float(np.sqrt(mean_squared_error(y_true_te, y_pred_te)))
                    test_r2 = float(r2_score(y_true_te, y_pred_te))
                    print(f"📊 Test -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

        else:
            raise ValueError(f"Model type '{self.model_name}' not supported for streaming training.")

        # Save history to CSV
        csv_path = os.path.join(output_metrics_dir, "streaming_training_history.csv")
        pd.DataFrame(history).to_csv(csv_path, index=False)
        print(f"Saved history -> {csv_path}")

        if writer:
            writer.close()
            print(f"✓ TensorBoard logs saved to {os.path.join(output_metrics_dir, 'tensorboard')}")

        return history
      
    
    
    def train_on_single_parquet_file_streaming_v6(
        self,
        epochs: int = 50,
        data_loading_batch_size: int = 8_388_608,  # Large batch for efficient disk I/O
        model_batch_size: int | None = 256,   # Smaller batch for model training
        target_col: str = 'y',
        exclude_cols: list | None = None,
        log_every: int = 10,
        early_stopping_patience: int = 10,
        output_metrics_dir: str | None = None,
        checkpoint_path: str = "best_model",
        enable_tensorboard: bool = True,
        lr: float = 1e-3
    ):
        """
        Train a model on a single parquet file with memory-efficient streaming.
        Fixes progress overflow: clamp percentages to [0,100] and compute total_batches accurately.
        """
        import os
        import torch
        import torch.nn as nn
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from sklearn.metrics import mean_squared_error, r2_score
        import pyarrow.parquet as pq
        import pyarrow as pa

        if not self.single_parquet_path:
            raise ValueError("single_parquet_path not set. This method requires a single parquet file.")

        # Validate/resolve batch sizes
        if model_batch_size is None or int(model_batch_size) <= 0:
            model_batch_size = int(getattr(self, "_Model__batch_size", 64) or 64)
        model_batch_size = int(model_batch_size)
        data_loading_batch_size = max(1, int(data_loading_batch_size))

        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./results_monitor")
        os.makedirs(output_metrics_dir, exist_ok=True)

        class ParquetLoader:
            """
            Streams a single Parquet file by row-group and yields batches of size 'batch_size'.
            Computes total_batches as sum over row-groups of ceil(n_rg / batch_size),
            so progress never exceeds 100%.
            """
            def __init__(
                self,
                parquet_file,
                feature_cols,
                target_col="y",
                batch_size=65536,
                indices=None,
                shuffle=True,
                seed=42,
            ):
                self.pf = pq.ParquetFile(parquet_file, memory_map=True)
                self.feature_cols = feature_cols
                self.target_col = target_col
                self.columns = feature_cols + [target_col]
                self.batch_size = int(batch_size)
                self.shuffle = shuffle
                self.seed = seed

                # Row-group offsets and sizes
                self.num_row_groups = self.pf.num_row_groups
                self.rg_offsets = []
                rg_sizes = []
                running = 0
                for rg in range(self.num_row_groups):
                    n = self.pf.metadata.row_group(rg).num_rows
                    self.rg_offsets.append((running, running + n))
                    rg_sizes.append(n)
                    running += n
                self.total_rows = running

                # Build selected indices per row-group
                if indices is None:
                    # All rows
                    self.selected_per_rg = {rg: None for rg in range(self.num_row_groups)}
                    per_rg_counts = rg_sizes[:]  # each RG fully used
                else:
                    idx = np.asarray(indices, dtype=np.int64)
                    idx.sort()
                    self.selected_per_rg = {}
                    per_rg_counts = []
                    for rg, (start, end) in enumerate(self.rg_offsets):
                        left = np.searchsorted(idx, start, side="left")
                        right = np.searchsorted(idx, end, side="left")
                        local = idx[left:right] - start  # local row indices in RG
                        if local.size > 0:
                            self.selected_per_rg[rg] = local
                            per_rg_counts.append(int(local.size))
                        else:
                            self.selected_per_rg[rg] = np.empty((0,), dtype=np.int64)
                            per_rg_counts.append(0)

                # Accurate totals for progress
                self.per_rg_counts = per_rg_counts
                self.total_selected = int(sum(per_rg_counts))
                # Sum of ceil per RG to match how we actually yield
                self.total_batches = int(sum((c + self.batch_size - 1) // self.batch_size for c in per_rg_counts if c > 0))
                # Guard against zero
                if self.total_batches <= 0:
                    self.total_batches = 1

            def __len__(self):
                return self.total_batches

            def _table_to_numpy(self, table: pa.Table):
                # Fast Arrow->NumPy conversion (avoid pandas when possible)
                try:
                    X_cols = [table[c].to_numpy(zero_copy_only=False) for c in self.feature_cols]
                    X = np.column_stack(X_cols).astype(np.float32, copy=False)
                    y = table[self.target_col].to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1, 1)
                    return X, y
                except Exception:
                    pdf = table.to_pandas()
                    X = pdf[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
                    y = pdf[[self.target_col]].to_numpy(dtype=np.float32, copy=False)
                    return X, y

            def __iter__(self):
                import random
                order = list(range(self.num_row_groups))
                if self.shuffle:
                    random.seed(self.seed)
                    random.shuffle(order)

                for rg in order:
                    # Skip empty RGs for the selection
                    local_idx = self.selected_per_rg.get(rg, None)
                    if local_idx is not None and isinstance(local_idx, np.ndarray) and local_idx.size == 0:
                        continue

                    # Read RG once
                    tbl = self.pf.read_row_group(rg, columns=self.columns)

                    # Subset rows for this RG if indices provided
                    if local_idx is not None:
                        if self.shuffle and local_idx.size > 0:
                            rng = np.random.default_rng(self.seed + rg)
                            local_idx = local_idx.copy()
                            rng.shuffle(local_idx)
                        if local_idx.size > 0:
                            tbl = tbl.take(pa.array(local_idx, type=pa.int64()))
                        else:
                            continue  # nothing selected in this RG

                    n = tbl.num_rows
                    if n == 0:
                        continue

                    # Yield in chunks of batch_size
                    for start in range(0, n, self.batch_size):
                        end = min(start + self.batch_size, n)
                        batch_tbl = tbl.slice(start, end - start)
                        X, y = self._table_to_numpy(batch_tbl)
                        yield torch.from_numpy(X), torch.from_numpy(y)

        # Setup TensorBoard logging if enabled
        writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tensorboard_dir = os.path.join(output_metrics_dir, "tensorboard")
                os.makedirs(tensorboard_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=tensorboard_dir)
                print(f"TensorBoard logging -> {tensorboard_dir}")
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")

        # Infer features from Parquet schema
        pf = pq.ParquetFile(self.single_parquet_path)
        schema = pf.schema_arrow
        col_names = list(schema.names)

        if target_col not in col_names:
            raise ValueError(f"Target column '{target_col}' not found in parquet file.")

        drop_meta = {"date", "year", "month"}
        feature_cols = [c for c in col_names if c not in {target_col, *drop_meta}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        self.feature_names_ = feature_cols

        # Count total rows
        total_rows = sum(pf.metadata.row_group(rg).num_rows for rg in range(pf.num_row_groups))
        print(f"Rows: {total_rows:,} | Features: {len(feature_cols)}")

        # Train/Val/Test split over row indices
        from sklearn.model_selection import train_test_split
        all_indices = np.arange(total_rows)

        # 1) Train vs Test
        train_full_indices, test_indices = train_test_split(
            all_indices,
            train_size=self.train_percent,
            random_state=42
        )

        # 2) From training, take validation percentage
        if self.validation_percentage and self.validation_percentage > 0:
            # clamp to avoid edge cases
            val_size = min(0.999, float(self.validation_percentage))
            train_indices, val_indices = train_test_split(
                train_full_indices,
                test_size=val_size,
                random_state=42
            )
        else:
            train_indices = train_full_indices
            val_indices = np.array([], dtype=int)

        print(f"Split: train={len(train_indices):,}, val={len(val_indices):,}, test={len(test_indices):,}")

        # Build loaders with accurate total_batches
        train_loader = ParquetLoader(
            self.single_parquet_path,
            feature_cols,
            target_col=target_col,
            batch_size=data_loading_batch_size,
            indices=train_indices,
            shuffle=True
        )
        val_loader = ParquetLoader(
            self.single_parquet_path,
            feature_cols,
            target_col=target_col,
            batch_size=data_loading_batch_size,
            indices=val_indices,
            shuffle=False
        ) if len(val_indices) > 0 else None
        test_loader = ParquetLoader(
            self.single_parquet_path,
            feature_cols,
            target_col=target_col,
            batch_size=data_loading_batch_size,
            indices=test_indices,
            shuffle=False
        ) if len(test_indices) > 0 else None

        # Train
        if self.model_name == 'tabformer':
            # Ensure correct input dimension (if model not yet built)
            in_dim = len(feature_cols)
            if not isinstance(self.__model, FTTransformer) or self.__model.tokenizer.embedding.in_features != in_dim:
                self.__model = FTTransformer(num_features=in_dim)
            self.__model.to(self.device)

            optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            best_val_rmse = float('inf')
            patience_counter = 0
            best_model_state = None
            global_step = 0
            history = []

            for epoch in range(epochs):
                # Train pass
                self.__model.train()
                train_loss_sum = 0.0
                samples_processed = 0

                total_batches_est = max(1, getattr(train_loader, "total_batches", None) or len(train_loader))
                train_bar = tqdm(enumerate(train_loader), total=total_batches_est,
                                desc=f"Epoch {epoch+1}/{epochs} (Train)", leave=False)

                for bi, (X_batch, y_batch) in train_bar:
                    # Clamp progress to [0, 100]
                    progress_pct = min(100.0, 100.0 * (bi + 1) / max(1, total_batches_est))

                    num_samples = int(len(X_batch))
                    # Number of model mini-batches in this data batch
                    num_mini_batches = max(1, (num_samples + model_batch_size - 1) // model_batch_size)

                    # Inner mini-batch progress
                    mini_bar = tqdm(range(0, num_samples, model_batch_size),
                                    total=num_mini_batches,
                                    desc=f"  Mini-batches ({min(bi+1, total_batches_est)}/{total_batches_est})",
                                    leave=False)
                    for start in mini_bar:
                        end = min(start + model_batch_size, num_samples)
                        # Clamp for display
                        disp_start = min(start + 1, num_samples)
                        disp_end = min(end, num_samples)

                        mini_X = X_batch[start:end].to(self.device, non_blocking=True)
                        mini_y = y_batch[start:end].to(self.device, non_blocking=True)

                        optimizer.zero_grad()
                        outputs = self.__model(mini_X)
                        loss = criterion(outputs, mini_y)
                        loss.backward()
                        optimizer.step()

                        batch_loss = loss.item() * mini_X.size(0)
                        train_loss_sum += batch_loss
                        samples_processed += mini_X.size(0)
                        global_step += 1

                        # Update mini progress (loss per-sample to keep stable)
                        per_sample_loss = batch_loss / max(1, mini_X.size(0))
                        mini_bar.set_postfix(
                            loss=f"{per_sample_loss:.6f}",
                            samples=f"{disp_start}-{disp_end}/{num_samples}"
                        )

                        if writer and (global_step % log_every == 0):
                            writer.add_scalar('Batch/train_loss', per_sample_loss, global_step)

                    # Update outer bar
                    avg_train_loss = train_loss_sum / max(1, samples_processed)
                    train_bar.set_postfix(progress=f"{progress_pct:5.1f}%", loss=f"{avg_train_loss:.6f}")

                # Epoch train loss
                avg_train_loss = train_loss_sum / max(1, samples_processed)

                # Validation
                val_rmse = None
                val_r2 = None
                if val_loader is not None:
                    self.__model.eval()
                    preds_all, trues_all = [], []

                    total_val_batches_est = max(1, getattr(val_loader, "total_batches", None) or len(val_loader))
                    val_bar = tqdm(enumerate(val_loader), total=total_val_batches_est,
                                desc=f"Epoch {epoch+1}/{epochs} (Val)", leave=False)

                    with torch.no_grad():
                        for vbi, (Xb, yb) in val_bar:
                            progress_pct = min(100.0, 100.0 * (vbi + 1) / max(1, total_val_batches_est))

                            ns = int(len(Xb))
                            nmb = max(1, (ns + model_batch_size - 1) // model_batch_size)
                            mini_bar = tqdm(range(0, ns, model_batch_size), total=nmb,
                                            desc=f"  Val mini-batches ({min(vbi+1, total_val_batches_est)}/{total_val_batches_est})",
                                            leave=False)
                            for start in mini_bar:
                                end = min(start + model_batch_size, ns)
                                disp_start = min(start + 1, ns)
                                disp_end = min(end, ns)

                                mini_X = Xb[start:end].to(self.device, non_blocking=True)
                                mini_y = yb[start:end].to(self.device, non_blocking=True)
                                out = self.__model(mini_X)

                                preds_all.append(out.detach().cpu().numpy())
                                trues_all.append(mini_y.detach().cpu().numpy())

                                mini_bar.set_postfix(samples=f"{disp_start}-{disp_end}/{ns}")

                            val_bar.set_postfix(progress=f"{progress_pct:5.1f}%")

                    if preds_all:
                        y_pred = np.vstack(preds_all).reshape(-1)
                        y_true = np.vstack(trues_all).reshape(-1)
                        mse = mean_squared_error(y_true, y_pred)
                        val_rmse = float(np.sqrt(mse))
                        val_r2 = float(r2_score(y_true, y_pred))

                        # Early stopping
                        if val_rmse < best_val_rmse:
                            best_val_rmse = val_rmse
                            patience_counter = 0
                            best_model_state = {
                                'epoch': epoch,
                                'model_state_dict': self.__model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_rmse': val_rmse,
                                'val_r2': val_r2
                            }
                            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                            torch.save(best_model_state, f"{checkpoint_path}.pth")
                            print(f"✓ Saved best model at epoch {epoch+1} (val_rmse={val_rmse:.4f})")
                        else:
                            patience_counter += 1
                            if patience_counter >= early_stopping_patience:
                                print(f"Early stopping after epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                                break

                # Log epoch
                if writer:
                    writer.add_scalar('Epoch/train_loss', float(avg_train_loss), epoch)
                    if val_rmse is not None:
                        writer.add_scalar('Epoch/val_rmse', float(val_rmse), epoch)
                        writer.add_scalar('Epoch/val_r2', float(val_r2), epoch)

                history.append({
                    'epoch': epoch + 1,
                    'train_loss': float(avg_train_loss),
                    'val_loss': float(mse) if val_rmse is not None else None,
                    'val_rmse': float(val_rmse) if val_rmse is not None else None,
                    'val_r2': float(val_r2) if val_r2 is not None else None,
                })

            # Load best model if available
            if best_model_state is not None:
                print(f"Loading best model from epoch {best_model_state['epoch']+1}")
                self.__model.load_state_dict(best_model_state['model_state_dict'])

            # Test
            if test_loader is not None:
                self.__model.eval()
                preds_all, trues_all = [], []
                total_test_batches_est = max(1, getattr(test_loader, "total_batches", None) or len(test_loader))
                test_bar = tqdm(enumerate(test_loader), total=total_test_batches_est, desc="Testing", leave=False)
                with torch.no_grad():
                    for tbi, (Xb, yb) in test_bar:
                        progress_pct = min(100.0, 100.0 * (tbi + 1) / max(1, total_test_batches_est))

                        ns = int(len(Xb))
                        nmb = max(1, (ns + model_batch_size - 1) // model_batch_size)
                        mini_bar = tqdm(range(0, ns, model_batch_size), total=nmb,
                                        desc=f"  Test mini-batches ({min(tbi+1, total_test_batches_est)}/{total_test_batches_est})",
                                        leave=False)
                        for start in mini_bar:
                            end = min(start + model_batch_size, ns)
                            disp_start = min(start + 1, ns)
                            disp_end = min(end, ns)

                            mini_X = Xb[start:end].to(self.device, non_blocking=True)
                            mini_y = yb[start:end].to(self.device, non_blocking=True)
                            out = self.__model(mini_X)

                            preds_all.append(out.detach().cpu().numpy())
                            trues_all.append(mini_y.detach().cpu().numpy())

                            mini_bar.set_postfix(samples=f"{disp_start}-{disp_end}/{ns}")

                        test_bar.set_postfix(progress=f"{progress_pct:5.1f}%")

                if preds_all:
                    y_pred = np.vstack(preds_all).reshape(-1)
                    y_true = np.vstack(trues_all).reshape(-1)
                    test_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    test_r2 = float(r2_score(y_true, y_pred))
                    print(f"Test -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
                    if writer:
                        writer.add_scalar('Test/rmse', test_rmse, 0)
                        writer.add_scalar('Test/r2', test_r2, 0)
                    self.y_test = y_true
                    self.y_test_pred = y_pred

        elif self.model_name in ['catboost', 'xgboost', 'random_forest', 'decision_tree', 'gradient_boosting']:
            # Classic ML: single-pass fit over streaming batches
            from sklearn.metrics import mean_squared_error, r2_score
            history = []

            # Quiet CatBoost
            if self.model_name == 'catboost' and hasattr(self.__model, 'set_params'):
                self.__model.set_params(verbose=False)

            # Train
            total_batches_est = max(1, getattr(train_loader, "total_batches", None) or len(train_loader))
            train_bar = tqdm(enumerate(train_loader), total=total_batches_est,
                            desc=f"Epoch 1/1 (Train)", leave=False)
            train_losses = []

            for bi, (Xb, yb) in train_bar:
                progress_pct = min(100.0, 100.0 * (bi + 1) / max(1, total_batches_est))

                X_np = Xb.numpy()
                y_np = yb.numpy().reshape(-1)

                # Fit/update on this batch
                if self.model_name == 'catboost':
                    from catboost import Pool
                    self.__model.fit(Pool(X_np, y_np), verbose=False)
                else:
                    self.__model.fit(X_np, y_np)

                y_pred = self.__model.predict(X_np)
                mse = mean_squared_error(y_np, y_pred)
                train_losses.append(mse)

                train_bar.set_postfix(progress=f"{progress_pct:5.1f}%", mse=f"{mse:.6f}")

            avg_train_rmse = float(np.sqrt(np.mean(train_losses))) if train_losses else None

            # Validation
            val_rmse = None
            val_r2 = None
            if val_loader is not None:
                preds, trues = [], []
                total_val_batches_est = max(1, getattr(val_loader, "total_batches", None) or len(val_loader))
                val_bar = tqdm(enumerate(val_loader), total=total_val_batches_est,
                            desc=f"Validating", leave=False)
                for vbi, (Xb, yb) in val_bar:
                    progress_pct = min(100.0, 100.0 * (vbi + 1) / max(1, total_val_batches_est))
                    X_np = Xb.numpy()
                    y_np = yb.numpy().reshape(-1)
                    preds.append(self.__model.predict(X_np))
                    trues.append(y_np)
                    val_bar.set_postfix(progress=f"{progress_pct:5.1f}%")
                if preds:
                    y_pred = np.concatenate(preds)
                    y_true = np.concatenate(trues)
                    val_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    val_r2 = float(r2_score(y_true, y_pred))

            history = [{
                "epoch": 1,
                "train_rmse": avg_train_rmse if avg_train_rmse is not None else np.nan,
                "val_rmse": val_rmse if val_rmse is not None else np.nan,
                "val_r2": val_r2 if val_r2 is not None else np.nan,
            }]

            # Test
            if test_loader is not None:
                preds, trues = [], []
                total_test_batches_est = max(1, getattr(test_loader, "total_batches", None) or len(test_loader))
                test_bar = tqdm(enumerate(test_loader), total=total_test_batches_est, desc="Testing", leave=False)
                for tbi, (Xb, yb) in test_bar:
                    progress_pct = min(100.0, 100.0 * (tbi + 1) / max(1, total_test_batches_est))
                    X_np = Xb.numpy()
                    y_np = yb.numpy().reshape(-1)
                    preds.append(self.__model.predict(X_np))
                    trues.append(y_np)
                    test_bar.set_postfix(progress=f"{progress_pct:5.1f}%")
                if preds:
                    y_pred = np.concatenate(preds)
                    y_true = np.concatenate(trues)
                    test_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    test_r2 = float(r2_score(y_true, y_pred))
                    print(f"Test -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

        else:
            raise ValueError(f"Model type '{self.model_name}' not supported for streaming training.")

        # Save history to CSV
        csv_path = os.path.join(output_metrics_dir, "streaming_training_history.csv")
        pd.DataFrame(history).to_csv(csv_path, index=False)
        print(f"Saved history -> {csv_path}")

        if writer:
            writer.close()
            print(f"✓ TensorBoard logs saved to {os.path.join(output_metrics_dir, 'tensorboard')}")

        return history
    
    
    def train_on_single_parquet_file_streaming_v2(
        self,
        epochs: int = 50,
        data_loading_batch_size: int = 65536,  # Large batch for efficient disk I/O
        model_batch_size: int | None = None,   # Smaller batch for model training
        target_col: str = 'y',
        exclude_cols: list | None = None,
        log_every: int = 10,
        early_stopping_patience: int = 10,
        output_metrics_dir: str | None = None,
        checkpoint_path: str = "best_model",
        enable_tensorboard: bool = True,
        lr: float = 1e-3
    ):
        """
        Train a model on a single parquet file with memory-efficient streaming.
        
        This method efficiently handles large parquet files by:
        1. Reading data from disk in large batches (data_loading_batch_size)
        2. Processing model training in smaller batches (model_batch_size)
        
        Parameters:
        -----------
        epochs: int
            Number of training epochs
        data_loading_batch_size: int
            Batch size for loading data from disk (larger = more efficient I/O)
        model_batch_size: int or None
            Batch size for model training (if None, uses self.__batch_size)
        target_col: str
            Name of the target column
        exclude_cols: list or None
            Columns to exclude from features
        log_every: int
            How often to log batch information
        early_stopping_patience: int
            Number of epochs to wait for improvement before stopping
        output_metrics_dir: str or None
            Directory to save metrics and model checkpoints
        checkpoint_path: str
            Path to save best model checkpoint
        enable_tensorboard: bool
            Whether to enable TensorBoard logging
        lr: float
            Learning rate for optimizers
        
        Returns:
        --------
        list
            Training history
        """
        import os
        import torch
        import torch.nn as nn
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from sklearn.metrics import mean_squared_error, r2_score
        import pyarrow.parquet as pq
        
        if not self.single_parquet_path:
            raise ValueError("single_parquet_path not set. This method requires a single parquet file.")
            
        if model_batch_size is None:
            model_batch_size = getattr(self, "__batch_size", 64)
            
        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./results_monitor")
        
        # Ensure output directory exists
        os.makedirs(output_metrics_dir, exist_ok=True)
        
        import pyarrow.parquet as pq
        import pyarrow as pa
        # Create ParquetLoader for streaming data from a single parquet file
        class ParquetLoader:
            def __init__(
                self,
                parquet_file,
                feature_cols,
                target_col="y",
                batch_size=65536,
                indices=None,
                shuffle=True,
                seed=42,
            ):
                # Fast memory-mapped reads
                self.pf = pq.ParquetFile(parquet_file, memory_map=True)
                self.feature_cols = feature_cols
                self.target_col = target_col
                self.columns = feature_cols + [target_col]
                self.batch_size = int(batch_size)
                self.shuffle = shuffle
                self.seed = seed

                # Build row-group metadata
                self.num_row_groups = self.pf.num_row_groups
                rg_sizes = [self.pf.metadata.row_group(rg).num_rows for rg in range(self.num_row_groups)]
                self.rg_offsets = []
                running = 0
                for n in rg_sizes:
                    self.rg_offsets.append((running, running + n))  # [global_start, global_end)
                    running += n
                self.total_rows = running

                # Resolve selected indices (row-level split) and map to row groups
                if indices is None:
                    # Use all rows; we’ll stream per row-group
                    self.selected_per_rg = {rg: None for rg in range(self.num_row_groups)}
                    self.total_selected = self.total_rows
                else:
                    idx = np.asarray(indices, dtype=np.int64)
                    idx.sort()  # important for fast slicing
                    self.selected_per_rg = {}
                    self.total_selected = 0
                    for rg, (start, end) in enumerate(self.rg_offsets):
                        # indices that fall in this row-group
                        left = np.searchsorted(idx, start, side="left")
                        right = np.searchsorted(idx, end, side="left")
                        local = idx[left:right] - start  # convert to local indices
                        if local.size > 0:
                            self.selected_per_rg[rg] = local
                            self.total_selected += int(local.size)

                # Estimated batches (for tqdm %)
                self.total_batches = max(1, (self.total_selected + self.batch_size - 1) // self.batch_size)

            def __len__(self):
                return self.total_batches

            def _table_to_numpy(self, table: pa.Table):
                # Avoid pandas conversion; fallback if not supported
                try:
                    X_cols = []
                    for c in self.feature_cols:
                        ca = table[c]  # ChunkedArray
                        X_cols.append(ca.to_numpy(zero_copy_only=False))
                    X = np.column_stack(X_cols).astype(np.float32, copy=False)
                    y = table[self.target_col].to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1, 1)
                    return X, y
                except Exception:
                    # Fallback to pandas (slower)
                    pdf = table.to_pandas()
                    X = pdf[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
                    y = pdf[[self.target_col]].to_numpy(dtype=np.float32, copy=False)
                    return X, y

            def __iter__(self):
                import random
                order = list(range(self.num_row_groups))
                if self.shuffle:
                    random.seed(self.seed)
                    random.shuffle(order)

                for rg in order:
                    # Quick skip if no rows from this RG are selected
                    local_idx = self.selected_per_rg.get(rg, None)
                    if local_idx is not None and isinstance(local_idx, np.ndarray) and local_idx.size == 0:
                        continue

                    # Read RG once
                    tbl = self.pf.read_row_group(rg, columns=self.columns)

                    # Subset to selected rows for this RG (if indices specified)
                    if local_idx is not None:
                        if self.shuffle:
                            # Shuffle within RG to avoid i.i.d. bias
                            rng = np.random.default_rng(self.seed + rg)
                            local_idx = local_idx.copy()
                            rng.shuffle(local_idx)
                        tbl = tbl.take(pa.array(local_idx, type=pa.int64()))

                    # Yield in batches directly from Arrow without pandas
                    n = tbl.num_rows
                    if n == 0:
                        continue
                    for start in range(0, n, self.batch_size):
                        end = min(start + self.batch_size, n)
                        batch_tbl = tbl.slice(start, end - start)
                        X, y = self._table_to_numpy(batch_tbl)
                        yield torch.from_numpy(X), torch.from_numpy(y)
        
        # Setup TensorBoard logging if enabled
        writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tensorboard_dir = os.path.join(output_metrics_dir, "tensorboard")
                os.makedirs(tensorboard_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=tensorboard_dir)
                print(f"✅ TensorBoard logging enabled at {tensorboard_dir}")
            except ImportError:
                print("⚠️ TensorBoard not available. Install with: pip install tensorboard")
        
        # Extract feature columns and make train/val/test split
        pf = pq.ParquetFile(self.single_parquet_path)
        schema = pf.schema_arrow
        col_names = list(schema.names)
        
        if target_col not in col_names:
            raise ValueError(f"Target column '{target_col}' not found in parquet file.")
            
        # Determine feature columns
        drop_meta = {"date", "year", "month"}
        feature_cols = [c for c in col_names if c not in {target_col, *drop_meta}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        
        # Store feature names
        self.feature_names_ = feature_cols
        
        # Count total rows
        total_rows = sum(pf.metadata.row_group(rg).num_rows for rg in range(pf.num_row_groups))
        print(f"📊 Total rows in parquet file: {total_rows:,}")
        print(f"📋 Features: {len(feature_cols)} columns")
        
        # Create train/val/test split
        from sklearn.model_selection import train_test_split
        
        all_indices = np.arange(total_rows)
        
        # First split: train vs rest
        train_indices, remaining_indices = train_test_split(
            all_indices, 
            train_size=self.train_percent,
            random_state=42
        )
        
        # Second split: val vs test (if needed)
        if len(remaining_indices) > 0 and self.validation_percentage > 0:
            # Calculate relative validation size with safety check
            val_size_relative = min(0.999, self.validation_percentage / (1 - self.train_percent))
            
            val_indices, test_indices = train_test_split(
                remaining_indices,
                train_size=val_size_relative,
                random_state=42
            )
        else:
            val_indices = []
            test_indices = remaining_indices
        
        print(f"🔄 Data split: {len(train_indices):,} train, {len(val_indices):,} validation, {len(test_indices):,} test samples")
        
        # Create data loaders
        train_loader = ParquetLoader(
            self.single_parquet_path,
            feature_cols,
            target_col=target_col,
            batch_size=data_loading_batch_size,
            indices=train_indices,
            shuffle=True
        )
        
        val_loader = None
        if len(val_indices) > 0:
            val_loader = ParquetLoader(
                self.single_parquet_path,
                feature_cols,
                target_col=target_col,
                batch_size=data_loading_batch_size,
                indices=val_indices,
                shuffle=False
            )
        
        test_loader = None
        if len(test_indices) > 0:
            test_loader = ParquetLoader(
                self.single_parquet_path,
                feature_cols,
                target_col=target_col,
                batch_size=data_loading_batch_size,
                indices=test_indices,
                shuffle=False
            )
        
        # Check model type and train accordingly
        if self.model_name == 'tabformer':
            # TabFormer (PyTorch) model
            
            self.__model.to(self.device)
            
            # Setup optimizer and loss
            optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            # Training tracking variables
            best_val_rmse = float('inf')
            patience_counter = 0
            best_model_state = None
            global_step = 0
            history = []
            print(f'model batch size: {model_batch_size}')
            # Training loop
            for epoch in range(epochs):
                # Training phase
                self.__model.train()
                train_loss = 0.0
                samples_processed = 0
                
                # Process each data loading batch
                train_bar = tqdm(enumerate(train_loader), total=train_loader.total_batches, 
                            desc=f"Epoch {epoch+1}/{epochs} (Train)", leave=False)
                
                for batch_idx, (X_batch, y_batch) in train_bar:
                    # Calculate and display percentage progress
                    progress_pct = 100.0 * (batch_idx + 1) / train_loader.total_batches
                    
                    # Calculate number of mini-batches for this data batch
                    num_samples = len(X_batch)
                    num_mini_batches = (num_samples + model_batch_size - 1) // model_batch_size
                    
                    # Process in model-sized mini-batches with progress tracking
                    mini_batch_bar = tqdm(
                        range(0, num_samples, model_batch_size), 
                        total=num_mini_batches,
                        desc=f"  Mini-batches ({batch_idx+1}/{train_loader.total_batches})", 
                        leave=False
                    )
                    
                    for mini_batch_start in mini_batch_bar:
                        mini_batch_end = min(mini_batch_start + model_batch_size, num_samples)
                        mini_X = X_batch[mini_batch_start:mini_batch_end].to(self.device)
                        mini_y = y_batch[mini_batch_start:mini_batch_end].to(self.device)
                        
                        # Forward pass
                        optimizer.zero_grad()
                        outputs = self.__model(mini_X)
                        loss = criterion(outputs, mini_y)
                        
                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()
                        
                        # Update statistics
                        batch_loss = loss.item() * len(mini_X)
                        train_loss += batch_loss
                        samples_processed += len(mini_X)
                        
                        # Update mini-batch progress bar
                        mini_batch_bar.set_postfix(
                            loss=f"{batch_loss/len(mini_X):.6f}",
                            samples=f"{mini_batch_start+1}-{mini_batch_end}/{num_samples}"
                        )
                        
                        # Log to TensorBoard
                        if writer and (global_step % log_every == 0):
                            writer.add_scalar('Batch/train_loss', loss.item(), global_step)
                        
                        global_step += 1
                    
                    # Update main progress bar after all mini-batches
                    train_bar.set_postfix(
                        progress=f"{progress_pct:.1f}%",
                        loss=f"{train_loss/max(1, samples_processed):.6f}"
                    )
                
                # Calculate average training loss
                avg_train_loss = train_loss / max(1, samples_processed)
                
                # Validation phase
                val_rmse = None
                val_r2 = None
                
                if val_loader:
                    self.__model.eval()
                    all_val_preds = []
                    all_val_targets = []
                    
                    with torch.no_grad():
                        val_bar = tqdm(enumerate(val_loader), total=val_loader.total_batches,
                                    desc=f"Epoch {epoch+1}/{epochs} (Val)", leave=False)
                        
                        for batch_idx, (X_batch, y_batch) in val_bar:
                            progress_pct = 100.0 * (batch_idx + 1) / val_loader.total_batches
                            
                            # Calculate number of mini-batches for this validation batch
                            num_samples = len(X_batch)
                            num_mini_batches = (num_samples + model_batch_size - 1) // model_batch_size
                            
                            # Process in mini-batches with progress tracking
                            mini_batch_bar = tqdm(
                                range(0, num_samples, model_batch_size), 
                                total=num_mini_batches,
                                desc=f"  Val mini-batches ({batch_idx+1}/{val_loader.total_batches})", 
                                leave=False
                            )
                            
                            for mini_batch_start in mini_batch_bar:
                                mini_batch_end = min(mini_batch_start + model_batch_size, num_samples)
                                mini_X = X_batch[mini_batch_start:mini_batch_end].to(self.device)
                                mini_y = y_batch[mini_batch_start:mini_batch_end].to(self.device)
                                
                                outputs = self.__model(mini_X)
                                
                                all_val_preds.append(outputs.cpu().numpy())
                                all_val_targets.append(mini_y.cpu().numpy())
                                
                                # Update mini-batch progress bar
                                mini_batch_bar.set_postfix(
                                    samples=f"{mini_batch_start+1}-{mini_batch_end}/{num_samples}"
                                )
                            
                            # Update validation bar
                            val_bar.set_postfix(progress=f"{progress_pct:.1f}%")
                    
                    # Calculate validation metrics
                    if all_val_preds:
                        val_preds = np.vstack(all_val_preds).reshape(-1)
                        val_targets = np.vstack(all_val_targets).reshape(-1)
                        
                        val_mse = mean_squared_error(val_targets, val_preds)
                        val_rmse = np.sqrt(val_mse)
                        val_r2 = r2_score(val_targets, val_preds)
                        
                        # Early stopping logic
                        if val_rmse < best_val_rmse:
                            best_val_rmse = val_rmse
                            patience_counter = 0
                            
                            # Save best model
                            best_model_state = {
                                'epoch': epoch,
                                'model_state_dict': self.__model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_rmse': val_rmse,
                                'val_r2': val_r2
                            }
                            
                            # Make sure directory exists
                            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                            torch.save(best_model_state, f"{checkpoint_path}.pth")
                            print(f"✓ Saved best model at epoch {epoch+1} with val_rmse={val_rmse:.4f}")
                        else:
                            patience_counter += 1
                            if patience_counter >= early_stopping_patience:
                                print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
                                break
                
                # Record metrics
                metrics = {
                    'epoch': epoch + 1,
                    'train_loss': float(avg_train_loss),
                    'val_rmse': float(val_rmse) if val_rmse is not None else None,
                    'val_r2': float(val_r2) if val_r2 is not None else None,
                }
                history.append(metrics)
                
                # Log to TensorBoard
                if writer:
                    writer.add_scalar('Epoch/train_loss', avg_train_loss, epoch)
                    if val_rmse is not None:
                        writer.add_scalar('Epoch/val_rmse', val_rmse, epoch)
                        writer.add_scalar('Epoch/val_r2', val_r2, epoch)
                
                # Print epoch summary
                status = f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}"
                if val_rmse is not None:
                    status += f" - Val RMSE: {val_rmse:.4f} - Val R²: {val_r2:.4f}"
                if patience_counter > 0:
                    status += f" (patience: {patience_counter}/{early_stopping_patience})"
                print(status)
            
            # Load best model if available
            if best_model_state is not None:
                print(f"ℹ️ Loading best model from epoch {best_model_state['epoch']+1}")
                self.__model.load_state_dict(best_model_state['model_state_dict'])
            
            # Test evaluation
            if test_loader:
                self.__model.eval()
                all_test_preds = []
                all_test_targets = []
                
                with torch.no_grad():
                    test_bar = tqdm(enumerate(test_loader), total=test_loader.total_batches,
                                desc="Testing", leave=False)
                    
                    for batch_idx, (X_batch, y_batch) in test_bar:
                        progress_pct = 100.0 * (batch_idx + 1) / test_loader.total_batches
                        
                        for mini_batch_start in range(0, len(X_batch), model_batch_size):
                            mini_batch_end = min(mini_batch_start + model_batch_size, len(X_batch))
                            mini_X = X_batch[mini_batch_start:mini_batch_end].to(self.device)
                            mini_y = y_batch[mini_batch_start:mini_batch_end].to(self.device)
                            
                            outputs = self.__model(mini_X)
                            
                            all_test_preds.append(outputs.cpu().numpy())
                            all_test_targets.append(mini_y.cpu().numpy())
                        
                        # Update progress bar
                        test_bar.set_postfix(progress=f"{progress_pct:.1f}%")
                
                # Calculate test metrics
                if all_test_preds:
                    test_preds = np.vstack(all_test_preds).reshape(-1)
                    test_targets = np.vstack(all_test_targets).reshape(-1)
                    
                    test_mse = mean_squared_error(test_targets, test_preds)
                    test_rmse = np.sqrt(test_mse)
                    test_r2 = r2_score(test_targets, test_preds)
                    
                    print(f"📈 Test Results: RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}")
                    
                    # Log to TensorBoard
                    if writer:
                        writer.add_scalar('Test/rmse', test_rmse, 0)
                        writer.add_scalar('Test/r2', test_r2, 0)
                    
                    # Store results for possible later use
                    self.y_test = test_targets
                    self.y_test_pred = test_preds
                    
        elif self.model_name in ['catboost', 'xgboost', 'random_forest', 'decision_tree', 'gradient_boosting']:
            # Traditional ML models (incrementally trainable)
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Setup for specific model types
            if self.model_name == 'catboost':
                from catboost import Pool
                
                # Set quiet mode
                if hasattr(self.__model, 'set_params'):
                    self.__model.set_params(verbose=False)
            
            # Training tracking variables
            history = []
            
            # Only one epoch for most ML models
            epochs_to_use = 1 if self.model_name != 'catboost' else min(epochs, 5)
            
            for epoch in range(epochs_to_use):
                train_losses = []
                samples_processed = 0
                
                # Process each batch
                train_bar = tqdm(enumerate(train_loader), total=train_loader.total_batches,
                            desc=f"Epoch {epoch+1}/{epochs_to_use}", leave=False)
                
                for batch_idx, (X_batch, y_batch) in train_bar:
                    progress_pct = 100.0 * (batch_idx + 1) / train_loader.total_batches
                    
                    # Convert to numpy
                    X_np = X_batch.numpy()
                    y_np = y_batch.numpy().reshape(-1)
                    
                    # Train on this batch
                    if self.model_name == 'catboost':
                        train_pool = Pool(X_np, y_np)
                        self.__model.fit(train_pool, verbose=False)
                    else:
                        self.__model.fit(X_np, y_np)
                    
                    # Compute metrics for this batch
                    y_pred = self.__model.predict(X_np)
                    mse = mean_squared_error(y_np, y_pred)
                    train_losses.append(mse)
                    samples_processed += len(X_np)
                    
                    # Update progress bar
                    train_bar.set_postfix(
                        progress=f"{progress_pct:.1f}%",
                        mse=f"{mse:.6f}"
                    )
                    
                    # Log batch info
                    if batch_idx % log_every == 0:
                        rmse = np.sqrt(mse)
                        print(f"  Batch {batch_idx+1}: RMSE = {rmse:.6f}")
                        
                        if writer:
                            writer.add_scalar('Batch/train_rmse', rmse, epoch * train_loader.total_batches + batch_idx)
                
                # Calculate average metrics
                avg_train_mse = np.mean(train_losses)
                avg_train_rmse = np.sqrt(avg_train_mse)
                
                # Validation
                val_rmse = None
                val_r2 = None
                
                if val_loader:
                    val_preds = []
                    val_targets = []
                    
                    val_bar = tqdm(enumerate(val_loader), total=val_loader.total_batches,
                                desc=f"Validating (Epoch {epoch+1})", leave=False)
                    
                    for batch_idx, (X_batch, y_batch) in val_bar:
                        progress_pct = 100.0 * (batch_idx + 1) / val_loader.total_batches
                        
                        X_np = X_batch.numpy()
                        y_np = y_batch.numpy().reshape(-1)
                        
                        preds = self.__model.predict(X_np)
                        val_preds.extend(preds)
                        val_targets.extend(y_np)
                        
                        # Update progress bar
                        val_bar.set_postfix(progress=f"{progress_pct:.1f}%")
                    
                    if val_preds:
                        val_mse = mean_squared_error(val_targets, val_preds)
                        val_rmse = np.sqrt(val_mse)
                        val_r2 = r2_score(val_targets, val_preds)
                
                # Record metrics
                metrics = {
                    'epoch': epoch + 1,
                    'train_rmse': float(avg_train_rmse),
                    'val_rmse': float(val_rmse) if val_rmse is not None else None,
                    'val_r2': float(val_r2) if val_r2 is not None else None,
                }
                history.append(metrics)
                
                # Log to TensorBoard
                if writer:
                    writer.add_scalar('Epoch/train_rmse', avg_train_rmse, epoch)
                    if val_rmse is not None:
                        writer.add_scalar('Epoch/val_rmse', val_rmse, epoch)
                        writer.add_scalar('Epoch/val_r2', val_r2, epoch)
                
                # Print epoch summary
                status = f"Epoch {epoch+1}/{epochs_to_use} - Train RMSE: {avg_train_rmse:.4f}"
                if val_rmse is not None:
                    status += f" - Val RMSE: {val_rmse:.4f} - Val R²: {val_r2:.4f}"
                print(status)
            
            # Test evaluation
            if test_loader:
                test_preds = []
                test_targets = []
                
                test_bar = tqdm(enumerate(test_loader), total=test_loader.total_batches,
                            desc="Testing", leave=False)
                
                for batch_idx, (X_batch, y_batch) in test_bar:
                    progress_pct = 100.0 * (batch_idx + 1) / test_loader.total_batches
                    
                    X_np = X_batch.numpy()
                    y_np = y_batch.numpy().reshape(-1)
                    
                    preds = self.__model.predict(X_np)
                    test_preds.extend(preds)
                    test_targets.extend(y_np)
                    
                    # Update progress bar
                    test_bar.set_postfix(progress=f"{progress_pct:.1f}%")
                
                if test_preds:
                    test_mse = mean_squared_error(test_targets, test_preds)
                    test_rmse = np.sqrt(test_mse)
                    test_r2 = r2_score(test_targets, test_preds)
                    
                    print(f"📈 Test Results: RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}")
                    
                    # Log to TensorBoard
                    if writer:
                        writer.add_scalar('Test/rmse', test_rmse, 0)
                        writer.add_scalar('Test/r2', test_r2, 0)
        
        else:
            raise ValueError(f"Model type '{self.model_name}' not supported for streaming training.")
        
        # Save history to CSV
        history_df = pd.DataFrame(history)
        history_path = os.path.join(output_metrics_dir, "streaming_training_history.csv")
        history_df.to_csv(history_path, index=False)
        print(f"✅ Training history saved to {history_path}")
        
        # Close TensorBoard writer
        if writer:
            writer.close()
            print(f"✅ TensorBoard logs saved to {os.path.join(output_metrics_dir, 'tensorboard')}")
        
        return history
    
    
    
    def train_on_single_parquet_file_streaming_v1(
        self,
        epochs: int = 50,
        data_loading_batch_size: int = 65536,  # Large batch for efficient disk I/O
        model_batch_size: int | None = None,   # Smaller batch for model training
        target_col: str = 'y',
        exclude_cols: list | None = None,
        log_every: int = 10,
        early_stopping_patience: int = 10,
        output_metrics_dir: str | None = None,
        checkpoint_path: str = "best_model",
        enable_tensorboard: bool = True,
        lr: float = 1e-3
    ):
        """
        Train a model on a single parquet file with memory-efficient streaming.
        
        This method efficiently handles large parquet files by:
        1. Reading data from disk in large batches (data_loading_batch_size)
        2. Processing model training in smaller batches (model_batch_size)
        
        Parameters:
        -----------
        epochs: int
            Number of training epochs
        data_loading_batch_size: int
            Batch size for loading data from disk (larger = more efficient I/O)
        model_batch_size: int or None
            Batch size for model training (if None, uses self.__batch_size)
        target_col: str
            Name of the target column
        exclude_cols: list or None
            Columns to exclude from features
        log_every: int
            How often to log batch information
        early_stopping_patience: int
            Number of epochs to wait for improvement before stopping
        output_metrics_dir: str or None
            Directory to save metrics and model checkpoints
        checkpoint_path: str
            Path to save best model checkpoint
        enable_tensorboard: bool
            Whether to enable TensorBoard logging
        lr: float
            Learning rate for optimizers
        
        Returns:
        --------
        list
            Training history
        """
        import os
        import torch
        import torch.nn as nn
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from sklearn.metrics import mean_squared_error, r2_score
        import pyarrow.parquet as pq
        
        if not self.single_parquet_path:
            raise ValueError("single_parquet_path not set. This method requires a single parquet file.")
            
        if model_batch_size is None:
            model_batch_size = getattr(self, "__batch_size", 64)
            
        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./results_monitor")
        
        # Ensure output directory exists
        os.makedirs(output_metrics_dir, exist_ok=True)
        
        # Create ParquetLoader for streaming data from a single parquet file
        class ParquetLoader:
            def __init__(
                self, 
                parquet_file, 
                feature_cols, 
                target_col="y", 
                batch_size=65536,
                indices=None, 
                shuffle=True, 
                seed=42
            ):
                self.pf = pq.ParquetFile(parquet_file)
                self.feature_cols = feature_cols
                self.target_col = target_col
                self.columns = feature_cols + [target_col]
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.seed = seed
                
                # Map row groups and their boundaries
                self.row_group_map = []
                total_rows = 0
                for rg in range(self.pf.num_row_groups):
                    rg_rows = self.pf.metadata.row_group(rg).num_rows
                    self.row_group_map.append((total_rows, total_rows + rg_rows))
                    total_rows += rg_rows
                
                self.total_rows = total_rows
                
                # Set indices for data selection (train/val/test split)
                if indices is not None:
                    self.indices = sorted(indices)  # Sort for more efficient disk access
                    self.length = len(indices)
                else:
                    self.indices = list(range(total_rows))
                    self.length = total_rows
                    
                    # Shuffle indices if requested
                    if self.shuffle:
                        import random
                        random.seed(self.seed)
                        random.shuffle(self.indices)
                        
                # Calculate total number of batches
                self.total_batches = (self.length + self.batch_size - 1) // self.batch_size
            
            def __iter__(self):
                # Process data in batches
                for batch_start in range(0, self.length, self.batch_size):
                    batch_indices = self.indices[batch_start:batch_start + self.batch_size]
                    
                    # Group indices by row group for efficient reading
                    indices_by_row_group = {}
                    for idx in batch_indices:
                        for rg_idx, (start, end) in enumerate(self.row_group_map):
                            if start <= idx < end:
                                if rg_idx not in indices_by_row_group:
                                    indices_by_row_group[rg_idx] = []
                                indices_by_row_group[rg_idx].append(idx - start)
                                break
                    
                    # Read data from each row group
                    X_parts, y_parts = [], []
                    for rg_idx, row_indices in indices_by_row_group.items():
                        # Read the whole row group
                        table = self.pf.read_row_group(rg_idx, columns=self.columns)
                        
                        # Extract only the rows we need
                        if len(row_indices) == table.num_rows:
                            # Take all rows
                            X_batch = table.select(self.feature_cols).to_pandas().values.astype(np.float32)
                            y_batch = table.select([self.target_col]).to_pandas().values.astype(np.float32)
                        else:
                            # Take specific rows
                            X_batch = table.select(self.feature_cols).take(row_indices).to_pandas().values.astype(np.float32)
                            y_batch = table.select([self.target_col]).take(row_indices).to_pandas().values.astype(np.float32)
                        
                        X_parts.append(X_batch)
                        y_parts.append(y_batch)
                    
                    if X_parts:
                        X = np.vstack(X_parts)
                        y = np.vstack(y_parts).reshape(-1, 1)
                        
                        # Convert to PyTorch tensors
                        X_tensor = torch.tensor(X, dtype=torch.float32)
                        y_tensor = torch.tensor(y, dtype=torch.float32)
                        
                        yield X_tensor, y_tensor
        
        # Setup TensorBoard logging if enabled
        writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tensorboard_dir = os.path.join(output_metrics_dir, "tensorboard")
                os.makedirs(tensorboard_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=tensorboard_dir)
                print(f"✅ TensorBoard logging enabled at {tensorboard_dir}")
            except ImportError:
                print("⚠️ TensorBoard not available. Install with: pip install tensorboard")
        
        # Extract feature columns and make train/val/test split
        pf = pq.ParquetFile(self.single_parquet_path)
        schema = pf.schema_arrow
        col_names = list(schema.names)
        
        if target_col not in col_names:
            raise ValueError(f"Target column '{target_col}' not found in parquet file.")
            
        # Determine feature columns
        drop_meta = {"date", "year", "month"}
        feature_cols = [c for c in col_names if c not in {target_col, *drop_meta}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        
        # Store feature names
        self.feature_names_ = feature_cols
        
        # Count total rows
        total_rows = sum(pf.metadata.row_group(rg).num_rows for rg in range(pf.num_row_groups))
        print(f"📊 Total rows in parquet file: {total_rows:,}")
        print(f"📋 Features: {len(feature_cols)} columns")
        
        # Create train/val/test split
        from sklearn.model_selection import train_test_split
        
        all_indices = np.arange(total_rows)
        
        # First split: train vs rest
        train_indices, remaining_indices = train_test_split(
            all_indices, 
            train_size=self.train_percent,
            random_state=42
        )
        
        # Second split: val vs test (if needed)
        if len(remaining_indices) > 0 and self.validation_percentage > 0:
            # Calculate relative validation size with safety check
            val_size_relative = min(0.999, self.validation_percentage / (1 - self.train_percent))
            
            val_indices, test_indices = train_test_split(
                remaining_indices,
                train_size=val_size_relative,
                random_state=42
            )
        else:
            val_indices = []
            test_indices = remaining_indices
        
        print(f"🔄 Data split: {len(train_indices):,} train, {len(val_indices):,} validation, {len(test_indices):,} test samples")
        
        # Create data loaders
        train_loader = ParquetLoader(
            self.single_parquet_path,
            feature_cols,
            target_col=target_col,
            batch_size=data_loading_batch_size,
            indices=train_indices,
            shuffle=True
        )
        
        val_loader = None
        if len(val_indices) > 0:
            val_loader = ParquetLoader(
                self.single_parquet_path,
                feature_cols,
                target_col=target_col,
                batch_size=data_loading_batch_size,
                indices=val_indices,
                shuffle=False
            )
        
        test_loader = None
        if len(test_indices) > 0:
            test_loader = ParquetLoader(
                self.single_parquet_path,
                feature_cols,
                target_col=target_col,
                batch_size=data_loading_batch_size,
                indices=test_indices,
                shuffle=False
            )
        
        # Check model type and train accordingly
        if self.model_name == 'tabformer':
            # TabFormer (PyTorch) model
            
            # Initialize model with correct input dimension
            input_dim = len(feature_cols)
            if not isinstance(self.__model, FTTransformer) or self.__model.tokenizer.embedding.in_features != input_dim:
                self.__model = FTTransformer(num_features=input_dim)
            
            self.__model.to(self.device)
            
            # Setup optimizer and loss
            optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            # Training tracking variables
            best_val_rmse = float('inf')
            patience_counter = 0
            best_model_state = None
            global_step = 0
            history = []
            
            # Training loop
            for epoch in range(epochs):
                # Training phase
                self.__model.train()
                train_loss = 0.0
                samples_processed = 0
                
                # Process each data loading batch
                train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)", leave=False)
                for batch_idx, (X_batch, y_batch) in enumerate(train_bar):
                    # Process in model-sized mini-batches
                    for mini_batch_start in range(0, len(X_batch), model_batch_size):
                        mini_batch_end = min(mini_batch_start + model_batch_size, len(X_batch))
                        mini_X = X_batch[mini_batch_start:mini_batch_end].to(self.device)
                        mini_y = y_batch[mini_batch_start:mini_batch_end].to(self.device)
                        
                        # Forward pass
                        optimizer.zero_grad()
                        outputs = self.__model(mini_X)
                        loss = criterion(outputs, mini_y)
                        
                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()
                        
                        # Update statistics
                        batch_loss = loss.item() * len(mini_X)
                        train_loss += batch_loss
                        samples_processed += len(mini_X)
                        
                        # Log to TensorBoard
                        if writer and (global_step % log_every == 0):
                            writer.add_scalar('Batch/train_loss', loss.item(), global_step)
                        
                        global_step += 1
                    
                    # Update progress bar
                    train_bar.set_postfix(loss=f"{train_loss/max(1, samples_processed):.6f}")
                
                # Calculate average training loss
                avg_train_loss = train_loss / max(1, samples_processed)
                
                # Validation phase
                val_rmse = None
                val_r2 = None
                
                if val_loader:
                    self.__model.eval()
                    all_val_preds = []
                    all_val_targets = []
                    
                    with torch.no_grad():
                        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)", leave=False)
                        for X_batch, y_batch in val_bar:
                            for mini_batch_start in range(0, len(X_batch), model_batch_size):
                                mini_batch_end = min(mini_batch_start + model_batch_size, len(X_batch))
                                mini_X = X_batch[mini_batch_start:mini_batch_end].to(self.device)
                                mini_y = y_batch[mini_batch_start:mini_batch_end].to(self.device)
                                
                                outputs = self.__model(mini_X)
                                
                                all_val_preds.append(outputs.cpu().numpy())
                                all_val_targets.append(mini_y.cpu().numpy())
                    
                    # Calculate validation metrics
                    if all_val_preds:
                        val_preds = np.vstack(all_val_preds).reshape(-1)
                        val_targets = np.vstack(all_val_targets).reshape(-1)
                        
                        val_mse = mean_squared_error(val_targets, val_preds)
                        val_rmse = np.sqrt(val_mse)
                        val_r2 = r2_score(val_targets, val_preds)
                        
                        # Early stopping logic
                        if val_rmse < best_val_rmse:
                            best_val_rmse = val_rmse
                            patience_counter = 0
                            
                            # Save best model
                            best_model_state = {
                                'epoch': epoch,
                                'model_state_dict': self.__model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_rmse': val_rmse,
                                'val_r2': val_r2
                            }
                            
                            # Make sure directory exists
                            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                            torch.save(best_model_state, f"{checkpoint_path}.pth")
                            print(f"✓ Saved best model at epoch {epoch+1} with val_rmse={val_rmse:.4f}")
                        else:
                            patience_counter += 1
                            if patience_counter >= early_stopping_patience:
                                print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
                                break
                
                # Record metrics
                metrics = {
                    'epoch': epoch + 1,
                    'train_loss': float(avg_train_loss),
                    'val_rmse': float(val_rmse) if val_rmse is not None else None,
                    'val_r2': float(val_r2) if val_r2 is not None else None,
                }
                history.append(metrics)
                
                # Log to TensorBoard
                if writer:
                    writer.add_scalar('Epoch/train_loss', avg_train_loss, epoch)
                    if val_rmse is not None:
                        writer.add_scalar('Epoch/val_rmse', val_rmse, epoch)
                        writer.add_scalar('Epoch/val_r2', val_r2, epoch)
                
                # Print epoch summary
                status = f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}"
                if val_rmse is not None:
                    status += f" - Val RMSE: {val_rmse:.4f} - Val R²: {val_r2:.4f}"
                if patience_counter > 0:
                    status += f" (patience: {patience_counter}/{early_stopping_patience})"
                print(status)
            
            # Load best model if available
            if best_model_state is not None:
                print(f"ℹ️ Loading best model from epoch {best_model_state['epoch']+1}")
                self.__model.load_state_dict(best_model_state['model_state_dict'])
            
            # Test evaluation
            if test_loader:
                self.__model.eval()
                all_test_preds = []
                all_test_targets = []
                
                with torch.no_grad():
                    test_bar = tqdm(test_loader, desc="Testing", leave=False)
                    for X_batch, y_batch in test_bar:
                        for mini_batch_start in range(0, len(X_batch), model_batch_size):
                            mini_batch_end = min(mini_batch_start + model_batch_size, len(X_batch))
                            mini_X = X_batch[mini_batch_start:mini_batch_end].to(self.device)
                            mini_y = y_batch[mini_batch_start:mini_batch_end].to(self.device)
                            
                            outputs = self.__model(mini_X)
                            
                            all_test_preds.append(outputs.cpu().numpy())
                            all_test_targets.append(mini_y.cpu().numpy())
                
                # Calculate test metrics
                if all_test_preds:
                    test_preds = np.vstack(all_test_preds).reshape(-1)
                    test_targets = np.vstack(all_test_targets).reshape(-1)
                    
                    test_mse = mean_squared_error(test_targets, test_preds)
                    test_rmse = np.sqrt(test_mse)
                    test_r2 = r2_score(test_targets, test_preds)
                    
                    print(f"📈 Test Results: RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}")
                    
                    # Log to TensorBoard
                    if writer:
                        writer.add_scalar('Test/rmse', test_rmse, 0)
                        writer.add_scalar('Test/r2', test_r2, 0)
                    
                    # Store results for possible later use
                    self.y_test = test_targets
                    self.y_test_pred = test_preds
                    
        elif self.model_name in ['catboost', 'xgboost', 'random_forest', 'decision_tree', 'gradient_boosting']:
            # Traditional ML models (incrementally trainable)
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Setup for specific model types
            if self.model_name == 'catboost':
                from catboost import Pool
                
                # Set quiet mode
                if hasattr(self.__model, 'set_params'):
                    self.__model.set_params(verbose=False)
            
            # Training tracking variables
            history = []
            
            # Only one epoch for most ML models
            epochs_to_use = 1 if self.model_name != 'catboost' else min(epochs, 5)
            
            for epoch in range(epochs_to_use):
                train_losses = []
                samples_processed = 0
                
                # Process each batch
                train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_to_use}", leave=False)
                for batch_idx, (X_batch, y_batch) in enumerate(train_bar):
                    # Convert to numpy
                    X_np = X_batch.numpy()
                    y_np = y_batch.numpy().reshape(-1)
                    
                    # Train on this batch
                    if self.model_name == 'catboost':
                        train_pool = Pool(X_np, y_np)
                        self.__model.fit(train_pool, verbose=False)
                    else:
                        self.__model.fit(X_np, y_np)
                    
                    # Compute metrics for this batch
                    y_pred = self.__model.predict(X_np)
                    mse = mean_squared_error(y_np, y_pred)
                    train_losses.append(mse)
                    samples_processed += len(X_np)
                    
                    # Update progress bar
                    train_bar.set_postfix(mse=f"{mse:.6f}")
                    
                    # Log batch info
                    if batch_idx % log_every == 0:
                        rmse = np.sqrt(mse)
                        print(f"  Batch {batch_idx+1}: RMSE = {rmse:.6f}")
                        
                        if writer:
                            writer.add_scalar('Batch/train_rmse', rmse, epoch * train_loader.total_batches + batch_idx)
                
                # Calculate average metrics
                avg_train_mse = np.mean(train_losses)
                avg_train_rmse = np.sqrt(avg_train_mse)
                
                # Validation
                val_rmse = None
                val_r2 = None
                
                if val_loader:
                    val_preds = []
                    val_targets = []
                    
                    val_bar = tqdm(val_loader, desc=f"Validating (Epoch {epoch+1})", leave=False)
                    for X_batch, y_batch in val_bar:
                        X_np = X_batch.numpy()
                        y_np = y_batch.numpy().reshape(-1)
                        
                        preds = self.__model.predict(X_np)
                        val_preds.extend(preds)
                        val_targets.extend(y_np)
                    
                    if val_preds:
                        val_mse = mean_squared_error(val_targets, val_preds)
                        val_rmse = np.sqrt(val_mse)
                        val_r2 = r2_score(val_targets, val_preds)
                
                # Record metrics
                metrics = {
                    'epoch': epoch + 1,
                    'train_rmse': float(avg_train_rmse),
                    'val_rmse': float(val_rmse) if val_rmse is not None else None,
                    'val_r2': float(val_r2) if val_r2 is not None else None,
                }
                history.append(metrics)
                
                # Log to TensorBoard
                if writer:
                    writer.add_scalar('Epoch/train_rmse', avg_train_rmse, epoch)
                    if val_rmse is not None:
                        writer.add_scalar('Epoch/val_rmse', val_rmse, epoch)
                        writer.add_scalar('Epoch/val_r2', val_r2, epoch)
                
                # Print epoch summary
                status = f"Epoch {epoch+1}/{epochs_to_use} - Train RMSE: {avg_train_rmse:.4f}"
                if val_rmse is not None:
                    status += f" - Val RMSE: {val_rmse:.4f} - Val R²: {val_r2:.4f}"
                print(status)
            
            # Test evaluation
            if test_loader:
                test_preds = []
                test_targets = []
                
                test_bar = tqdm(test_loader, desc="Testing", leave=False)
                for X_batch, y_batch in test_bar:
                    X_np = X_batch.numpy()
                    y_np = y_batch.numpy().reshape(-1)
                    
                    preds = self.__model.predict(X_np)
                    test_preds.extend(preds)
                    test_targets.extend(y_np)
                
                if test_preds:
                    test_mse = mean_squared_error(test_targets, test_preds)
                    test_rmse = np.sqrt(test_mse)
                    test_r2 = r2_score(test_targets, test_preds)
                    
                    print(f"📈 Test Results: RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}")
                    
                    # Log to TensorBoard
                    if writer:
                        writer.add_scalar('Test/rmse', test_rmse, 0)
                        writer.add_scalar('Test/r2', test_r2, 0)
        
        else:
            raise ValueError(f"Model type '{self.model_name}' not supported for streaming training.")
        
        # Save history to CSV
        history_df = pd.DataFrame(history)
        history_path = os.path.join(output_metrics_dir, "streaming_training_history.csv")
        history_df.to_csv(history_path, index=False)
        print(f"✅ Training history saved to {history_path}")
        
        # Close TensorBoard writer
        if writer:
            writer.close()
            print(f"✅ TensorBoard logs saved to {os.path.join(output_metrics_dir, 'tensorboard')}")
        
        return history
    
    
    def _load_single_parquet_into_memory(self, target_col='y', exclude_cols=None):
        """
        Load entire single parquet file into RAM (X,y) once (with progress bar).
        Sets internal train/test split arrays.
        """
        if not self.single_parquet_path:
            raise ValueError("single_parquet_path not set.")
        import pyarrow.parquet as pq
        import pyarrow as pa
        import numpy as np
        from tqdm.auto import tqdm

        pf = pq.ParquetFile(self.single_parquet_path)
        schema = pf.schema_arrow
        col_names = list(schema.names)
        if target_col not in col_names:
            raise ValueError(f"Target column '{target_col}' not found in parquet file.")
        drop_meta = {"date", "year", "month"}
        feature_cols = [c for c in col_names if c not in {target_col, *drop_meta}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]

        # Row‑group streaming with progress
        tables = []
        total_rows = 0
        for rg in tqdm(range(pf.num_row_groups), desc="Loading Parquet (row groups)", leave=False):
            rg_tbl = pf.read_row_group(rg, columns=feature_cols + [target_col])
            tables.append(rg_tbl)
            total_rows += rg_tbl.num_rows

        if not tables:
            raise ValueError("Parquet file appears empty.")

        full_table = pa.concat_tables(tables, promote=True)
        # Convert to pandas (single materialization)
        df = full_table.to_pandas()

        y = df[target_col].to_numpy().astype(np.float32, copy=False)
        X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)

        self.feature_names_ = feature_cols
        self.x = X
        self.y = y

        from sklearn.model_selection import train_test_split
        if self.train_percent != 1:
            self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(
                self.x, self.y,
                train_size=self.train_percent,
                test_size=1 - self.train_percent,
                random_state=42
            )
        else:
            self.__x_train, self.__x_test, self.__y_train, self.__y_test = self.x, None, self.y, None

    
    
    def train_single_parquet_tabpfn_fullram(
        self,
        *,
        target_col: str = "y",
        exclude_cols: list[str] | None = None,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        subsample_samples: int = 10_000,
        max_predict_time: int = 60,
        fit_nodes: bool = True,
        adaptive_tree: bool = True,
        verbose: int = 1,
        output_metrics_dir: str | None = None,
    ):
        """
        Full in-RAM training with TabPFN + RandomForest preprocessing.
        Supports regression ('r') and classification ('c').
        - Loads entire parquet file into memory
        - Splits train/val/test (val carved from the train split)
        - Fits RandomForestTabPFN[Regressor|Classifier]
        - Saves model and metrics history CSV
        """
        try:
            from tabpfn_extensions import TabPFNRegressor, TabPFNClassifier
            from tabpfn_extensions.rf_pfn import (
                RandomForestTabPFNRegressor,
                RandomForestTabPFNClassifier,
            )
        except Exception as e:
            raise ImportError(
                f"TabPFN extensions not available: {e}. Please install 'tabpfn_extensions'."
            )

        import os
        import numpy as np
        import pandas as pd
        import joblib
        import pyarrow as pa
        import pyarrow.parquet as pq
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            mean_squared_error,
            r2_score,
            accuracy_score,
            f1_score,
        )

        if not self.single_parquet_path:
            raise ValueError("single_parquet_path not set.")

        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./model") or "./model"
        os.makedirs(output_metrics_dir, exist_ok=True)

        # Discover feature columns
        pf = pq.ParquetFile(self.single_parquet_path, memory_map=True)
        schema = pf.schema_arrow
        col_names = list(schema.names)
        if target_col not in col_names:
            raise ValueError(f"Target column '{target_col}' not found in parquet file.")
        drop_meta = {"date", "year", "month"}
        feature_cols = [c for c in col_names if c not in {target_col, *drop_meta}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        self.feature_names_ = feature_cols

        # Load whole file to RAM (row-group concat)
        tables = []
        total_rows = 0
        for rg in range(pf.num_row_groups):
            rg_tbl = pf.read_row_group(rg, columns=feature_cols + [target_col])
            tables.append(rg_tbl)
            total_rows += rg_tbl.num_rows
        if not tables:
            raise ValueError("Parquet file appears empty.")
        full_table = pa.concat_tables(tables, promote=True)
        df = full_table.to_pandas()

        # Prepare X, y
        X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        y_raw = df[target_col].to_numpy(copy=False)

        is_classification = str(getattr(self, "_Model__task", self.__task)).lower().startswith("c")
        if is_classification:
            # Ensure labels are integers; factorize non-numeric/categorical
            if not np.issubdtype(y_raw.dtype, np.integer):
                # If floats representing classes, convert to int if all values are close to integers
                try:
                    unique_vals = pd.unique(y_raw)
                    if np.issubdtype(unique_vals.dtype, np.floating) and np.allclose(unique_vals, np.round(unique_vals)):
                        y = unique_vals.astype(int)
                        mapping = {v: int(v) for v in unique_vals}
                        y = np.array([mapping[v] for v in y_raw], dtype=np.int64)
                        self.class_names_ = [str(int(v)) for v in unique_vals]
                    else:
                        y, class_names = pd.factorize(y_raw)
                        y = y.astype(np.int64, copy=False)
                        self.class_names_ = [str(c) for c in class_names]
                except Exception:
                    y, class_names = pd.factorize(y_raw)
                    y = y.astype(np.int64, copy=False)
                    self.class_names_ = [str(c) for c in class_names]
            else:
                y = y_raw.astype(np.int64, copy=False)
        else:
            # Regression: keep as float
            y = y_raw.astype(np.float32, copy=False)

        # Ratios
        tr_ratio = float(train_ratio) if train_ratio is not None else float(self.train_percent)
        vl_ratio = float(val_ratio) if val_ratio is not None else float(self.validation_percentage)
        print(f"🔀 Splitting data with train_ratio={tr_ratio}, val_ratio={vl_ratio}")

        # Split: train+val vs test, then val from train portion
        all_indices = np.arange(total_rows)
        trainval_indices, test_indices = train_test_split(
            all_indices,
            train_size=tr_ratio,
            random_state=42,
            shuffle=True,
        )
        if vl_ratio > 0:
            train_indices, val_indices = train_test_split(
                trainval_indices,
                test_size=vl_ratio,
                random_state=42,
                shuffle=True,
            )
        else:
            train_indices = trainval_indices
            val_indices = np.array([], dtype=np.int64)

        print(
            f"🔄 Split -> train: {len(train_indices):,}, "
            f"val: {len(val_indices):,}, test: {len(test_indices):,}"
        )

        # Materialize splits
        X_train, y_train = X[train_indices], y[train_indices]
        X_val = y_val = None
        if val_indices.size > 0:
            X_val, y_val = X[val_indices], y[val_indices]
        X_test = y_test = None
        if test_indices.size > 0:
            X_test, y_test = X[test_indices], y[test_indices]

        # Build model
        if is_classification:
            base = TabPFNClassifier(
                ignore_pretraining_limits=True,
                inference_config={"SUBSAMPLE_SAMPLES": int(subsample_samples)},
            )
            model = RandomForestTabPFNClassifier(
                tabpfn=base,
                verbose=int(verbose),
                max_predict_time=int(max_predict_time),
                fit_nodes=bool(fit_nodes),
                adaptive_tree=bool(adaptive_tree),
            )
        else:
            base = TabPFNRegressor(
                ignore_pretraining_limits=True,
                inference_config={"SUBSAMPLE_SAMPLES": int(subsample_samples)},
            )
            model = RandomForestTabPFNRegressor(
                tabpfn=base,
                verbose=int(verbose),
                max_predict_time=int(max_predict_time),
                fit_nodes=bool(fit_nodes),
                adaptive_tree=bool(adaptive_tree),
                show_progress=bool(verbose > 0),
            )

        # Fit
        model.fit(X_train, y_train)
        self.__model = model

        # Metrics
        history = []
        if is_classification:
            y_tr_pred = model.predict(X_train)
            tr_acc = float(accuracy_score(y_train, y_tr_pred))
            tr_f1 = float(f1_score(y_train, y_tr_pred, average="macro"))

            val_acc = val_f1 = None
            if X_val is not None:
                y_val_pred = model.predict(X_val)
                val_acc = float(accuracy_score(y_val, y_val_pred))
                val_f1 = float(f1_score(y_val, y_val_pred, average="macro"))

            test_acc = test_f1 = None
            if X_test is not None:
                y_test_pred = model.predict(X_test)
                test_acc = float(accuracy_score(y_test, y_test_pred))
                test_f1 = float(f1_score(y_test, y_test_pred, average="macro"))
                self.y_test = y_test
                self.y_test_pred = y_test_pred

            history.append({
                "task": "classification",
                "train_acc": tr_acc,
                "train_f1_macro": tr_f1,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
                "test_acc": test_acc,
                "test_f1_macro": test_f1,
                "train_rows": int(len(y_train)),
                "val_rows": int(len(y_val)) if y_val is not None else 0,
                "test_rows": int(len(y_test)) if y_test is not None else 0,
            })
        else:
            y_tr_pred = model.predict(X_train)
            tr_mse = mean_squared_error(y_train, y_tr_pred)
            tr_rmse = float(np.sqrt(tr_mse))
            tr_r2 = float(r2_score(y_train, y_tr_pred))

            val_rmse = val_r2 = None
            if X_val is not None:
                y_val_pred = model.predict(X_val)
                val_mse = mean_squared_error(y_val, y_val_pred)
                val_rmse = float(np.sqrt(val_mse))
                val_r2 = float(r2_score(y_val, y_val_pred))

            test_rmse = test_r2 = None
            if X_test is not None:
                y_test_pred = model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_rmse = float(np.sqrt(test_mse))
                test_r2 = float(r2_score(y_test, y_test_pred))
                self.y_test = y_test
                self.y_test_pred = y_test_pred

            history.append({
                "task": "regression",
                "train_rmse": tr_rmse,
                "train_r2": tr_r2,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "train_rows": int(len(y_train)),
                "val_rows": int(len(y_val)) if y_val is not None else 0,
                "test_rows": int(len(y_test)) if y_test is not None else 0,
            })

        # Save artifacts
        model_path = os.path.join(output_metrics_dir, "best_tabpfn_rf_fullram.pkl")
        try:
            joblib.dump(model, model_path)
            print(f"✓ Saved TabPFN RF model -> {model_path}")
        except Exception as e:
            print(f"Warning: could not save model ({e})")

        hist_path = os.path.join(output_metrics_dir, "tabpfn_rf_fullram_history.csv")
        pd.DataFrame(history).to_csv(hist_path, index=False)
        print(f"✓ Saved training history -> {hist_path}")

        return history
    
    
    def train_single_parquet_tabpfn_streaming_subsampling(
        self,
        *,
        target_col: str = "y",
        exclude_cols: list[str] | None = None,
        data_loading_batch_size: int = 65536,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        n_estimators: int = 32,
        subsample_samples: int = 10_000,
        verbose: int = 1,
        output_metrics_dir: str | None = None,
        use_cpu_inference: bool = False,
        chunk_size: int = 65536,
        n_jobs: int = 1,
    ):
        """
        Streaming trainer using TabPFN subsampled ensemble (no RF wrapper).
        - Streams a single parquet file by row-groups with tqdm progress.
        - Splits into train/val/test, with validation taken from the training portion.
        - Supports regression ('r') via TabPFNRegressor and classification ('c') via TabPFNClassifier.

        Artifacts:
        - Saves model to best_tabpfn_subsample_streaming.pkl
        - Saves metrics to tabpfn_subsample_streaming_history.csv
        """
        # Prefer tabpfn_extensions for the Regressor; fall back to tabpfn for classifier
        try:
            from tabpfn_extensions import TabPFNRegressor, TabPFNClassifier
            _has_reg = True
        except Exception:
            try:
                from tabpfn import TabPFNClassifier  # classifier fallback
                TabPFNRegressor = None
                _has_reg = False
            except Exception as e:
                raise ImportError(
                    f"TabPFN not available: {e}. Install 'tabpfn_extensions' (preferred) or 'tabpfn'."
                )

        import os
        import numpy as np
        import pandas as pd
        import joblib
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tqdm.auto import tqdm
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

        if not self.single_parquet_path:
            raise ValueError("single_parquet_path not set.")

        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./model") or "./model"
        os.makedirs(output_metrics_dir, exist_ok=True)

        # Schema discovery
        pf = pq.ParquetFile(self.single_parquet_path, memory_map=True)
        schema = pf.schema_arrow
        col_names = list(schema.names)
        if target_col not in col_names:
            raise ValueError(f"Target column '{target_col}' not found in parquet file.")
        drop_meta = {"date", "year", "month"}
        feature_cols = [c for c in col_names if c not in {target_col, *drop_meta}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        self.feature_names_ = feature_cols

        # Row-group info
        rg_offsets, rg_sizes = [], []
        running = 0
        for rg in range(pf.num_row_groups):
            nrg = pf.metadata.row_group(rg).num_rows
            rg_offsets.append(running)
            rg_sizes.append(nrg)
            running += nrg
        total_rows = running
        print(f"📊 Total rows: {total_rows:,} | Features: {len(feature_cols)}")

        # Split (val from train portion only if val_ratio is explicitly provided)
        all_indices = np.arange(total_rows)

        tr_ratio = float(train_ratio) if train_ratio is not None else float(self.train_percent)

        # If val_ratio is None → disable validation; else use the provided value
        use_val = (val_ratio is not None) and (float(val_ratio) > 0)
        if use_val:
            vl_ratio = float(val_ratio)
            print(f"🔀 Splitting data with train_ratio={tr_ratio}, val_ratio={vl_ratio}")
        else:
            print(f"🔀 Splitting data with train_ratio={tr_ratio}, val_ratio=None")

        # First split: train+val vs test
        trainval_indices, test_indices = train_test_split(
            all_indices,
            train_size=tr_ratio,
            random_state=42,
            shuffle=True,
        )

        # Second split: only if validation explicitly requested
        if use_val:
            train_indices, val_indices = train_test_split(
                trainval_indices,
                test_size=vl_ratio,
                random_state=42,
                shuffle=True,
            )
        else:
            train_indices = trainval_indices
            val_indices = np.array([], dtype=np.int64)

        print(
            f"🔄 Split -> train: {len(train_indices):,}, "
            f"val: {len(val_indices):,}, test: {len(test_indices):,}"
        )
        
        print("🗂️  Parquet rows: {:,} | 🧩 Features: {}".format(total_rows, len(feature_cols)))
        print(f"Features: {feature_cols}")

        print(f"Split: train={len(train_indices):,}, val={len(val_indices):,}, test={len(test_indices):,}")

        # Streaming helper
        def stream_selected(indices: np.ndarray, batch_size: int):
            if indices is None or len(indices) == 0:
                return
            sel = np.sort(indices.astype(np.int64, copy=False))
            for rg in range(pf.num_row_groups):
                lo = rg_offsets[rg]
                hi = lo + rg_sizes[rg]
                mask = (sel >= lo) & (sel < hi)
                if not np.any(mask):
                    continue
                local_idx = sel[mask] - lo
                tbl = pf.read_row_group(rg, columns=feature_cols + [target_col])
                df_rg = tbl.to_pandas()
                for i in range(0, local_idx.size, batch_size):
                    sl = local_idx[i : i + batch_size]
                    df_b = df_rg.iloc[sl]
                    Xb = df_b[feature_cols].to_numpy(dtype=np.float32, copy=False)
                    yb = df_b[target_col].to_numpy(copy=False)
                    yield Xb, yb

        # Load splits with progress
        def collect(indices, desc):
            if indices is None or len(indices) == 0:
                return None, None
            X_parts, y_parts = [], []
            pbar = tqdm(total=len(indices), desc=desc, unit="rows", position=0)
            for Xb, yb in stream_selected(indices, data_loading_batch_size):
                X_parts.append(Xb)
                y_parts.append(yb)
                pbar.update(len(yb))
            pbar.close()
            if not X_parts:
                return None, None
            X = np.vstack(X_parts)
            y = np.concatenate(y_parts)
            return X, y

        X_train, y_train = collect(train_indices, "Loading train data")
        if X_train is None:
            raise ValueError("No training data loaded. Check indices/split settings.")
        X_val, y_val = collect(val_indices, "Loading val data")
        X_test, y_test = collect(test_indices, "Loading test data")

        # Task type
        is_classification = str(getattr(self, "_Model__task", self.__task)).lower().startswith("c")

        # Prepare y dtype
        if is_classification:
            # Ensure integer labels
            import pandas as pd  # alias to avoid shadowing
            if not np.issubdtype(y_train.dtype, np.integer):
                y_train, classes = pd.factorize(y_train)
                y_train = y_train.astype(np.int64, copy=False)
                if y_val is not None:
                    y_val = pd.Categorical(y_val, categories=classes).codes.astype(np.int64, copy=False)
                if y_test is not None:
                    y_test = pd.Categorical(y_test, categories=classes).codes.astype(np.int64, copy=False)
                self.class_names_ = [str(c) for c in classes]
            else:
                y_train = y_train.astype(np.int64, copy=False)
                if y_val is not None:
                    y_val = y_val.astype(np.int64, copy=False)
                if y_test is not None:
                    y_test = y_test.astype(np.int64, copy=False)
        else:
            # Regression as float32
            y_train = y_train.astype(np.float32, copy=False)
            if y_val is not None:
                y_val = y_val.astype(np.float32, copy=False)
            if y_test is not None:
                y_test = y_test.astype(np.float32, copy=False)

        # Build subsampled ensemble
        if is_classification:
            model = TabPFNClassifier(
                ignore_pretraining_limits=True,
                n_estimators=int(n_estimators),
                inference_config={"SUBSAMPLE_SAMPLES": int(subsample_samples)},
            )
        else:
            if TabPFNRegressor is None:
                raise ImportError(
                    "TabPFNRegressor not found. Install 'tabpfn_extensions' to use regression subsampled ensemble."
                )
            model = TabPFNRegressor(
                ignore_pretraining_limits=True,
                n_estimators=int(n_estimators),
                inference_config={"SUBSAMPLE_SAMPLES": int(subsample_samples)},
                #memory_saving_mode=True,
                fit_mode="fit_preprocessors",
                n_preprocessing_jobs=n_jobs,
            )

        # Save fitted TabPFN model and reload for inference
        model_fit_path = os.path.join(output_metrics_dir, "best_tabpfn_subsample_streaming.tabpfn_fit")
        
        if os.path.exists(model_fit_path):
            print(f"ℹ️  Found existing fitted model at {model_fit_path}, loading it.")
            if use_cpu_inference:
                print("⚙️  Using CPU for inference as requested.")
                self.use_gpu = False  # force CPU for inference
            else:                
                print("⚙️  Using GPU for inference if available.")
            
            self.load_model(model_fit_path)
        else:
            print(f"🔧 Fitting TabPFN subsampled ensemble ({n_estimators} est.)")
            # Fit with a simple training phase bar
            train_phase = tqdm(total=1, desc=f"Training TabPFN ({n_estimators} est.)", position=0, unit="phase")
            model.fit(X_train, y_train)
            train_phase.update(1)
            train_phase.close()
            self.__model = model
            self.save_model(model_fit_path)

        # Metrics
        history = []
        if is_classification:
            y_tr_pred = model.predict(X_train)
            tr_acc = float(accuracy_score(y_train, y_tr_pred))
            tr_f1 = float(f1_score(y_train, y_tr_pred, average="macro"))

            val_acc = val_f1 = None
            if X_val is not None:
                y_val_pred = model.predict(X_val)
                val_acc = float(accuracy_score(y_val, y_val_pred))
                val_f1 = float(f1_score(y_val, y_val_pred, average="macro"))

            test_acc = test_f1 = None
            if X_test is not None:
                y_test_pred = model.predict(X_test)
                test_acc = float(accuracy_score(y_test, y_test_pred))
                test_f1 = float(f1_score(y_test, y_test_pred, average="macro"))
                self.y_test = y_test
                self.y_test_pred = y_test_pred

            history.append({
                "task": "classification",
                "n_estimators": int(n_estimators),
                "subsample_samples": int(subsample_samples),
                "train_acc": tr_acc,
                "train_f1_macro": tr_f1,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
                "test_acc": test_acc,
                "test_f1_macro": test_f1,
                "train_rows": int(len(y_train)),
                "val_rows": int(len(y_val)) if y_val is not None else 0,
                "test_rows": int(len(y_test)) if y_test is not None else 0,
            })
        else:
            # Chunked predictions with tqdm for progress + lower memory spikes
            def _predict_with_tqdm(X, desc: str, chunk_size: int = chunk_size):
                n = len(X)
                if n == 0:
                    return np.array([], dtype=np.float32)
                preds = []
                for i in tqdm(range(0, n, chunk_size), desc=desc, unit="rows", position=0):
                    xb = X[i:i+chunk_size]
                    pb = self.predict(xb)
                    # Ensure 1D float array
                    pb = np.asarray(pb)
                    if pb.ndim > 1 and pb.shape[-1] == 1:
                        pb = pb.reshape(-1)
                    preds.append(pb)
                return np.concatenate(preds, axis=0)

            val_rmse = val_r2 = None
            if X_val is not None:
                y_val_pred = _predict_with_tqdm(X_val, "Predict val")
                val_mse = mean_squared_error(y_val, y_val_pred)
                val_rmse = float(np.sqrt(val_mse))
                val_r2 = float(r2_score(y_val, y_val_pred))

            test_rmse = test_r2 = None
            if X_test is not None:
                y_test_pred = _predict_with_tqdm(X_test, "Predict test")
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_rmse = float(np.sqrt(test_mse))
                test_r2 = float(r2_score(y_test, y_test_pred))
                self.y_test = y_test
                self.y_test_pred = y_test_pred

            history.append({
                "task": "regression",
                "n_estimators": int(n_estimators),
                "subsample_samples": int(subsample_samples),
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "train_rows": int(len(y_train)),
                "val_rows": int(len(y_val)) if y_val is not None else 0,
                "test_rows": int(len(y_test)) if y_test is not None else 0,
            })

        hist_path = os.path.join(output_metrics_dir, "tabpfn_subsample_streaming_history.csv")
        pd.DataFrame(history).to_csv(hist_path, index=False)
        print(f"✓ Saved training history -> {hist_path}")

        return history
    
    
    def train_on_single_parquet_file_streaming_xgboost(
        self,
        *,
        target_col: str = "y",
        exclude_cols: list[str] | None = None,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        iterations: int = 2000,
        depth: int = 8,
        lr: float = 0.03,
        output_metrics_dir: str | None = None,
        show_progress: bool = True,
    ):
        """
        Train XGBoost on a single Parquet file with the same streaming and split behavior
        as CatBoost/TabPFN subsampling:
        - Split as: train+val vs test; carve validation from train only if val_ratio is provided.
        - Stream row-groups to materialize (X, y) splits.
        - Fit an XGBoost Reg/Clf with optional early stopping when validation exists.
        - Save per-iteration metrics CSV, store test predictions for regression.
        """
        import os
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tqdm.auto import tqdm
        from sklearn.model_selection import train_test_split

        if self.model_name not in ("xgboost", "xgb"):
            raise ValueError("This method is only for XGBoost models. Set model_name='xgboost'.")

        if not getattr(self, "single_parquet_path", None):
            raise ValueError("single_parquet_path not set.")

        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./results_monitor") or "./results_monitor"
        os.makedirs(output_metrics_dir, exist_ok=True)

        # Discover schema and feature columns
        pf = pq.ParquetFile(self.single_parquet_path, memory_map=True)
        schema = pf.schema_arrow
        names = list(schema.names)
        if target_col not in names:
            raise ValueError(f"Target column '{target_col}' not found in parquet.")

        meta_drop = {"date", "year", "month"}
        feature_cols = [c for c in names if c not in {target_col, *meta_drop}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        self.feature_names_ = feature_cols

        # Count rows
        total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
        print(f"📊 Total rows: {total_rows:,} | Features: {len(feature_cols)}")

        # Split logic: match TabPFN subsampling (val only if explicitly provided)
        tr_ratio = float(train_ratio) if train_ratio is not None else float(getattr(self, "train_percent", 0.8))
        use_val = (val_ratio is not None) and (float(val_ratio) > 0)
        if use_val:
            vl_ratio = float(val_ratio)
            print(f"🔀 Splitting data with train_ratio={tr_ratio}, val_ratio={vl_ratio}")
        else:
            print(f"🔀 Splitting data with train_ratio={tr_ratio}, val_ratio=None")

        all_idx = np.arange(total_rows, dtype=np.int64)
        trainval_idx, test_idx = train_test_split(
            all_idx,
            train_size=tr_ratio,
            random_state=42,
            shuffle=True,
        )
        if use_val:
            train_idx, val_idx = train_test_split(
                trainval_idx,
                test_size=vl_ratio,
                random_state=42,
                shuffle=True,
            )
        else:
            train_idx = trainval_idx
            val_idx = np.array([], dtype=np.int64)

        print(f"🔄 Split -> train: {len(train_idx):,}, val: {len(val_idx):,}, test: {len(test_idx):,}")

        # Helper: gather (X,y) for selected global indices via row-group iteration
        def _gather_by_indices(indices: np.ndarray, desc: str):
            if indices is None or indices.size == 0:
                return None, None
            indices_sorted = np.sort(indices)

            # Precompute row-group bounds
            rg_bounds = []
            acc = 0
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                rg_bounds.append((acc, acc + n))
                acc += n

            feat_parts, tgt_parts = [], []
            bar = tqdm(range(pf.num_row_groups), desc=desc, leave=False, disable=not show_progress)
            for rg in bar:
                start, end = rg_bounds[rg]
                left = np.searchsorted(indices_sorted, start, side="left")
                right = np.searchsorted(indices_sorted, end, side="left")
                local = indices_sorted[left:right] - start
                if local.size == 0:
                    continue
                tbl = pf.read_row_group(rg, columns=feature_cols + [target_col])
                tbl_sel = tbl.take(pa.array(local, type=pa.int64()))
                # Convert to numpy
                try:
                    X_cols = [tbl_sel[c].to_numpy(zero_copy_only=False) for c in feature_cols]
                    X = np.column_stack(X_cols).astype(np.float32, copy=False)
                    y = tbl_sel[target_col].to_numpy(zero_copy_only=False).reshape(-1)
                except Exception:
                    pdf = tbl_sel.to_pandas()
                    X = pdf[feature_cols].to_numpy(dtype=np.float32, copy=False)
                    y = pdf[target_col].to_numpy(copy=False)
                feat_parts.append(X)
                tgt_parts.append(y)

            if not feat_parts:
                return None, None
            X_all = np.vstack(feat_parts)
            y_all = np.concatenate(tgt_parts)
            return X_all, y_all

        # Materialize splits
        X_train, y_train = _gather_by_indices(train_idx, "Load train")
        if X_train is None:
            raise RuntimeError("No training rows collected.")
        X_val, y_val = _gather_by_indices(val_idx, "Load val") if val_idx.size > 0 else (None, None)
        X_test, y_test = _gather_by_indices(test_idx, "Load test") if test_idx.size > 0 else (None, None)

        # Task detection
        is_classification = str(getattr(self, "_Model__task", getattr(self, "__task", "r"))).lower().startswith("c")

        # y dtype and class handling
        if is_classification:
            y_train, classes = pd.factorize(y_train)
            y_train = y_train.astype(np.int64, copy=False)
            if X_val is not None:
                y_val = pd.Categorical(y_val, categories=classes).codes.astype(np.int64, copy=False)
            if X_test is not None:
                y_test = pd.Categorical(y_test, categories=classes).codes.astype(np.int64, copy=False)
            self.class_names_ = [str(c) for c in classes]
            n_classes = int(len(classes))
        else:
            y_train = y_train.astype(np.float32, copy=False)
            if X_val is not None:
                y_val = y_val.astype(np.float32, copy=False)
            if X_test is not None:
                y_test = y_test.astype(np.float32, copy=False)

        # Build XGBoost model (GPU-aware)
        import xgboost as xgb
        xgb_version = getattr(xgb, "__version__", "1.7.0")

        def _version_ge(v: str, major: int, minor: int = 0):
            try:
                parts = [int(p) for p in v.split(".")[:2]]
                return (parts[0], parts[1]) >= (major, minor)
            except Exception:
                return False

        use_gpu = bool(getattr(self, "use_gpu", False))
        params_common = dict(
            n_estimators=int(iterations),
            max_depth=int(depth),
            learning_rate=float(lr),
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42,
        )

        # Device/tree_method setup across XGB versions
        if _version_ge(xgb_version, 2, 0):
            params_common["tree_method"] = "hist"
            params_common["device"] = "cuda" if use_gpu else "cpu"
        else:
            params_common["tree_method"] = "gpu_hist" if use_gpu else "hist"
            if use_gpu:
                params_common["predictor"] = "gpu_predictor"

        if is_classification:
            if n_classes == 2:
                objective = "binary:logistic"
                eval_metric = "logloss"
                model = xgb.XGBClassifier(objective=objective, eval_metric=eval_metric, **params_common)
            else:
                objective = "multi:softprob"
                eval_metric = "mlogloss"
                model = xgb.XGBClassifier(objective=objective, num_class=n_classes, eval_metric=eval_metric, **params_common)
        else:
            objective = "reg:squarederror"
            eval_metric = "rmse"
            model = xgb.XGBRegressor(objective=objective, eval_metric=eval_metric, **params_common)

        # Fit with optional early stopping (eval_metric stays in constructor)
        eval_set = []
        eval_set.append((X_train, y_train))
        if X_val is not None and y_val is not None and len(y_val) > 0:
            eval_set.append((X_val, y_val))

        fit_kwargs = dict(verbose=show_progress)
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
       

        model.fit(X_train, y_train, **fit_kwargs)
        self.__model = model

        # Save per-iteration metrics, if available
        try:
            evals_result = model.evals_result()
            rows = []
            splits = list(evals_result.keys())  # e.g., 'validation_0', 'validation_1'
            metric_names = set()
            for sp in splits:
                metric_names.update(evals_result[sp].keys())
            max_len = 0
            for sp in splits:
                for mk in evals_result[sp].keys():
                    max_len = max(max_len, len(evals_result[sp][mk]))
            for i in range(max_len):
                row = {"iteration": i}
                for sp in splits:
                    for mk in metric_names:
                        vals = evals_result.get(sp, {}).get(mk, [])
                        if i < len(vals):
                            row[f"{sp}_{mk}"] = vals[i]
                rows.append(row)
            if rows:
                pd.DataFrame(rows).to_csv(os.path.join(output_metrics_dir, "xgboost_iteration_metrics.csv"), index=False)
                print(f"✅ XGBoost iteration metrics saved to {os.path.join(output_metrics_dir, 'xgboost_iteration_metrics.csv')}")
        except Exception:
            pass

        # Keep test arrays for report() if regression
        if not is_classification and X_test is not None and y_test is not None:
            self.y_test = y_test
            self.y_test_pred = model.predict(X_test)

        print("✅ XGBoost training complete.")
        return True
    
    
    def train_single_parquet_tabpfn_streaming_rf(
        self,
        *,
        target_col: str = "y",
        exclude_cols: list[str] | None = None,
        data_loading_batch_size: int = 65536,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        subsample_samples: int = 10_000,
        max_predict_time: int = 60,
        fit_nodes: bool = True,
        adaptive_tree: bool = True,
        verbose: int = 1,
        output_metrics_dir: str | None = None,
        rf_kwargs={
        "n_estimators": 32,  # reduce trees
        "max_depth": 4,      # shallower trees
        "bootstrap": True,
        # "n_jobs": 4,        # override if needed
        # "show_progress": True,
        },
    ):
        """
        Stream a single parquet file and train a TabPFN model with RandomForest preprocessing.
        Supports regression ('r') and classification ('c') tasks.

        - Streams row-groups from parquet with tqdm progress.
        - Splits train/val/test using self.train_percent and self.validation_percentage unless overridden.
        - Fits RandomForestTabPFN[Regressor|Classifier].
        - Saves model and metrics history CSV.
        """
        
        try:
            from tabpfn_extensions import TabPFNRegressor, TabPFNClassifier
            from tabpfn_extensions.rf_pfn import (
                RandomForestTabPFNRegressor,
                RandomForestTabPFNClassifier,
            )
        except Exception as e:
            raise ImportError(
                f"TabPFN extensions not available: {e}. Please install 'tabpfn_extensions'."
            )
            
        import os
        import numpy as np
        import pandas as pd
        import joblib
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tqdm.auto import tqdm
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            mean_squared_error,
            r2_score,
            accuracy_score,
            f1_score,
        )

        if not self.single_parquet_path:
            raise ValueError("single_parquet_path not set.")

        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./model") or "./model"
        os.makedirs(output_metrics_dir, exist_ok=True)

        # Discover feature columns from parquet schema
        pf = pq.ParquetFile(self.single_parquet_path, memory_map=True)
        schema = pf.schema_arrow
        col_names = list(schema.names)
        if target_col not in col_names:
            raise ValueError(f"Target column '{target_col}' not found in parquet file.")
        drop_meta = {"date", "year", "month"}
        feature_cols = [c for c in col_names if c not in {target_col, *drop_meta}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        self.feature_names_ = feature_cols

        # Row-group metadata
        rg_offsets, rg_sizes = [], []
        running = 0
        for rg in range(pf.num_row_groups):
            nrg = pf.metadata.row_group(rg).num_rows
            rg_offsets.append(running)
            rg_sizes.append(nrg)
            running += nrg
        total_rows = running
        print(f"📊 Total rows: {total_rows:,} | Features: {len(feature_cols)}")

        # Split indices
        all_indices = np.arange(total_rows)
        tr_ratio = float(train_ratio) if train_ratio is not None else float(self.train_percent)
        vl_ratio = float(val_ratio) if val_ratio is not None else float(self.validation_percentage)

        print(f"🔀 Splitting data with train_ratio={tr_ratio}, val_ratio={vl_ratio}")

        # First split: train+val vs test
        trainval_indices, test_indices = train_test_split(
            all_indices,
            train_size=tr_ratio,
            random_state=42,
            shuffle=True,
        )

        # Second split: take validation from the training set (if requested)
        if vl_ratio > 0:
            train_indices, val_indices = train_test_split(
                trainval_indices,
                test_size=vl_ratio,
                random_state=42,
                shuffle=True,
            )
        else:
            train_indices = trainval_indices
            val_indices = np.array([], dtype=np.int64)

        print(
            f"🔄 Split -> train: {len(train_indices):,}, "
            f"val: {len(val_indices):,}, test: {len(test_indices):,}"
        )

        print(
            f"🔄 Split -> train: {len(train_indices):,}, "
            f"val: {len(val_indices):,}, test: {len(test_indices):,}"
        )

        # Helper to stream selected indices from parquet as batches
        def stream_selected(indices: np.ndarray, batch_size: int):
            if indices is None or len(indices) == 0:
                return
            sel = np.sort(indices.astype(np.int64, copy=False))
            for rg in range(pf.num_row_groups):
                offset = rg_offsets[rg]
                size = rg_sizes[rg]
                lo = offset
                hi = offset + size
                mask = (sel >= lo) & (sel < hi)
                if not np.any(mask):
                    continue
                local_idx = sel[mask] - lo  # positions within this row-group
                # Read row-group once
                tbl = pf.read_row_group(rg, columns=feature_cols + [target_col])
                df_rg = tbl.to_pandas()
                # Yield in smaller slices
                for i in range(0, local_idx.size, batch_size):
                    sl = local_idx[i : i + batch_size]
                    df_b = df_rg.iloc[sl]
                    Xb = df_b[feature_cols].to_numpy(dtype=np.float32, copy=False)
                    yb = df_b[target_col].to_numpy(copy=False)
                    yield Xb, yb

        # Data loading with progress
        train_X, train_y = [], []
        pbar_train = tqdm(
            total=len(train_indices),
            desc="Loading train data",
            unit="rows",
            position=0,
        )
        loaded_rows = 0
        for Xb, yb in stream_selected(train_indices, data_loading_batch_size):
            train_X.append(Xb)
            train_y.append(yb)
            loaded_rows += len(yb)
            pbar_train.update(len(yb))
        pbar_train.close()

        if len(train_X) == 0:
            raise ValueError("No training data loaded. Check indices/split settings.")

        X_train = np.vstack(train_X)
        y_train = np.concatenate(train_y)

        # Validation streaming (optional)
        X_val = y_val = None
        if len(val_indices) > 0:
            val_X, val_y = [], []
            pbar_val = tqdm(
                total=len(val_indices),
                desc="Loading val data",
                unit="rows",
                position=0,
            )
            for Xb, yb in stream_selected(val_indices, data_loading_batch_size):
                val_X.append(Xb)
                val_y.append(yb)
                pbar_val.update(len(yb))
            pbar_val.close()
            if len(val_X) > 0:
                X_val = np.vstack(val_X)
                y_val = np.concatenate(val_y)

        # Test streaming (optional)
        X_test = y_test = None
        if len(test_indices) > 0:
            test_X, test_y = [], []
            pbar_test = tqdm(
                total=len(test_indices),
                desc="Loading test data",
                unit="rows",
                position=0,
            )
            for Xb, yb in stream_selected(test_indices, data_loading_batch_size):
                test_X.append(Xb)
                test_y.append(yb)
                pbar_test.update(len(yb))
            pbar_test.close()
            if len(test_X) > 0:
                X_test = np.vstack(test_X)
                y_test = np.concatenate(test_test := test_y)

                # The previous line introduced a variable; fix scope
                del test_test  # no-op, to avoid lint warnings

        # Choose task: 'r' for regression, 'c' for classification
        is_classification = str(getattr(self, "_Model__task", self.__task)).lower().startswith("c")

        if is_classification:
            base = TabPFNClassifier(
                ignore_pretraining_limits=True,
                inference_config={"SUBSAMPLE_SAMPLES": int(subsample_samples)},
            )
            model = RandomForestTabPFNClassifier(
                tabpfn=base,
                verbose=int(verbose),
                max_predict_time=int(max_predict_time),
                fit_nodes=bool(fit_nodes),
                adaptive_tree=bool(adaptive_tree),
            )
        else:
            base = TabPFNRegressor(
                ignore_pretraining_limits=True,
                inference_config={"SUBSAMPLE_SAMPLES": int(1000)},
            )
            model = RandomForestTabPFNRegressor(
                tabpfn=base,
                verbose=int(verbose),
                max_predict_time=int(max_predict_time),
                fit_nodes=bool(fit_nodes),
                adaptive_tree=bool(adaptive_tree),
                show_progress=True,
                n_estimators=4,
            )

        # Training progress (single fit call; show a simple phase bar)
        train_phase = tqdm(total=1, desc="Training TabPFN RF", position=0, unit="phase")
        model.fit(X_train, y_train)
        train_phase.update(1)
        train_phase.close()

        self.__model = model

        history = []

        if is_classification:
            y_tr_pred = model.predict(X_train)
            tr_acc = float(accuracy_score(y_train, y_tr_pred))
            tr_f1 = float(f1_score(y_train, y_tr_pred, average="macro"))

            val_acc = val_f1 = None
            if X_val is not None:
                y_val_pred = model.predict(X_val)
                val_acc = float(accuracy_score(y_val, y_val_pred))
                val_f1 = float(f1_score(y_val, y_val_pred, average="macro"))

            test_acc = test_f1 = None
            if X_test is not None:
                y_test_pred = model.predict(X_test)
                test_acc = float(accuracy_score(y_test, y_test_pred))
                test_f1 = float(f1_score(y_test, y_test_pred, average="macro"))
                self.y_test = y_test
                self.y_test_pred = y_test_pred

            history.append({
                "task": "classification",
                "train_acc": tr_acc,
                "train_f1_macro": tr_f1,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
                "test_acc": test_acc,
                "test_f1_macro": test_f1,
                "train_rows": int(len(y_train)),
                "val_rows": int(len(y_val)) if y_val is not None else 0,
                "test_rows": int(len(y_test)) if y_test is not None else 0,
            })
        else:
            y_tr_pred = model.predict(X_train)
            tr_mse = mean_squared_error(y_train, y_tr_pred)
            tr_rmse = float(np.sqrt(tr_mse))
            tr_r2 = float(r2_score(y_train, y_tr_pred))

            val_rmse = val_r2 = None
            if X_val is not None:
                y_val_pred = model.predict(X_val)
                val_mse = mean_squared_error(y_val, y_val_pred)
                val_rmse = float(np.sqrt(val_mse))
                val_r2 = float(r2_score(y_val, y_val_pred))

            test_rmse = test_r2 = None
            if X_test is not None:
                y_test_pred = model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_rmse = float(np.sqrt(test_mse))
                test_r2 = float(r2_score(y_test, y_test_pred))
                self.y_test = y_test
                self.y_test_pred = y_test_pred

            history.append({
                "task": "regression",
                "train_rmse": tr_rmse,
                "train_r2": tr_r2,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "train_rows": int(len(y_train)),
                "val_rows": int(len(y_val)) if y_val is not None else 0,
                "test_rows": int(len(y_test)) if y_test is not None else 0,
            })

        # Save model and history
        model_path = os.path.join(output_metrics_dir, "best_tabpfn_rf_streaming.pkl")
        try:
            joblib.dump(model, model_path)
            print(f"✓ Saved TabPFN RF model -> {model_path}")
        except Exception as e:
            print(f"Warning: could not save model ({e})")

        hist_path = os.path.join(output_metrics_dir, "tabpfn_rf_streaming_history.csv")
        pd.DataFrame(history).to_csv(hist_path, index=False)
        print(f"✓ Saved training history -> {hist_path}")

        return history
    
    
    
    def train_single_parquet_tabpfn(
        self,
        *,
        target_col: str = "y",
        exclude_cols: list[str] | None = None,
        output_metrics_dir: str | None = None,
        subsample_samples: int = 10_000,
        max_predict_time: int = 60,
        fit_nodes: bool = True,
        adaptive_tree: bool = True,
        verbose: int = 1,
    ):
        try:
            from tabpfn_extensions import TabPFNRegressor
            from tabpfn_extensions.rf_pfn import RandomForestTabPFNRegressor
        except Exception as e:
            raise ImportError(
                f"TabPFN extensions not available: {e}. Please install 'tabpfn_extensions'."
            )

        import os
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        import joblib

        if not self.single_parquet_path:
            raise ValueError("single_parquet_path not set.")

        if output_metrics_dir is None:
            output_metrics_dir = getattr(self, "output_metrics_dir", "./model") or "./model"
        os.makedirs(output_metrics_dir, exist_ok=True)

        self._load_single_parquet_into_memory(target_col=target_col, exclude_cols=exclude_cols)

        x_train_full = self.__x_train
        y_train_full = self.__y_train

        if x_train_full is None or y_train_full is None:
            raise ValueError("Training split not prepared; check train_percent and parquet loading.")

        if float(self.validation_percentage) > 0.0:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train_full, y_train_full,
                test_size=self.validation_percentage,
                random_state=42
            )
        else:
            x_train, y_train = x_train_full, y_train_full
            x_val, y_val = None, None

        reg_base = TabPFNRegressor(
            ignore_pretraining_limits=True,
            inference_config={"SUBSAMPLE_SAMPLES": int(subsample_samples)}
        )

        model = RandomForestTabPFNRegressor(
            tabpfn=reg_base,
            verbose=int(verbose),
            max_predict_time=int(max_predict_time),
            fit_nodes=bool(fit_nodes),
            adaptive_tree=bool(adaptive_tree),
        )

        model.fit(x_train, y_train)
        self.__model = model

        y_tr_pred = model.predict(x_train)
        tr_rmse = float(np.sqrt(mean_squared_error(y_train, y_tr_pred)))
        tr_r2 = float(r2_score(y_train, y_tr_pred))

        val_rmse, val_r2 = None, None
        if x_val is not None and x_val.size > 0:
            y_val_pred = model.predict(x_val)
            val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
            val_r2 = float(r2_score(y_val, y_val_pred))

        test_rmse, test_r2 = None, None
        if getattr(self, "_Model__x_test", None) is not None and self.__x_test is not None and self.__x_test.size > 0:
            y_test_pred = model.predict(self.__x_test)
            test_rmse = float(np.sqrt(mean_squared_error(self.__y_test, y_test_pred)))
            test_r2 = float(r2_score(self.__y_test, y_test_pred))
            self.y_test = self.__y_test
            self.y_test_pred = y_test_pred

        history = [{
            "train_rmse": tr_rmse,
            "train_r2": tr_r2,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        }]

        model_path = os.path.join(output_metrics_dir, "best_tabpfn_model.pkl")
        try:
            joblib.dump(model, model_path)
            print(f"✓ Saved TabPFN model to {model_path}")
        except Exception as e:
            print(f"Warning: could not save model ({e})")

        history_path = os.path.join(output_metrics_dir, "tabpfn_training_history.csv")
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"✓ Saved training history to {history_path}")

        return history
    
    
    def train_single_parquet(
        self,
        n_epochs=50,
        lr=1e-3,
        target_col='y',
        exclude_cols=None,
        log_every=10,
        early_stopping_patience=10,          
        enable_tensorboard=True 
    ):
        """
        Full in‑RAM training on a single parquet file (non‑streaming) with:
        - Batch & epoch tqdm
        - Early stopping on validation RMSE
        - Best model checkpoint
        - Optional TensorBoard (per batch loss + epoch metrics)
        """
        import numpy as np, os
        import torch, torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        import pandas as pd
        from tqdm.auto import tqdm

        # Load data if not yet loaded
        if not hasattr(self, "feature_names_") or getattr(self, "x", None) is None or getattr(self, "y", None) is None:
            self._load_single_parquet_into_memory(target_col=target_col, exclude_cols=exclude_cols)

        # Non-tabformer: just fit (no early stopping logic)
        if self.model_name != 'tabformer':
            if self.train_percent == 1:
                self.__model.fit(self.x, self.y)
            else:
                self.__model.fit(self.__x_train, self.__y_train)
                if self.__x_test is not None:
                    self.__y_pred = self.__model.predict(self.__x_test)
            return None

        # Train / Val split from training portion
        x_train_full = self.__x_train
        y_train_full = self.__y_train
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full, y_train_full,
            test_size=self.validation_percentage,
            random_state=42
        )

        X_train_tensor = torch.as_tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor   = torch.as_tensor(x_val,   dtype=torch.float32)
        y_val_tensor   = torch.as_tensor(y_val,   dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset   = TensorDataset(X_val_tensor,   y_val_tensor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.__batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.workers_data_loaders,
            persistent_workers=self.workers_data_loaders > 0,
            prefetch_factor=self.prefetch_factor if self.workers_data_loaders > 0 else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.__batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.workers_data_loaders,
            persistent_workers=self.workers_data_loaders > 0,
            prefetch_factor=self.prefetch_factor if self.workers_data_loaders > 0 else None
        )

        # Re-init TabFormer if needed
        in_dim = X_train_tensor.shape[1]
        if not isinstance(self.__model, FTTransformer) or self.__model.head.in_features != in_dim:
            self.__model = FTTransformer(num_features=in_dim)
        self.__model.to(self.device)

        optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # TensorBoard
        writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                run_dir = os.path.join(self.output_metrics_dir, "tensorboard")
                os.makedirs(run_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=run_dir)
                print(f"TensorBoard logging -> {run_dir}")
            except Exception as e:
                print(f"TensorBoard disabled ({e})")
                writer = None

        best_state = None
        best_val_rmse = float("inf")
        patience_counter = 0
        global_step = 0
        history = []

        epoch_bar = tqdm(range(n_epochs), desc="Epochs", position=0)
        for epoch in epoch_bar:
            # ---- Train ----
            self.__model.train()
            batch_losses = []
            running = 0.0
            train_batch_bar = tqdm(train_loader, desc=f"Train {epoch+1}/{n_epochs}", leave=False, position=1)
            for i, (bx, by) in enumerate(train_batch_bar, 1):
                bx = bx.to(self.device); by = by.to(self.device)
                optimizer.zero_grad()
                out = self.__model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                li = loss.item()
                batch_losses.append(li)
                running += li
                global_step += 1
                if (i % log_every == 0) or (i == len(train_loader)):
                    train_batch_bar.set_postfix(avg_batch_loss=running / i)
                if writer:
                    writer.add_scalar("Batch/train_loss", li, global_step)

            epoch_train_batch_mse = float(np.mean(batch_losses))

            # ---- Eval Train & Val ----
            self.__model.eval()
            with torch.no_grad():
                # Train metrics
                tr_preds, tr_tgts = [], []
                for bx, by in train_loader:
                    bx = bx.to(self.device); by = by.to(self.device)
                    pb = self.__model(bx)
                    tr_preds.append(pb.detach().cpu())
                    tr_tgts.append(by.detach().cpu())
                y_tr_pred = torch.cat(tr_preds, 0).numpy().reshape(-1)
                y_tr_true = torch.cat(tr_tgts, 0).numpy().reshape(-1)

                val_preds, val_tgts = [], []
                for bx, by in val_loader:
                    bx = bx.to(self.device); by = by.to(self.device)
                    pb = self.__model(bx)
                    val_preds.append(pb.detach().cpu())
                    val_tgts.append(by.detach().cpu())
                y_val_pred = torch.cat(val_preds, 0).numpy().reshape(-1)
                y_val_true = torch.cat(val_tgts, 0).numpy().reshape(-1)

            # Metrics
            tr_mse = mean_squared_error(y_tr_true, y_tr_pred)
            tr_rmse = float(np.sqrt(tr_mse))
            tr_r2 = r2_score(y_tr_true, y_tr_pred)

            val_mse = mean_squared_error(y_val_true, y_val_pred)
            val_rmse = float(np.sqrt(val_mse))
            val_r2 = r2_score(y_val_true, y_val_pred)

            history.append({
                "epoch": epoch + 1,
                "batch_train_mse": epoch_train_batch_mse,
                "train_rmse": tr_rmse,
                "train_r2": tr_r2,
                "val_rmse": val_rmse,
                "val_r2": val_r2
            })

            # TensorBoard epoch metrics
            if writer:
                writer.add_scalar("Epoch/train_rmse", tr_rmse, epoch)
                writer.add_scalar("Epoch/train_r2", tr_r2, epoch)
                writer.add_scalar("Epoch/val_rmse", val_rmse, epoch)
                writer.add_scalar("Epoch/val_r2", val_r2, epoch)

            # Progress bar postfix
            epoch_bar.set_postfix(train_rmse=f"{tr_rmse:.4f}",
                                  train_r2=f"{tr_r2:.4f}",
                                  val_rmse=f"{val_rmse:.4f}",
                                  val_r2=f"{val_r2:.4f}",
                                  patience=f"{patience_counter}/{early_stopping_patience}")

            # ---- Early Stopping ----
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse = val_rmse
                patience_counter = 0
                # Save best
                save_path = self.output_metrics_dir + "/best_model" + f"_{epoch+1}.pth"
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                best_state = {
                    "epoch": epoch,
                    "model_state_dict": self.__model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_rmse": val_rmse,
                    "val_r2": val_r2
                }
                torch.save(best_state, f"{save_path}.pth")
                print(f"  (Best model saved at epoch {epoch+1} with val_rmse={val_rmse:.4f} to {save_path})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1} (no val improvement {early_stopping_patience} epochs).")
                    break

        # Load best model if saved
        if best_state is not None:
            self.__model.load_state_dict(best_state["model_state_dict"])
            print(f"Loaded best model (epoch {best_state['epoch']+1}, val_rmse={best_state['val_rmse']:.4f})")

        # Test set evaluation
        if self.__x_test is not None and self.__x_test.size > 0:
            X_test_tensor = torch.as_tensor(self.__x_test, dtype=torch.float32)
            y_test_tensor = torch.as_tensor(self.__y_test, dtype=torch.float32).unsqueeze(1)
            test_loader = DataLoader(
                TensorDataset(X_test_tensor, y_test_tensor),
                batch_size=self.__batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.workers_data_loaders,
                persistent_workers=self.workers_data_loaders > 0,
                prefetch_factor=self.prefetch_factor if self.workers_data_loaders > 0 else None
            )
            self.__model.eval()
            t_preds, t_tgts = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx = bx.to(self.device); by = by.to(self.device)
                    pb = self.__model(bx)
                    t_preds.append(pb.detach().cpu())
                    t_tgts.append(by.detach().cpu())
            y_test_pred = torch.cat(t_preds, 0).numpy().reshape(-1)
            y_test_true = torch.cat(t_tgts, 0).numpy().reshape(-1)
            test_mse = mean_squared_error(y_test_true, y_test_pred)
            test_rmse = float(np.sqrt(test_mse))
            test_r2 = r2_score(y_test_true, y_test_pred)
            print(f"Test -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
            if writer:
                writer.add_scalar("Test/rmse", test_rmse, 0)
                writer.add_scalar("Test/r2", test_r2, 0)
            self.y_test = y_test_true
            self.y_test_pred = y_test_pred

        # Save history
        save_path_csv = os.path.join(self.output_metrics_dir, "training_history.csv")
        pd.DataFrame(history).to_csv(
            save_path_csv,
            index=False
        )
        print(f"Saved training history to {save_path_csv}")

        if writer:
            # Close TensorBoard writer
            writer.close()
            print("TensorBoard writer closed.")
            print(f"✓ TensorBoard logs saved to {self.output_metrics_dir}/tensorboard")
            print(f"  To view: tensorboard --logdir={self.output_metrics_dir}/tensorboard")
            
            
        return history
    
    def train_single_parquet_v1(
        self,
        n_epochs=50,
        lr=1e-3,
        target_col='y',
        exclude_cols=None,
        log_every=10
    ):
        """
        Full in‑RAM training on a single parquet file (non‑streaming).
        Loads file once, splits into train/val/test like classic in‑memory path
        and trains TabFormer (PyTorch) or classic ML model.
        """
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        import pandas as pd
        from tqdm.auto import tqdm

        if self.model_name not in ('tabformer', 'catboost', 'xgboost', 'random_forest',
                                     'decision_tree', 'gradient_boosting', 'svm',
                                     'linear_regression', 'adaboost', 'knn'):
            raise NotImplementedError("train_single_parquet only implemented for current supervised models.")

        # Load data if not already loaded
        if not hasattr(self, "feature_names_") or self.x is None or self.y is None or (
            self._single_parquet_mode and not hasattr(self, "_single_loaded")
        ):
            self._load_single_parquet_into_memory(target_col=target_col, exclude_cols=exclude_cols)
            self._single_loaded = True

        # If classic ML (non tabformer) just fit once
        if self.model_name != 'tabformer':
            if self.train_percent == 1:
                self.__model.fit(self.x, self.y)
            else:
                self.__model.fit(self.__x_train, self.__y_train)
                if self.__x_test is not None:
                    self.__y_pred = self.__model.predict(self.__x_test)
            return None

        # TabFormer training
        # Further split training into train/val
        x_train_full = self.__x_train
        y_train_full = self.__y_train
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full, y_train_full,
            test_size=self.validation_percentage,
            random_state=42
        )

        X_train_tensor = torch.as_tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor   = torch.as_tensor(x_val,   dtype=torch.float32)
        y_val_tensor   = torch.as_tensor(y_val,   dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset   = TensorDataset(X_val_tensor,   y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.__batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=self.__batch_size, shuffle=False)

        # Re-init TabFormer with correct input dim if needed
        in_dim = X_train_tensor.shape[1]
        if not isinstance(self.__model, FTTransformer) or self.__model.head.in_features != in_dim:
            self.__model = FTTransformer(num_features=in_dim)
        self.__model.to(self.device)

        optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        training_history = []

        epoch_bar = tqdm(range(n_epochs), desc="Epochs", position=0)
        for epoch in epoch_bar:
            self.__model.train()
            batch_losses = []
            train_batch_bar = tqdm(train_loader, desc=f"Train {epoch+1}/{n_epochs}", leave=False, position=1)
            running = 0.0
            total_train_batches = len(train_loader)
            for i, (bx, by) in enumerate(train_batch_bar, 1):
                bx = bx.to(self.device); by = by.to(self.device)
                optimizer.zero_grad()
                out = self.__model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                li = loss.item()
                batch_losses.append(li)
                running += li
                if i % log_every == 0 or i == total_train_batches:
                    pct = 100.0 * i / total_train_batches
                    train_batch_bar.set_postfix(avg_batch_loss=running / i, progress=f"{pct:5.1f}%")

            # Eval
            self.__model.eval()
            with torch.no_grad():
                # train metrics
                tr_preds, tr_targets = [], []
                for bx, by in train_loader:
                    bx = bx.to(self.device); by = by.to(self.device)
                    pb = self.__model(bx)
                    tr_preds.append(pb.detach().cpu()); tr_targets.append(by.detach().cpu())
                y_train_pred = torch.cat(tr_preds, 0).numpy().reshape(-1)
                y_train_true = torch.cat(tr_targets, 0).numpy().reshape(-1)

                val_preds, val_targets = [], []
                for bx, by in val_loader:
                    bx = bx.to(self.device); by = by.to(self.device)
                    pb = self.__model(bx)
                    val_preds.append(pb.detach().cpu()); val_targets.append(by.detach().cpu())
                y_val_pred = torch.cat(val_preds, 0).numpy().reshape(-1)
                y_val_true = torch.cat(val_targets, 0).numpy().reshape(-1)

            train_mse = mean_squared_error(y_train_true, y_train_pred)
            train_rmse = float(np.sqrt(train_mse))
            train_r2 = r2_score(y_train_true, y_train_pred)

            val_mse = mean_squared_error(y_val_true, y_val_pred)
            val_rmse = float(np.sqrt(val_mse))
            val_r2 = r2_score(y_val_true, y_val_pred)

            training_history.append({
                "epoch": epoch + 1,
                "batch_train_mse": float(np.mean(batch_losses)),
                "train_rmse": train_rmse,
                "train_r2": train_r2,
                "val_rmse": val_rmse,
                "val_r2": val_r2
            })

            epoch_bar.set_postfix(train_rmse=f"{train_rmse:.4f}",
                                  train_r2=f"{train_r2:.4f}",
                                  val_rmse=f"{val_rmse:.4f}",
                                  val_r2=f"{val_r2:.4f}")

        # Test evaluation if split
        if self.__x_test is not None and self.__x_test.size > 0:
            X_test_tensor = torch.as_tensor(self.__x_test, dtype=torch.float32)
            y_test_tensor = torch.as_tensor(self.__y_test, dtype=torch.float32).unsqueeze(1)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)
            self.__model.eval()
            test_preds, test_targets = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx = bx.to(self.device); by = by.to(self.device)
                    pb = self.__model(bx)
                    test_preds.append(pb.detach().cpu()); test_targets.append(by.detach().cpu())
            y_test_pred = torch.cat(test_preds, 0).numpy().reshape(-1)
            y_test_true = torch.cat(test_targets, 0).numpy().reshape(-1)
            test_mse = mean_squared_error(y_test_true, y_test_pred)
            test_rmse = float(np.sqrt(test_mse))
            test_r2 = r2_score(y_test_true, y_test_pred)
            print(f"Test -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
            self.y_test = y_test_true
            self.y_test_pred = y_test_pred

        import pandas as pd
        pd.DataFrame(training_history).to_csv("training_history_single_parquet.csv", index=False)
        print("Saved training_history_single_parquet.csv")
        return training_history
    
    
    def train_streaming(
        self,
        epochs: int | None = None,
        lr: float = 1e-3,
        log_every: int = 50,
        patience: int = 10,  # Early stopping patience
        checkpoint_path: str = "best_model",  # Where to save best model
        output_metrics_dir: str | None = "./model",
    ):
        """
        Streaming trainer that reports:
        train_r2=..., train_rmse=..., val_r2=..., val_rmse=...
        per epoch, similar to the in-memory `train()` method.
        
        Features:
        - Early stopping when validation metrics don't improve
        - Model checkpointing to save best model
        - TensorBoard logging of metrics
        """
        import torch
        import torch.nn as nn
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from sklearn.metrics import mean_squared_error, r2_score
        import os
        from datetime import datetime
    
        if self.train_loader is None:
            raise ValueError("train_streaming requires self.train_loader (set via parquet_batches_data).")
    
        E = int(epochs)
        history = []
        
        # Set up TensorBoard if available
        try:
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_available = True
            # Create unique run name with timestamp
            run_name = f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_dir = os.path.join(output_metrics_dir + "/tensorboard", run_name)
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logs will be saved to: {log_dir}")
        except ImportError:
            tensorboard_available = False
            print("TensorBoard not available. Install with: pip install tensorboard")
    
        if self.model_name == 'tabformer':
            self.__model.to(self.device)
            optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr)
            criterion = nn.MSELoss()
    
        # Early stopping variables
        best_val_metric = float('inf')  # For RMSE (lower is better)
        counter = 0  # Counter for patience
        best_model_state = None
    
        def _metrics_over_loader(loader, desc="Eval", show_progress=True):
            """
            Compute (rmse, r2) over all batches of a loader.
            Uses GPU if available, accumulates tensors on device, single CPU transfer at end.
            Returns None if loader yields nothing.
            """
            import torch
            preds, trues = [], []
            got_any = False
            self.__model.eval()

            # Try tqdm
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(loader, desc=desc, leave=False) if show_progress else loader
            except Exception:
                iterator = loader

            with torch.no_grad():
                for Xb, yb in iterator:
                    got_any = True
                    Xb = Xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    pb = self.__model(Xb)
                    preds.append(pb.detach())  # stay on device
                    trues.append(yb.detach())

            if not got_any:
                return None

            # Single device→host transfer
            y_pred = torch.cat(preds, 0).view(-1).cpu().numpy()
            y_true = torch.cat(trues, 0).view(-1).cpu().numpy()

            mse = mean_squared_error(y_true, y_pred)
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_true, y_pred))
            return rmse, r2
    
        # Initialize global step counter for TensorBoard
        global_step = 0
        
        for ep in tqdm(range(E), desc="Epochs", position=0):
            # ---- Train pass ----
            self.__model.train()
            running = 0.0
            steps = 0
            train_bar = tqdm(self.train_loader, desc=f"Train {ep+1}/{E}", leave=False, position=1)
            
            batch_count = 0  # Count batches processed in this epoch
            
            for i, (Xb, yb) in enumerate(train_bar, 1):
                Xb = Xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
    
                optimizer.zero_grad(set_to_none=True)
                pb = self.__model(Xb)
                loss = criterion(pb, yb)
                loss.backward()
                optimizer.step()
    
                #lv = float(loss.detach().cpu())
                lv = loss.detach().item()   # scalar float
                running += lv
                steps += 1
                batch_count += 1
                global_step += 1  # Increment global step counter
                
                if (i % log_every) == 0:
                    batch_loss = running / max(1, steps)
                    train_bar.set_postfix(avg_batch_loss=batch_loss)
                    # Log to TensorBoard
                    if tensorboard_available:
                        writer.add_scalar('Batch/train_loss', batch_loss, global_step)

            print("\n Epoch-end metrics on TRAIN:")
            # ---- Epoch-end metrics on TRAIN ----
            self.__model.eval()
            train_metrics = _metrics_over_loader(self.train_loader)
            if train_metrics is None:
                # No train batches produced (shouldn't happen, but be safe)
                train_rmse, train_r2 = None, None
            else:
                train_rmse, train_r2 = train_metrics

            print("\n Epoch-end metrics on VAL:")
            # ---- Epoch-end metrics on VAL (optional) ----
            if self.val_loader is not None:
                val_metrics = _metrics_over_loader(self.val_loader)
                if val_metrics is None:
                    val_rmse, val_r2 = None, None
                else:
                    val_rmse, val_r2 = val_metrics
    
                # Early stopping check - using RMSE as the metric to monitor
                if val_rmse is not None:
                    current_metric = val_rmse  # For RMSE (lower is better)
                    
                    if current_metric < best_val_metric:  # For RMSE (lower is better)
                        best_val_metric = current_metric
                        counter = 0
                        
                        # Save best model state
                        best_model_state = {
                            'epoch': ep,
                            'model_state_dict': self.__model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_rmse': val_rmse,
                            'val_r2': val_r2
                        }
                        
                        # Save best model checkpoint
                        os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else '.', exist_ok=True)
                        model_path = f"{checkpoint_path}.pth"
                        torch.save(best_model_state, model_path)
                        print(f"✓ Saved best model at epoch {ep+1} with val_rmse={val_rmse:.4f}, val_r2={val_r2:.4f}")
                    else:
                        counter += 1
                        if counter >= patience:
                            print(f"⚠ Early stopping triggered after {ep+1} epochs (no improvement for {patience} epochs)")
                            break
            else:
                val_rmse, val_r2 = None, None
    
            # ---- Print like in-memory path ----
            parts = []
            if train_r2 is not None:  parts.append(f"train_r2={train_r2:.4f}")
            if train_rmse is not None: parts.append(f"train_rmse={train_rmse:.4f}")
            if val_r2 is not None:    parts.append(f"val_r2={val_r2:.4f}")
            if val_rmse is not None:  parts.append(f"val_rmse={val_rmse:.4f}")
            if counter > 0:          parts.append(f"early_stop={counter}/{patience}")
            if parts:
                tqdm.write(", ".join(parts))
            else:
                tqdm.write(f"Epoch {ep+1}/{E} — no metrics available (empty loaders)")
    
            # ---- Log history ----
            history.append({
                "epoch": ep + 1,
                "train_rmse": train_rmse if train_rmse is not None else np.nan,
                "train_r2":   train_r2   if train_r2   is not None else np.nan,
                "val_rmse":   val_rmse   if val_rmse   is not None else np.nan,
                "val_r2":     val_r2     if val_r2     is not None else np.nan,
            })
            
            # ---- Log to TensorBoard ----
            if tensorboard_available:
                if train_rmse is not None:
                    writer.add_scalar('Epoch/train_rmse', train_rmse, ep)
                if train_r2 is not None:
                    writer.add_scalar('Epoch/train_r2', train_r2, ep)
                if val_rmse is not None:
                    writer.add_scalar('Epoch/val_rmse', val_rmse, ep)
                if val_r2 is not None:
                    writer.add_scalar('Epoch/val_r2', val_r2, ep)
    
        # Save history CSV (parity with in-memory)
        dataframe_path = self.output_metrics_dir
        pd.DataFrame(history).to_csv(f"{dataframe_path}/training_history_streaming.csv", index=False)
        print("Training history exported to training_history_streaming.csv")
        
        # ---- Optional: Load best model for final evaluation ----
        if best_model_state is not None:
            print(f"ℹ Loading best model from epoch {best_model_state['epoch']+1}")
            self.__model.load_state_dict(best_model_state['model_state_dict'])
    
        print("Final evaluation on TEST set:")
        # ---- Optional Test ----
        if self.test_loader is not None:
            self.__model.eval()
            test = _metrics_over_loader(self.test_loader)
            if test is None:
                print("Test loader is empty — skipping test metrics.")
            else:
                test_rmse, test_r2 = test
                print(f"Performance on Test set (streamed) -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
                
                # Log test metrics to TensorBoard
                if tensorboard_available:
                    writer.add_scalar('Test/rmse', test_rmse, 0)
                    writer.add_scalar('Test/r2', test_r2, 0)
                
                self.y_test = None  # you can fill these if you want full arrays
                self.y_test_pred = None
        
        # Close TensorBoard writer
        if tensorboard_available:
            writer.close()
            print(f"✓ TensorBoard logs saved to {log_dir}")
            print(f"  To view: tensorboard --logdir={output_metrics_dir}/tensorboard")
        
        return history
    
    
    def train_streaming_v1(
        self,
        epochs: int | None = None,
        lr: float = 1e-3,
        log_every: int = 50,
    ):
        """
        Streaming trainer that reports:
        train_r2=..., train_rmse=..., val_r2=..., val_rmse=...
        per epoch, similar to the in-memory `train()` method.
        """
        import torch
        import torch.nn as nn
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from sklearn.metrics import mean_squared_error, r2_score

        if self.train_loader is None:
            raise ValueError("train_streaming requires self.train_loader (set via parquet_batches_data).")

        E = int(epochs or self.__epochs)

        if self.model_name == 'tabformer':
            self.__model.to(self.device)
            optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr)
            criterion = nn.MSELoss()

        history = []

        def _metrics_over_loader(loader):
            """Return (rmse, r2) over all batches of a loader; None if loader empty."""
            preds, trues = [], []
            with torch.no_grad():
                # create a fresh iterator; IterableDataset supports multiple passes
                it = iter(loader)
                got_any = False
                for Xb, yb in it:
                    got_any = True
                    Xb = Xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    pb = self.__model(Xb)
                    preds.append(pb.detach().cpu())
                    trues.append(yb.detach().cpu())
            if not got_any:
                return None
            y_pred = torch.cat(preds, 0).numpy().reshape(-1)
            y_true = torch.cat(trues, 0).numpy().reshape(-1)
            mse = mean_squared_error(y_true, y_pred)
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_true, y_pred))
            return rmse, r2

        for ep in tqdm(range(E), desc="Epochs", position=0):
            # ---- Train pass ----
            self.__model.train()
            running = 0.0
            steps = 0
            train_bar = tqdm(self.train_loader, desc=f"Train {ep+1}/{E}", leave=False, position=1)
            for i, (Xb, yb) in enumerate(train_bar, 1):
                Xb = Xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pb = self.__model(Xb)
                loss = criterion(pb, yb)
                loss.backward()
                optimizer.step()

                lv = float(loss.detach().cpu())
                running += lv
                steps += 1
                if (i % log_every) == 0:
                    train_bar.set_postfix(avg_batch_loss=running / max(1, steps))

            # ---- Epoch-end metrics on TRAIN ----
            self.__model.eval()
            train_metrics = _metrics_over_loader(self.train_loader)
            if train_metrics is None:
                # No train batches produced (shouldn't happen, but be safe)
                train_rmse, train_r2 = None, None
            else:
                train_rmse, train_r2 = train_metrics

            # ---- Epoch-end metrics on VAL (optional) ----
            if self.val_loader is not None:
                val_metrics = _metrics_over_loader(self.val_loader)
                if val_metrics is None:
                    val_rmse, val_r2 = None, None
                else:
                    val_rmse, val_r2 = val_metrics
            else:
                val_rmse, val_r2 = None, None

            # ---- Print like in-memory path ----
            parts = []
            if train_r2 is not None:  parts.append(f"train_r2={train_r2:.4f}")
            if train_rmse is not None: parts.append(f"train_rmse={train_rmse:.4f}")
            if val_r2 is not None:    parts.append(f"val_r2={val_r2:.4f}")
            if val_rmse is not None:  parts.append(f"val_rmse={val_rmse:.4f}")
            if parts:
                tqdm.write(", ".join(parts))
            else:
                tqdm.write(f"Epoch {ep+1}/{E} — no metrics available (empty loaders)")

            # ---- Log history ----
            history.append({
                "epoch": ep + 1,
                "train_rmse": train_rmse if train_rmse is not None else np.nan,
                "train_r2":   train_r2   if train_r2   is not None else np.nan,
                "val_rmse":   val_rmse   if val_rmse   is not None else np.nan,
                "val_r2":     val_r2     if val_r2     is not None else np.nan,
            })

        # Save history CSV (parity with in-memory)
        pd.DataFrame(history).to_csv("training_history_streaming.csv", index=False)
        print("Training history exported to training_history_streaming.csv")

        # ---- Optional Test ----
        if self.test_loader is not None:
            self.__model.eval()
            test = _metrics_over_loader(self.test_loader)
            if test is None:
                print("Test loader is empty — skipping test metrics.")
            else:
                test_rmse, test_r2 = test
                print(f"Performance on Test set (streamed) -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
                self.y_test = None  # you can fill these if you want full arrays
                self.y_test_pred = None


    def train_streaming_old(
        self,
        epochs: int | None = None,
        lr: float = 1e-3,
        log_every: int = 50,
        patience: int = 10,
        checkpoint_path: str = "best_model",
        output_metrics_dir: str | None = "./model",
    ):
        import torch, torch.nn as nn, numpy as np, pandas as pd, os
        from tqdm.auto import tqdm
        from sklearn.metrics import mean_squared_error, r2_score
        from datetime import datetime

        if self.train_loader is None:
            raise ValueError("train_streaming requires self.train_loader.")

        E = int(epochs)
        history = []

        # TensorBoard (unchanged) ...
        # ...

        if self.model_name != 'tabformer':
            raise NotImplementedError("train_streaming implemented for tabformer only.")

        optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # --- Accelerate prepare (replaces DataParallel) ---
        if self.use_accelerate and self.accelerator is not None:
            self.__model, optimizer, self.train_loader, self.val_loader, self.test_loader = \
                self.accelerator.prepare(
                    self.__model,
                    optimizer,
                    self.train_loader,
                    self.val_loader if self.val_loader is not None else [],
                    self.test_loader if self.test_loader is not None else []
                )
            device_ref = self.accelerator.device
        else:
            # classic path
            self.__model.to(self.device)
            device_ref = self.device

        train_est = getattr(self.train_loader, "total_batches", None)
        val_est   = getattr(self.val_loader, "total_batches", None) if self.val_loader else None
        test_est  = getattr(self.test_loader, "total_batches", None) if self.test_loader else None

        best_val_metric = float('inf')
        counter = 0
        best_model_state = None
        global_step = 0

        def _metrics_over_loader(loader, desc="Eval", est=None):
            if loader is None:
                return None
            preds, trues = [], []
            self.__model.eval()
            iterator = tqdm(loader, desc=desc, leave=False)
            processed = 0
            with torch.no_grad():
                for Xb, yb in iterator:
                    processed += 1
                    if not self.use_accelerate:
                        Xb = Xb.to(device_ref, non_blocking=True)
                        yb = yb.to(device_ref, non_blocking=True)
                    pb = self.__model(Xb)
                    preds.append(pb.detach())
                    trues.append(yb.detach())
                    if est:
                        pct = 100.0 * processed / est
                        iterator.set_postfix(progress=f"{pct:5.1f}%")
            if not preds:
                return None
            if self.use_accelerate:
                import torch
                preds_cat = torch.cat(preds)
                trues_cat = torch.cat(trues)
                preds_g = self.accelerator.gather(preds_cat).view(-1)
                trues_g = self.accelerator.gather(trues_cat).view(-1)
                y_pred = preds_g.cpu().numpy()
                y_true = trues_g.cpu().numpy()
                # On non-zero ranks skip metric printing (only rank 0 prints)
                if not self.accelerator.is_main_process:
                    return (0.0, 0.0)
            else:
                y_pred = torch.cat(preds, 0).view(-1).cpu().numpy()
                y_true = torch.cat(trues, 0).view(-1).cpu().numpy()
            mse = mean_squared_error(y_true, y_pred)
            return float(np.sqrt(mse)), float(r2_score(y_true, y_pred))

        for ep in tqdm(range(E), desc="Epochs", position=0):
            self.__model.train()
            running = 0.0
            steps = 0
            processed_batches = 0
            train_bar = tqdm(self.train_loader, desc=f"Train {ep+1}/{E}", leave=False, position=1)

            optimizer.zero_grad(set_to_none=True)
            for i, (Xb, yb) in enumerate(train_bar, 1):
                processed_batches += 1
                if not self.use_accelerate:
                    Xb = Xb.to(device_ref, non_blocking=True)
                    yb = yb.to(device_ref, non_blocking=True)
                pb = self.__model(Xb)
                loss = criterion(pb, yb) / self.grad_accum_steps
                if self.use_accelerate:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                if i % self.grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                lv = float(loss.detach().cpu()) * self.grad_accum_steps
                running += lv
                steps += 1
                global_step += 1
                if (i % log_every) == 0 or train_est:
                    avg_loss = running / max(1, steps)
                    postfix = {"avg_batch_loss": f"{avg_loss:.4f}"}
                    if train_est:
                        pct = 100.0 * processed_batches / train_est
                        postfix["progress"] = f"{pct:5.1f}%"
                    train_bar.set_postfix(**postfix)

            # Sync before metrics
            if self.use_accelerate:
                self.accelerator.wait_for_everyone()

            train_metrics = _metrics_over_loader(self.train_loader, "Eval-Train", train_est)
            train_rmse, train_r2 = train_metrics if train_metrics else (None, None)

            val_rmse = val_r2 = None
            if self.val_loader is not None:
                val_metrics = _metrics_over_loader(self.val_loader, "Eval-Val", val_est)
                if val_metrics:
                    val_rmse, val_r2 = val_metrics
                    if self.use_accelerate and not self.accelerator.is_main_process:
                        # Skip early-stop logic on non-main processes
                        pass
                    else:
                        if val_rmse < best_val_metric:
                            best_val_metric = val_rmse
                            counter = 0
                            if (not self.use_accelerate) or self.accelerator.is_main_process:
                                best_model_state = {
                                    'epoch': ep,
                                    'model_state_dict': (self.__model.module.state_dict()
                                                         if hasattr(self.__model, "module") else self.__model.state_dict()),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'val_rmse': val_rmse,
                                    'val_r2': val_r2
                                }
                                os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                                torch.save(best_model_state, f"{checkpoint_path}.pth")
                        else:
                            counter += 1
                            if counter >= patience:
                                if (not self.use_accelerate) or self.accelerator.is_main_process:
                                    print(f"⚠ Early stopping (no val improvement {patience} epochs)")
                                break

            if (not self.use_accelerate) or self.accelerator.is_main_process:
                parts = []
                if train_r2 is not None: parts.append(f"train_r2={train_r2:.4f}")
                if train_rmse is not None: parts.append(f"train_rmse={train_rmse:.4f}")
                if val_r2 is not None: parts.append(f"val_r2={val_r2:.4f}")
                if val_rmse is not None: parts.append(f"val_rmse={val_rmse:.4f}")
                if counter > 0: parts.append(f"early_stop={counter}/{patience}")
                tqdm.write(", ".join(parts) if parts else f"Epoch {ep+1}/{E} no metrics")

            history.append({
                "epoch": ep + 1,
                "train_rmse": train_rmse if train_rmse is not None else np.nan,
                "train_r2":   train_r2   if train_r2   is not None else np.nan,
                "val_rmse":   val_rmse   if val_rmse   is not None else np.nan,
                "val_r2":     val_r2     if val_r2     is not None else np.nan,
            })

        if (not self.use_accelerate) or self.accelerator.is_main_process:
            os.makedirs(output_metrics_dir, exist_ok=True)
            pd.DataFrame(history).to_csv(f"{output_metrics_dir}/training_history_streaming.csv", index=False)

            if best_model_state is not None:
                print(f"ℹ Loading best model (epoch {best_model_state['epoch']+1})")
                # unwrap if DataParallel or Accelerate
                target_model = self.__model.module if hasattr(self.__model, "module") else self.__model
                target_model.load_state_dict(best_model_state['model_state_dict'])

            if self.test_loader is not None:
                test_metrics = _metrics_over_loader(self.test_loader, "Eval-Test", test_est)
                if test_metrics:
                    test_rmse, test_r2 = test_metrics
                    print(f"Test -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

        if self.use_accelerate:
            self.accelerator.wait_for_everyone()

        return history    
    
    def train_streaming_ml(
        self,
        batch_accumulation_size: int = 10000,  # Number of samples to accumulate before training/updating
        log_every: int = 50,
        output_metrics_dir: str | None = "./model",
        **model_params                         # Additional params for specific model types
    ):
        """
        Streaming trainer for traditional ML models (CatBoost, XGBoost, Random Forest, etc.).
        Unlike deep learning models, traditional ML models typically train in a single pass.
        
        Args:
            batch_accumulation_size: Number of samples to accumulate before updating the model
            log_every: How often to log batch progress
            **model_params: Additional model-specific parameters
        """
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from sklearn.metrics import mean_squared_error, r2_score
        
        if self.train_loader is None:
            raise ValueError("train_streaming_ml requires self.train_loader (set via parquet_batches_data).")
        
        history = []
        
        # Configure model-specific parameters
        if self.model_name == 'catboost':
            # CatBoost specific handling - initialize with quiet mode for streaming
            from catboost import CatBoostRegressor, CatBoostClassifier, Pool
            if self.__task == 'r':
                if not model_params.get('verbose', False):
                    self.__model.set_params(verbose=False)
            else:
                if not model_params.get('verbose', False):
                    self.__model.set_params(verbose=False)
        
        elif self.model_name == 'xgboost':
            # XGBoost specific handling
            import xgboost as xgb
            # Use DMatrix for efficient XGBoost training
            use_dmatrix = model_params.get('use_dmatrix', True)
        
        def _metrics_over_loader(loader, desc="Evaluating"):
            """Return (rmse, r2) over all batches of a loader; None if loader empty."""
            all_preds, all_trues = [], []
            got_any = False
            
            # Collect all data from the loader
            for Xb, yb in tqdm(loader, desc=desc, leave=False):
                got_any = True
                
                # Convert tensors to numpy if needed
                if hasattr(Xb, 'numpy'):
                    Xb = Xb.numpy()
                if hasattr(yb, 'numpy'):
                    yb = yb.numpy()
                
                # Reshape y if needed
                if len(yb.shape) > 1:
                    yb = yb.reshape(-1)
                    
                # Make prediction with current model
                preds = self.__model.predict(Xb)
                
                all_preds.append(preds)
                all_trues.append(yb)
            
            if not got_any:
                return None
            
            # Concatenate all predictions and true values
            y_pred = np.concatenate(all_preds, axis=0)
            y_true = np.concatenate(all_trues, axis=0)
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_true, y_pred))
            
            return rmse, r2
        
        # Process data in a single pass
        X_accumulated, y_accumulated = [], []
        samples_seen = 0
        updates = 0
        
        # Process training data
        train_bar = tqdm(self.train_loader, desc="Training", position=0)
        for i, (Xb, yb) in enumerate(train_bar, 1):
            # Convert tensors to numpy if needed
            if hasattr(Xb, 'numpy'):
                Xb = Xb.numpy()
            if hasattr(yb, 'numpy'):
                yb = yb.numpy()
                
            # Reshape y if needed
            if len(yb.shape) > 1:
                yb = yb.reshape(-1)
            
            # Accumulate samples
            X_accumulated.append(Xb)
            y_accumulated.append(yb)
            
            samples_seen += len(Xb)
            
            # Update status bar
            if i % log_every == 0:
                train_bar.set_postfix(samples=samples_seen, updates=updates)
            
            # If we've accumulated enough samples, train/update the model
            if samples_seen >= batch_accumulation_size:
                X_batch = np.vstack(X_accumulated)
                y_batch = np.concatenate(y_accumulated)
                
                # Model-specific training
                if self.model_name == 'catboost':
                    # CatBoost training using Pool
                    train_pool = Pool(X_batch, y_batch)
                    self.__model.fit(train_pool, verbose=False)
                
                elif self.model_name == 'xgboost':
                    # XGBoost training
                    if use_dmatrix:
                        dtrain = xgb.DMatrix(X_batch, label=y_batch)
                        self.__model.fit(dtrain)
                    else:
                        # Regular fit method
                        self.__model.fit(X_batch, y_batch)
                
                else:
                    # Default training
                    self.__model.fit(X_batch, y_batch)
                
                # Reset accumulation
                X_accumulated, y_accumulated = [], []
                samples_seen = 0
                updates += 1
        
        # Handle any remaining samples
        if X_accumulated:
            X_batch = np.vstack(X_accumulated)
            y_batch = np.concatenate(y_accumulated)
            
            if self.model_name == 'catboost':
                train_pool = Pool(X_batch, y_batch)
                self.__model.fit(train_pool, verbose=False)
            
            elif self.model_name == 'xgboost':
                if use_dmatrix:
                    dtrain = xgb.DMatrix(X_batch, label=y_batch)
                    self.__model.fit(dtrain)
                else:
                    self.__model.fit(X_batch, y_batch)
            
            else:
                self.__model.fit(X_batch, y_batch)
        
        # Evaluate on train dataset
        train_metrics = _metrics_over_loader(self.train_loader, desc="Eval Train")
        if train_metrics is None:
            # No train batches produced
            train_rmse, train_r2 = None, None
        else:
            train_rmse, train_r2 = train_metrics
        
        # Evaluate on validation dataset if available
        if self.val_loader is not None:
            val_metrics = _metrics_over_loader(self.val_loader, desc="Eval Val")
            if val_metrics is None:
                val_rmse, val_r2 = None, None
            else:
                val_rmse, val_r2 = val_metrics
        else:
            val_rmse, val_r2 = None, None
        
        # Print metrics
        parts = []
        if train_r2 is not None:
            parts.append(f"train_r2={train_r2:.4f}")
        if train_rmse is not None:
            parts.append(f"train_rmse={train_rmse:.4f}")
        if val_r2 is not None:
            parts.append(f"val_r2={val_r2:.4f}")
        if val_rmse is not None:
            parts.append(f"val_rmse={val_rmse:.4f}")
            
        if parts:
            print(", ".join(parts))
        else:
            print("No metrics available (empty loaders)")
        
        # Record metrics in history
        history.append({
            "train_rmse": train_rmse if train_rmse is not None else np.nan,
            "train_r2": train_r2 if train_r2 is not None else np.nan,
            "val_rmse": val_rmse if val_rmse is not None else np.nan,
            "val_r2": val_r2 if val_r2 is not None else np.nan,
        })
        
        
        # Save history to CSV
        dataframe_path = self.output_metrics_dir
        pd.DataFrame(history).to_csv(f"{dataframe_path}/training_history_streaming_ml.csv", index=False)
        print("Training history exported to training_history_streaming_ml.csv")
        
        # Evaluate on test set if available
        if self.test_loader is not None:
            test_metrics = _metrics_over_loader(self.test_loader, desc="Eval Test")
            if test_metrics is None:
                print("Test loader is empty — skipping test metrics.")
            else:
                test_rmse, test_r2 = test_metrics
                print(f"Performance on Test set (streamed) -> RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
                self.y_test = None  # Could collect full arrays if needed
                self.y_test_pred = None
        
        return history
    
    def scale_x(self):
        """
        Scale the train/test features using the provided scaler.
        - Fits scaler on training data only.
        - Applies transform to both train and test.
        - Converts numpy arrays to PyTorch tensors for model training.
        """
        if self.scaler_x is None:
            raise ValueError("No scaler provided. Pass one when initializing the class.")

        # Fit only on train, then transform both
        self.x = self.scaler_x.fit_transform(self.x)
        


    def scale_y(self):
        """
        Scale the train/test targets using the provided scaler.
        - Fits scaler on training data only.
        - Applies transform to both train and test.
        - Converts numpy arrays to PyTorch tensors for model training.
        """
        if self.scaler_y is None:
            raise ValueError("No scaler provided. Pass one when initializing the class.")

        # Fit only on train, then transform both
        # convert self.y series to nd array if its a series
        if hasattr(self.y, "values"):
            self.y = self.scaler_y.fit_transform(self.y.values.reshape(-1, 1))
        else:
            self.y = self.scaler_y.fit_transform(self.y.reshape(-1, 1))

    def summary(self):
        print(self.__model.summary())
        
    # banary classification
    def predict(self, x_to_pred):
        return self.__model.predict(x_to_pred)
    
    def forcast_next_step(self, window):
        current_batch = window.reshape((1, window.shape[0], 1))
        # One timestep ahead of historical 12 points
        return self.predict(current_batch)[0]

    def predict_proba(self, x_to_pred):
        return self.__model.predict_proba(x_to_pred)

    def accuracy(self):
        return accuracy_score(self.__y_test, self.__y_pred)

    def precision(self, binary_classification=False):
        if binary_classification:
            return precision_score(self.__y_test, self.__y_pred)
        return precision_score(self.__y_test, self.__y_pred, average=None)
    
    def recall(self):
        return recall_score(self.__y_test, self.__y_pred)

    def f1_score(self):
        return f1_score(self.__y_test, self.__y_pred)

    
    def regression_report(
        self,
        y_test=None,
        y_predicted=None,
        savefig: bool = True,
        figure_name: str = "regression_heat_scatter.png",
        write_files: bool = True,
        show_progress: bool = True,
        heatmap: bool = True,              # NEW: if True use density (heat) scatter instead of plain scatter
        heat_method: str = "hex",          # 'hex' | 'hist2d' | 'kde'
        bins: int = 200,                   # resolution for hist2d / hex
        cmap: str = "viridis",             # colormap
        point_size: int = 6,               # used only if heatmap=False
        **kwargs
    ):
        """
        Extended regression metrics report with optional heat map scatter.
        heatmap=True -> density visualization ('hex','hist2d','kde').
        """
        import numpy as np, matplotlib.pyplot as plt
        from math import sqrt
        from sklearn.metrics import (
            r2_score, mean_squared_error, mean_absolute_error,
            median_absolute_error, mean_squared_log_error
        )
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=4, disable=not show_progress, desc="regression_report", leave=False)
        except Exception:
            pbar = None

        # ---------- Resolve y_true / y_pred ----------
        if pbar: pbar.set_postfix(step="resolve")
        y_true = y_pred = None
        if y_test is not None and y_predicted is not None:
            y_true = np.asarray(y_test).reshape(-1)
            y_pred = np.asarray(y_predicted).reshape(-1)
        if (y_true is None or y_pred is None) and hasattr(self, "y_test") and hasattr(self, "y_test_pred"):
            if getattr(self, "y_test", None) is not None and getattr(self, "y_test_pred", None) is not None:
                y_true = np.asarray(self.y_test).reshape(-1)
                y_pred = np.asarray(self.y_test_pred).reshape(-1)
        if (y_true is None or y_pred is None) and hasattr(self, "_Model__y_test") and hasattr(self, "_Model__y_pred"):
            if self._Model__y_test is not None and self._Model__y_pred is not None:
                y_true = np.asarray(self._Model__y_test).reshape(-1)
                y_pred = np.asarray(self._Model__y_pred).reshape(-1)
        if y_true is None or y_pred is None or y_true.size == 0 or y_pred.size == 0:
            raise RuntimeError("regression_report: unresolved y_true / y_pred.")
        if pbar: pbar.update(1)

        # ---------- Plot (Heat map scatter) ----------
        if pbar: pbar.set_postfix(step="plot")
        y_min = float(np.nanmin([y_true.min(), y_pred.min()]))
        y_max = float(np.nanmax([y_true.max(), y_pred.max()]))
        if (not np.isfinite(y_min)) or (not np.isfinite(y_max)) or y_min == y_max:
            y_min, y_max = 0.0, 1.0

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        if heatmap:
            if heat_method == "hex":
                hb = ax.hexbin(
                    y_true, y_pred,
                    gridsize=bins,
                    cmap=cmap,
                    mincnt=1,
                    linewidths=0
                )
                cb = fig.colorbar(hb, ax=ax, shrink=0.85)
                cb.set_label("Count")
            elif heat_method == "hist2d":
                h = ax.hist2d(
                    y_true, y_pred,
                    bins=bins,
                    cmap=cmap
                )
                cb = fig.colorbar(h[3], ax=ax, shrink=0.85)
                cb.set_label("Count")
            elif heat_method == "kde":
                try:
                    import seaborn as sns
                    sns.kdeplot(
                        x=y_true,
                        y=y_pred,
                        fill=True,
                        cmap=cmap,
                        thresh=0,
                        levels=100,
                        ax=ax
                    )
                except Exception:
                    # fallback to hex
                    hb = ax.hexbin(y_true, y_pred, gridsize=bins, cmap=cmap, mincnt=1)
                    cb = fig.colorbar(hb, ax=ax, shrink=0.85)
                    cb.set_label("Count")
            # Identity line
            ax.plot([y_min, y_max], [y_min, y_max], color='red', lw=1.0)
        else:
            ax.scatter(y_true, y_pred, s=point_size, c="black")
            ax.plot([y_min, y_max], [y_min, y_max], color='red', lw=1.0, label="Identity")
            ax.legend(loc="upper left")

        ax.set_xlim(y_min, y_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title("Observed vs Predicted (Density)" if heatmap else "Observed vs Predicted")
        fig.tight_layout()
        if savefig:
            try:
                fig.savefig(f"{self.output_metrics_dir}/{figure_name}", dpi=250, bbox_inches="tight")
                print(f"\n Regression scatter plot saved to {self.output_metrics_dir}/{figure_name}")
            except Exception:
                pass
        plt.close(fig)
        if pbar: pbar.update(1)

        # ---------- Metrics ----------
        if pbar: pbar.set_postfix(step="metrics")
        def _nrmse_mean(y_t, y_p):
            m = np.mean(y_t)
            return sqrt(mean_squared_error(y_t, y_p)) / m if m != 0 else np.nan
        def _nrmse_range(y_t, y_p):
            rng = np.max(y_t) - np.min(y_t)
            return sqrt(mean_squared_error(y_t, y_p)) / rng if rng > 0 else np.nan
        def _mape(y_t, y_p):
            mask = y_t != 0
            return 100.0 * np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) if np.any(mask) else np.nan
        def _smape(y_t, y_p):
            denom = np.abs(y_t) + np.abs(y_p)
            mask = denom != 0
            return 100.0 * np.mean(2.0 * np.abs(y_p[mask] - y_t[mask]) / denom[mask]) if np.any(mask) else np.nan
        def _mpe(y_t, y_p):
            mask = y_t != 0
            return 100.0 * np.mean((y_p[mask] - y_t[mask]) / y_t[mask]) if np.any(mask) else np.nan
        def _bias(y_t, y_p): return float(np.mean(y_p - y_t))
        def _pbias(y_t, y_p):
            denom = np.sum(y_t)
            return 100.0 * np.sum(y_p - y_t) / denom if denom != 0 else np.nan
        def _nse(y_t, y_p):
            denom = np.sum((y_t - np.mean(y_t)) ** 2)
            return 1.0 - np.sum((y_p - y_t)**2) / denom if denom > 0 else np.nan
        def _kge(y_t, y_p):
            sy, sp = np.std(y_t), np.std(y_p)
            if sy == 0 or sp == 0: return np.nan
            r = np.corrcoef(y_t, y_p)[0, 1]
            alpha = sp / sy
            mu_y, mu_p = np.mean(y_t), np.mean(y_p)
            beta = (mu_p / mu_y) if mu_y != 0 else np.nan
            if np.isnan(r) or np.isnan(beta): return np.nan
            return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        def _cvrmse_percent(y_t, y_p):
            mean_y = np.mean(y_t)
            return 100.0 * sqrt(mean_squared_error(y_t, y_p)) / mean_y if mean_y != 0 else np.nan
        def _rae(y_t, y_p):
            denom = np.sum(np.abs(y_t - np.mean(y_t)))
            return np.sum(np.abs(y_t - y_p)) / denom if denom != 0 else np.nan
        def _rse(y_t, y_p):
            denom = np.sum((y_t - np.mean(y_t)) ** 2)
            return np.sum((y_t - y_p) ** 2) / denom if denom != 0 else np.nan

        mse_val = mean_squared_error(y_true, y_pred)
        rmse_val = sqrt(mse_val)
        use_msle = np.all(y_true > 0) and np.all(y_pred > 0)

        metrics = {
            'R2':           float(r2_score(y_true, y_pred)),
            'R':            float(np.corrcoef(y_true, y_pred)[0, 1]) if (np.std(y_true) > 0 and np.std(y_pred) > 0) else np.nan,
            'MSE':          float(mse_val),
            'RMSE':         float(rmse_val),
            'NRMSE_mean':   float(_nrmse_mean(y_true, y_pred)),
            'NRMSE_range':  float(_nrmse_range(y_true, y_pred)),
            'MAE':          float(mean_absolute_error(y_true, y_pred)),
            'MEDAE':        float(median_absolute_error(y_true, y_pred)),
            'MAPE':         float(_mape(y_true, y_pred)),
            'SMAPE':        float(_smape(y_true, y_pred)),
            'MPE':          float(_mpe(y_true, y_pred)),
            'Bias':         float(_bias(y_true, y_pred)),
            'PBIAS':        float(_pbias(y_true, y_pred)),
            'CV_RMSE_%':    float(_cvrmse_percent(y_true, y_pred)),
            'RAE':          float(_rae(y_true, y_pred)),
            'RSE':          float(_rse(y_true, y_pred)),
            'NSE':          float(_nse(y_true, y_pred)),
            'KGE':          float(_kge(y_true, y_pred)),
        }
        if use_msle:
            metrics['MSLE'] = float(mean_squared_log_error(y_true, y_pred))
        if pbar: pbar.update(1)

        # ---------- Write files ----------
        if pbar: pbar.set_postfix(step="write")
        if write_files:
            try:
                out1 = f"{self.output_metrics_dir}/regression_metrics.txt"
                out2 = f"{self.output_metrics_dir}/cross_validation_results.txt"
                for pth in (out1, out2):
                    with open(pth, "w", encoding="utf-8") as f:
                        for k, v in metrics.items():
                            f.write(f"{k}: {v}\n")
                print(f"Metrics written:\n  {out1}\n  {out2}")
            except Exception as e:
                print(f"Write failed: {e}")
        if pbar:
            pbar.update(1)
            pbar.close()
        return metrics
    
    
    def regression_report_v1(self, y_test=None, y_predicted=None, savefig=True, figure_name="regression_scatter_plot.png"):
        """
        Works for both in-memory and streaming (chunked) cases.
        - If y_test/y_predicted (array-like) are provided, they take precedence.
        - Else if in streaming mode with a test_loader, it will iterate the loader to compute y_true/y_pred.
        - Else it falls back to stored attributes from prior evaluation.

        Returns a dict of metrics. Optionally shows/saves a scatter identity plot.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from math import sqrt
        from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_squared_log_error, mean_absolute_error

        # Optional: seaborn styling if available
        try:
            import seaborn as sns
            seaborn_ok = True
        except Exception:
            seaborn_ok = False

        # ---------- 1) Resolve y_true / y_pred for all modes ----------
        y_true, y_pred = None, None

        # A) If user passed arrays explicitly
        if y_test is not None and y_predicted is not None:
            y_true = np.asarray(y_test).reshape(-1)
            y_pred = np.asarray(y_predicted).reshape(-1)

        # B) If streaming (parquet) mode and test_loader exists → compute fresh
        if y_true is None and getattr(self, "_parquet_mode", False) and getattr(self, "test_loader", None) is not None:
            try:
                if self.model_name == 'tabformer':
                    print('Evaluating PyTorch model on test loader')
                    import torch
                    try:
                        from tqdm.auto import tqdm
                        use_tqdm = True
                    except Exception:
                        use_tqdm = False
                    self.__model.eval()
                    preds, trues = [], []
                    with torch.no_grad():
                        iterator = tqdm(self.test_loader, desc="Eval Test", leave=False) if use_tqdm else self.test_loader
                        for Xb, yb in iterator:
                            Xb = Xb.to(self.device, non_blocking=True)
                            yb = yb.to(self.device, non_blocking=True)
                            pb = self.__model(Xb)
                            preds.append(pb.detach())   # keep on GPU; concat once
                            trues.append(yb.detach())
                            
                elif self.model_name == 'catboost':
                    print('Evaluating CatBoost model on test loader')
                    try:
                        from tqdm.auto import tqdm
                        use_tqdm = True
                    except Exception:
                        use_tqdm = False
                    preds_list, trues_list = [], []
                    iterator = tqdm(self.test_loader, desc="Eval Test (CatBoost)", leave=False) if use_tqdm else self.test_loader
                    for Xb, yb in iterator:
                        # Convert tensors to numpy if needed
                        Xn = Xb.numpy() if hasattr(Xb, "numpy") else np.asarray(Xb)
                        yn = yb.numpy().reshape(-1) if hasattr(yb, "numpy") else np.asarray(yb).reshape(-1)
                        # Predict (use Pool if available)
                        try:
                            from catboost import Pool
                            pred = self.__model.predict(Pool(Xn))
                        except Exception:
                            pred = self.__model.predict(Xn)
                        preds_list.append(np.asarray(pred).reshape(-1))
                        trues_list.append(yn)
                    if preds_list:
                        y_pred = np.concatenate(preds_list, axis=0)
                        y_true = np.concatenate(trues_list, axis=0)
                        self.y_test = y_true
                        self.y_test_pred = y_pred
                
                else:
                    # Classic ML model path
                    print('Evaluating classic ML model on test loader')
                    preds, trues = [], []
                    for Xb, yb in self.test_loader:
                        # Convert tensors to numpy if needed
                        if hasattr(Xb, 'numpy'):
                            Xb = Xb.numpy()
                        if hasattr(yb, 'numpy'):
                            yb = yb.numpy()
                        
                        # Reshape y if needed
                        if len(yb.shape) > 1:
                            yb = yb.reshape(-1)
                            
                        # Use standard predict method for ML models
                        pb = self.__model.predict(Xb)
                        
                        preds.append(pb)
                        trues.append(yb)
                
                # Common code for both paths
                if len(preds) > 0:
                    y_pred = torch.cat(preds, 0).view(-1).cpu().numpy()
                    y_true = torch.cat(trues, 0).view(-1).cpu().numpy()
                    self.y_test = y_true
                    self.y_test_pred = y_pred
            except Exception as e:
                print(f"Error during test evaluation: {e}")
                # if anything goes wrong, fall back to stored attrs below
                pass

        # C) Fall back to stored attributes depending on model branch
        if y_true is None or y_pred is None:
            if getattr(self, "_Modelmodel_name", None) == 'tabformer' or getattr(self, "model_name", None) == 'tabformer':
                if hasattr(self, "y_test") and hasattr(self, "y_test_pred") and self.y_test is not None and self.y_test_pred is not None:
                    y_true = np.asarray(self.y_test).reshape(-1)
                    y_pred = np.asarray(self.y_test_pred).reshape(-1)
            elif getattr(self, "_Modelmodel_name", None) == 'catboost' or getattr(self, "model_name", None) == 'catboost':
                if hasattr(self, "y_test") and hasattr(self, "y_test_pred") and self.y_test is not None and self.y_test_pred is not None:
                    y_true = np.asarray(self.y_test).reshape(-1)
                    y_pred = np.asarray(self.y_test_pred).reshape(-1)
            else:
                if hasattr(self, "_Model__y_test") and hasattr(self, "_Model__y_pred") and self._Model__y_test is not None and self._Model__y_pred is not None:
                    y_true = np.asarray(self._Model__y_test).reshape(-1)
                    y_pred = np.asarray(self._Model__y_pred).reshape(-1)

        # Final guard
        print("y_true and y_pred values:")
        print(y_true)
        print(y_pred)
        
        if y_true is None or y_pred is None or y_true.size == 0 or y_pred.size == 0:
            raise RuntimeError("regression_report: could not resolve non-empty y_true / y_pred. "
                            "Provide y_test & y_predicted, or ensure test/eval ran and stored attributes are set, "
                            "or supply a non-empty test_loader in streaming mode.")

        # ---------- 2) Scatter (identity) plot ----------
        # Compute plotting bounds robustly
        y_min = float(np.nanmin([y_true.min(), y_pred.min()]))
        y_max = float(np.nanmax([y_true.max(), y_pred.max()]))
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
            y_min, y_max = 0.0, 1.0  # fallback

        if seaborn_ok:
            import seaborn as sns
            sns.set_theme(color_codes=True)
        fig, ax = plt.subplots(figsize=(7, 7))
        if seaborn_ok:
            import pandas as pd
            df_plot = pd.DataFrame({"y_test": y_true, "y_predicted": y_pred})
            sns.scatterplot(data=df_plot, x="y_test", y="y_predicted", s=7, color='black', edgecolor='black', ax=ax)
        else:
            ax.scatter(y_true, y_pred, s=7, c='black', edgecolors='black')

        ax.plot([y_min, y_max], [y_min, y_max], color='red', label='Identity line')
        ax.set_xlabel('Real values')
        ax.set_ylabel('Estimated values')
        ax.set_xlim([y_min, y_max])
        ax.set_ylim([y_min, y_max])
        ax.legend()
        plt.tight_layout()
        if savefig:
            plt.savefig(f"{self.output_metrics_dir}/{figure_name}", dpi=300)
            print(f"Scatter plot saved to {self.output_metrics_dir}/{figure_name}")
        else:
            plt.show()

        # ---------- 3) Metrics ----------
        # MSLE requires strictly positive values
        use_msle = np.all(y_true > 0) and np.all(y_pred > 0)

        def _nse(y_t, y_p):
            denom = np.sum((y_t - np.mean(y_t))**2)
            if denom <= 0:
                return np.nan
            return 1.0 - np.sum((y_p - y_t)**2) / denom

        def _mape(y_t, y_p):
            mask = y_t != 0
            if not np.any(mask):
                return np.nan
            return 100.0 * np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask]))

        def _nrmse_mean(y_t, y_p):
            rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
            m = np.mean(y_t)
            return rmse / m if m != 0 else np.nan

        def _nrmse_range(y_t, y_p):
            rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
            rng = np.max(y_t) - np.min(y_t)
            return rmse / rng if rng > 0 else np.nan

        def _bias(y_t, y_p):
            return float(np.mean(y_p - y_t))

        def _pbias(y_t, y_p):
            denom = np.sum(y_t)
            if denom == 0:
                return np.nan
            return 100.0 * np.sum(y_p - y_t) / denom

        def _kge(y_t, y_p):
            # Kling-Gupta Efficiency
            sy, sp = np.std(y_t), np.std(y_p)
            if sy == 0 or sp == 0:
                return np.nan
            r = np.corrcoef(y_t, y_p)[0, 1]
            alpha = sp / sy
            mu_y, mu_p = np.mean(y_t), np.mean(y_p)
            beta = (mu_p / mu_y) if mu_y != 0 else np.nan
            if np.isnan(r) or np.isnan(beta):
                return np.nan
            return 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)

        metrics = {
            'R2':   float(r2_score(y_true, y_pred)),
            'R':    float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan,
            'MSE':  float(mean_squared_error(y_true, y_pred)),
            'RMSE': float(sqrt(mean_squared_error(y_true, y_pred))),
            'NRMSE_mean': float(_nrmse_mean(y_true, y_pred)),
            'NRMSE_range': float(_nrmse_range(y_true, y_pred)),
            'MAE':  float(mean_absolute_error(y_true, y_pred)),
            'MEDAE': float(median_absolute_error(y_true, y_pred)),
            'MAPE': float(_mape(y_true, y_pred)),
            'Bias': float(_bias(y_true, y_pred)),
            'PBIAS': float(_pbias(y_true, y_pred)),
            'NSE': float(_nse(y_true, y_pred)),
            'KGE': float(_kge(y_true, y_pred)),
        }
        if use_msle:
            metrics['MSLE'] = float(mean_squared_log_error(y_true, y_pred))
            
        # write metrics to file
        with open(f"{self.output_metrics_dir}/regression_metrics.txt", "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        print(f"Regression metrics written to {self.output_metrics_dir}/regression_metrics.txt")

        return metrics

        
    def classification_report(self, y_test=None, y_predicted=None, **kwargs): 
        """
        pass y_test and y_predected as pandas serie is get_column
        """
        
        from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_squared_log_error, mean_absolute_error, classification_report
        if y_test is not None and y_predicted is not None:
            self.__y_test = y_test
        
        classification_report_results = classification_report(self.__y_test, self.__y_pred, **kwargs)
        print(classification_report_results)
        return classification_report_results
        
    def roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.__y_test, self.__y_pred)
        roc_auc = auc(fpr, tpr)
        print("Air sous la courbe" + str(roc_auc)) 
        plt.figure()
        plt.plot(fpr, tpr, color='orange', lw=2, label='ROC curve(area under curve = % 0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='darkgrey', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate(1 - Specificity)')
        plt.ylabel('True Positive Rate(Sensitivity)')
        plt.title('ROC Curve')
        plt.legend(loc='upper left')
        plt.show() 

    def boost_model(self):
        ada_boost = AdaBoostClassifier(n_estimators=100, base_estimator=self.__model, learning_rate=0.1, random_state=0)
        self.__boosted_model = ada_boost
        self.__boosted_model.fit(self.__x_train, self.__y_train)

    def predict_with_boosted_model(self, x_to_pred):
        return self.__boosted_model.predict(x_to_pred)

    
    def save_model(self, file_path="data/geoai_model"):
        """
        Save the trained model instance for future use.

        The method supports:
        • PyTorch models (for 'tabformer'): saves the state_dict as a .pth file.
        • CatBoost models: uses the built-in save_model() method to save as a .cbm file.
        • TabPFN models (for 'tabpfn'): uses TabPFN's save API to .tabpfn_fit.
        • Other models: falls back to joblib.dump to save as a .pkl file.

        The file extension is automatically appended based on the model type.
        """
        if self.__model is None:
            print("No model available to save.")
            return

        name = (self.model_name or "").lower()
        if name == "tabformer":
            required_ext = ".pth"
        elif name == "catboost":
            required_ext = ".cbm"
        elif name == "tabpfn":
            required_ext = ".tabpfn_fit"
        else:
            required_ext = ".pkl"

        # Append or adjust the file extension if necessary
        base, ext = os.path.splitext(file_path)
        if ext.lower() != required_ext:
            file_path = base + required_ext

        if name == "catboost":
            try:
                self.__model.save_model(file_path)
                print(f"CatBoost model saved using its built-in method to: {file_path}")
            except Exception as e:
                print(f"Error saving CatBoost model: {e}")

        elif name == "tabformer":
            try:
                import torch
            except ImportError:
                print("torch module not found. Unable to save PyTorch model.")
                return
            try:
                torch.save(self.__model.state_dict(), file_path)
                print(f"PyTorch model state_dict saved to: {file_path}")
            except Exception as e:
                print(f"Error saving PyTorch model: {e}")

        elif name == "tabpfn":
            # Prefer TabPFN native fitted-model saver
            try:
                from tabpfn.model_loading import save_fitted_tabpfn_model
            except Exception as e:
                print(f"TabPFN save API not available ({e}); falling back to joblib .pkl")
                try:
                    import joblib
                except ImportError:
                    print("joblib module not found. Unable to save model.")
                    return
                alt_path = base + ".pkl"
                try:
                    joblib.dump(self.__model, alt_path)
                    print(f"Model saved via joblib.dump to: {alt_path}")
                except Exception as e2:
                    print(f"Error saving model with joblib: {e2}")
                return

            try:
                save_fitted_tabpfn_model(self.__model, file_path)
                print(f"TabPFN fitted model saved to: {file_path}")
            except Exception as e:
                print(f"Error saving TabPFN model: {e}")

        else:
            try:
                import joblib
            except ImportError:
                print("joblib module not found. Unable to save model.")
                return
            try:
                joblib.dump(self.__model, file_path)
                print(f"Model saved via joblib.dump to: {file_path}")
            except Exception as e:
                print(f"Error saving model with joblib: {e}")
    
    
    def save_model_v0(self, file_path="data/geoai_model"):
        """
        Save the trained model instance for future use.

        The method supports:
          • PyTorch models (for 'tabformer'): saves the state_dict as a .pth file.
          • CatBoost models: uses the built-in save_model() method to save as a .cbm file.
          • Other models: falls back to joblib.dump to save as a .pkl file.

        The file extension is automatically appended based on the model type.
        
        Args:
            file_path (str): Base file path for saving the model (extension will be appended automatically).
        """
        #print(self.__model)
        # Ensure there is a model to save.
        
        if self.__model is None:
            print("No model available to save.")
            return
        
        # Determine the required file extension
        if self.model_name.lower() == "tabformer":
            required_ext = ".pth"
        elif self.model_name.lower() == "catboost":
            required_ext = ".cbm"
        else:
            required_ext = ".pkl"

        # Append or adjust the file extension if necessary
        if not file_path.endswith(required_ext):
            file_path = os.path.splitext(file_path)[0] + required_ext

        # Save based on the model type
        if self.model_name.lower() == "catboost":
            try:
                self.__model.save_model(file_path)
                print(f"CatBoost model saved using its built-in method to: {file_path}")
            except Exception as e:
                print(f"Error saving CatBoost model: {e}")
        elif self.model_name.lower() == "tabformer":
            try:
                import torch
            except ImportError:
                print("torch module not found. Unable to save PyTorch model.")
                return
            try:
                torch.save(self.__model.state_dict(), file_path)
                print(f"PyTorch model state_dict saved to: {file_path}")
            except Exception as e:
                print(f"Error saving PyTorch model: {e}")
        else:
            try:
                import joblib
            except ImportError:
                print("joblib module not found. Unable to save model.")
                return
            try:
                joblib.dump(self.__model, file_path)
                print(f"Model saved via joblib.dump to: {file_path}")
            except Exception as e:
                print(f"Error saving model with joblib: {e}") 

    def load_model(self, file_path):
        """
        Load the trained model from file.

        The method supports:
          • PyTorch models (for 'tabformer'): expects a .pth file containing the state_dict.
          • CatBoost models: expects a .cbm file and uses load_model().
          • Other models: assumes a .pkl file loaded via joblib.load.

        Args:
            file_path (str): The path to the saved model file.
        """
        
        if not os.path.exists(file_path):
            print("Model file does not exist.")
            return

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pth":
            try:
                import torch
            except ImportError:
                print("torch module not found. Unable to load PyTorch model.")
                return
            try:
                # Load the state_dict (using CPU as fallback; adjust map_location if needed)
                state_dict = torch.load(file_path, map_location='cpu')
            except Exception as e:
                print(f"Error loading PyTorch model: {e}")
                return

            if hasattr(self.__model, "load_state_dict") and callable(self.__model.load_state_dict):
                self.__model.load_state_dict(state_dict)
                print(f"PyTorch model loaded from: {file_path}")
            else:
                print("Current model instance does not support loading a state_dict.")

        elif ext == ".cbm":
            # Ensure the model type is CatBoost
            if self.model_name.lower() == "catboost":
                try:
                    # Optionally reinitialize the model if needed
                    try:
                        self.__model = self.__model.__class__()
                    except Exception:
                        pass
                    self.__model.load_model(file_path)
                    print(f"CatBoost model loaded from: {file_path}")
                except Exception as e:
                    print(f"Error loading CatBoost model: {e}")
            else:
                print("Model type mismatch: expected a CatBoost model.")

        elif ext == ".pkl":
            try:
                import joblib
            except ImportError:
                print("joblib module not found. Unable to load model.")
                return
            try:
                self.__model = joblib.load(file_path)
                print(f"Model loaded via joblib from: {file_path}")
            except Exception as e:
                print(f"Error loading model with joblib: {e}")
                
        elif ext == ".tabpfn_fit":
            try:
                from tabpfn.model_loading import load_fitted_tabpfn_model
            except Exception as e:
                print(f"TabPFN load API not available: {e}")
                return
            # Choose device: use CPU by default; prefer CUDA if available and requested
            device = "cpu"
            try:
                import torch
                if torch.cuda.is_available() and getattr(self, "use_gpu", False):
                    device = "cuda"
            except Exception:
                pass
            try:
                self.__model = load_fitted_tabpfn_model(file_path, device=device)
                self.model_name = "tabpfn"
                print(f"TabPFN fitted model loaded from: {file_path} (device={device})")
            except Exception as e:
                print(f"Error loading TabPFN fitted model: {e}")

        else:
            print(f"Unsupported file extension: {ext}")

    def report(self):
        if self.model_name  == 'dl':
            if self.__task == 'ts' or self.__task == 'r':
                if self.__validation_percentage == 0:
                    loss = self.history.history['loss']
                    x = range(1, len(loss) + 1)
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.plot(x, loss, 'b', label='Training loss')
                    plt.title('Training and validation loss')
                    plt.legend()
                    plt.show()
                else:
                    loss = self.history.history['loss']
                    val_loss = self.history.history['val_loss']
                    r2 = self.history.history['r2_keras']
                    val_r2 = self.history.history['val_r2_keras']
                    x = range(1, len(loss) + 1)
                    # (1,2) one row and 2 columns
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    fig.suptitle('Training and validation monitoring of MSE and R²')
                    ax1.plot(x, loss, 'b', label='Training MSE')
                    ax1.plot(x, val_loss, 'r', label='Validation MSE')
                    ax1.set_title('MSE monitoring')
                    ax1.legend()
                    ax2.plot(x, r2, 'b', label='Training R²')
                    ax2.plot(x, val_r2, 'r', label='Validation R²')
                    ax2.set_title('R² monitoring')
                    ax2.legend()
                plt.show()
                
            elif self.__task == 'c':
                if self.__validation_percentage == 0:
                    acc = self.history.history['accuracy']
                    loss = self.history.history['loss']
                    x = range(1, len(acc) + 1)
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.title('Accuracy ')
                    plt.plot(x, acc, 'r', label='Accuracy')
                    plt.legend()
                    plt.subplot(1, 2, 2)
                    plt.plot(x, loss, 'b', label='Loss')
                    plt.title('Loss')
                    plt.legend()
                else:
                    acc = self.history.history['accuracy']
                    val_acc = self.history.history['val_accuracy']
                    loss = self.history.history['loss']
                    val_loss = self.history.history['val_loss']
                    x = range(1, len(acc) + 1)
                    plt.figure(figsize=(12, 5))
                    plt.subplot(2, 2, 1)
                    plt.plot(x, acc, 'b', label='Training accuracy')
                    plt.plot(x, val_acc, 'r', label='Validation accuracy')
                    plt.title('Training and validation accuracy')
                    plt.legend()
                    plt.subplot(2, 2, 2)
                    plt.plot(x, loss, 'b', label='Training loss')
                    plt.plot(x, val_loss, 'r', label='Validation loss')
                    plt.title('Training and validation loss')
                    plt.legend()
        else:
            if self.__task == 'r':
                report = self.regression_report()
                print(report) 
                return report
            else:
                report = self.classification_report()
                return report
                
    def cross_validation(
            self,
            k: int = 5,
            metric: str = 'r2',
            parquet_root: str | None = None,   # optional override; if None, use self.parquet_path if you store it
            feature_names: list[str] | None = None,
            split_strategy: str = "month",     # "month" (recommended) or "row" (i.i.d., optimistic)
            batch_size: int = 65536,
            epochs: int | None = None,
            lr: float = 1e-3,
        ):
            """
            Chunking-aware K-fold CV.
            - Streaming (chunked): month-based folds; trains via train_streaming per fold and evaluates on the held-out months.
            - In-memory: uses classic KFold on self.x / self.y.
    
            Returns: pandas.DataFrame with columns: Fold, R2, RMSE and a 'Mean' row.
            """
            import os, random
            from glob import glob
            import numpy as np
            import pandas as pd
            import torch
            from torch.utils.data import DataLoader
            from sklearn.metrics import r2_score, mean_squared_error
            from tqdm.auto import tqdm
    
            # ---------- Helper: dataset & loaders without scaler ----------
            class ParquetMonthBatches(torch.utils.data.IterableDataset):
                def __init__(
                    self,
                    root, feat_names, target_col="y",
                    batch_size=65536, month_keys=None,
                    shuffle_months=True, shuffle_within_month=True,
                    drop_last=False, seed=42
                ):
                    super().__init__()
                    self.root = root.rstrip(os.sep)
                    self.feature_names = feat_names
                    self.target_col = target_col
                    self.batch_size = batch_size
                    self.shuffle_months = shuffle_months
                    self.shuffle_within_month = shuffle_within_month
                    self.drop_last = drop_last
                    self.seed = seed
                    if month_keys is None:
                        month_keys = []
                        for ydir in sorted(glob(f"{self.root}/year=*")):
                            y = int(os.path.basename(ydir).split("=")[1])
                            for mdir in sorted(glob(os.path.join(ydir, "month=*"))):
                                m = int(os.path.basename(mdir).split("=")[1])
                                month_keys.append((y, m))
                    self.month_keys = month_keys
    
                def _load_month_df(self, year: int, month: int, columns=None):
                    import dask.dataframe as dd
                    path = os.path.join(self.root, f"year={int(year)}", f"month={int(month):02d}")
                    ddf = dd.read_parquet(path, engine="pyarrow", columns=columns)
                    return ddf.compute()
    
                def __iter__(self):
                    if self.seed is not None:
                        random.seed(self.seed)
                    keys = self.month_keys[:]
                    if self.shuffle_months:
                        random.shuffle(keys)
                    for (yr, mo) in keys:
                        df = self._load_month_df(yr, mo, columns=self.feature_names + ["y"])
                        if df.empty:
                            continue
                        if self.shuffle_within_month:
                            df = df.sample(frac=1.0, random_state=random.randint(0, 10_000))
                        X = df[self.feature_names].to_numpy(dtype=np.float32, copy=False)
                        y = df["y"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
                        n = X.shape[0]; bs = self.batch_size
                        full = (n // bs) * bs
                        end = full if self.drop_last else n
                        for i in range(0, end, bs):
                            yield torch.from_numpy(X[i:i+bs]), torch.from_numpy(y[i:i+bs])
    
            def _eval_loader_tabformer(model, loader, device):
                """Compute (RMSE, R2) over a loader for TabFormer models; returns (None, None) if empty."""
                preds, trues = [], []
                got_any = False
                model._Model__model.eval()
                with torch.no_grad():
                    for Xb, yb in loader:
                        got_any = True
                        Xb = Xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                        pb = model._Model__model(Xb)
                        preds.append(pb.detach().cpu())
                        trues.append(yb.detach().cpu())
                if not got_any:
                    return None, None
                y_pred = torch.cat(preds, 0).numpy().reshape(-1)
                y_true = torch.cat(trues, 0).numpy().reshape(-1)
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                r2   = float(r2_score(y_true, y_pred))
                return rmse, r2
            
            def _eval_loader_ml(model, loader):
                """Compute (RMSE, R2) over a loader for classic ML models; returns (None, None) if empty."""
                preds, trues = [], []
                got_any = False
                for Xb, yb in loader:
                    got_any = True
                    # Convert tensors to numpy if needed
                    if hasattr(Xb, 'numpy'):
                        Xb = Xb.numpy()
                    if hasattr(yb, 'numpy'):
                        yb = yb.numpy()
                    
                    # Reshape y if needed
                    if len(yb.shape) > 1:
                        yb = yb.reshape(-1)
                    
                    # Make predictions
                    pb = model._Model__model.predict(Xb)
                    
                    preds.append(pb)
                    trues.append(yb)
                    
                if not got_any:
                    return None, None
                    
                y_pred = np.concatenate(preds, axis=0)
                y_true = np.concatenate(trues, axis=0)
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                r2 = float(r2_score(y_true, y_pred))
                return rmse, r2
    
            # ---------- COMMON SETUP FOR STREAMING MODE ----------
            parquet_mode = getattr(self, "_parquet_mode", False) or (parquet_root is not None)
            if parquet_mode:
                root = parquet_root or getattr(self, "parquet_path", None)
                if not root:
                    raise ValueError("cross_validation(chunked): parquet_root not provided and self.parquet_path missing.")
    
                # feature names
                if feature_names is None:
                    feature_names = getattr(self, "feature_names_", None)
                if not feature_names:
                    # infer by peeking one partition
                    import dask.dataframe as dd
                    ddf = dd.read_parquet(root, engine="pyarrow")
                    exclude = {"y", "date", "year", "month"}
                    feature_names = [c for c in ddf.columns if c not in exclude]
    
                # discover non-empty months
                month_keys_all = []
                for ydir in sorted(glob(f"{root.rstrip(os.sep)}/year=*")):
                    y = int(os.path.basename(ydir).split("=")[1])
                    for mdir in sorted(glob(os.path.join(ydir, "month=*"))):
                        m = int(os.path.basename(mdir).split("=")[1])
                        # quick existence check; skip empties
                        try:
                            import dask.dataframe as dd
                            path = os.path.join(root.rstrip(os.sep), f"year={y}", f"month={m:02d}")
                            ddf = dd.read_parquet(path, engine="pyarrow", columns=[feature_names[0], "y"])
                            n = len(ddf)
                            if n > 0:
                                month_keys_all.append((y, m))
                        except Exception:
                            pass
    
                if len(month_keys_all) < k:
                    k = max(1, len(month_keys_all))
                if k < 2:
                    raise ValueError("Not enough non-empty month partitions to perform CV (need at least 2).")
    
                # Make k folds over months
                rnd = random.Random(42)
                keys = month_keys_all[:]
                rnd.shuffle(keys)
                folds = np.array_split(keys, k)
    
                # infer input dim by peeking one batch from a temporary loader
                tmp_ds = ParquetMonthBatches(root, feature_names, month_keys=[keys[0]], batch_size=1024,
                                            shuffle_months=False, shuffle_within_month=False)
                tmp_loader = DataLoader(tmp_ds, batch_size=None)
                Xb0, _ = next(iter(tmp_loader))
                in_dim = Xb0.shape[1]
    
                # ---------- STREAMING PATH FOR TABFORMER ----------
                if self.model_name == 'tabformer':
                    fold_ids, r2_results, rmse_results = [], [], []
                    
                    # loop folds
                    for fold_idx in range(k):
                        val_keys = list(folds[fold_idx])
                        train_keys = [km for i, f in enumerate(folds) if i != fold_idx for km in f]
    
                        # Build loaders
                        train_ds = ParquetMonthBatches(root, feature_names, month_keys=train_keys,
                                                    batch_size=batch_size, shuffle_months=True, shuffle_within_month=True)
                        val_ds   = ParquetMonthBatches(root, feature_names, month_keys=val_keys,
                                                    batch_size=batch_size, shuffle_months=False, shuffle_within_month=False)
    
                        train_loader = DataLoader(train_ds, batch_size=None)
                        val_loader   = DataLoader(val_ds, batch_size=None)
    
                        # Fresh model per fold
                        m = Model(
                            data_x=None, data_y=None,
                            model_name='tabformer',
                            epochs=(epochs or getattr(self, "_Model__epochs", 50)),
                            parquet_batches_data={ "train_loader": train_loader, "val_loader": val_loader },
                            batch_size=getattr(self, "_Model__batch_size", 64),
                        )
                        # Build the TabFormer backbone
                        if not hasattr(m, "_Model__model") or m._Model__model is None:
                            m._Model__model = FTTransformer(num_features=in_dim)
                        device = m.device
    
                        # Train
                        m.train_streaming(
                            epochs=epochs or getattr(self, "_Model__epochs", 50),
                            lr=lr,
                            log_every=50,
                        )
    
                        # Evaluate on held-out months
                        rmse, r2 = _eval_loader_tabformer(m, val_loader, device)
                        if rmse is None or r2 is None:
                            continue
    
                        fold_ids.append(fold_idx + 1)
                        r2_results.append(r2)
                        rmse_results.append(rmse)
    
                    # Summarize
                    df = pd.DataFrame({"Fold": fold_ids, "R2": r2_results, "RMSE": rmse_results})
                    if not df.empty:
                        df.loc["Mean"] = {"Fold": "Mean", "R2": df["R2"].mean(), "RMSE": df["RMSE"].mean()}
                    return df
                
                # ---------- STREAMING PATH FOR CLASSIC ML MODELS ----------
                elif self.model_name in ['catboost', 'xgboost', 'random_forest', 'decision_tree', 'gradient_boosting', 'linear_regression', 'svm', 'adaboost']:
                    fold_ids, r2_results, rmse_results = [], [], []
                    
                    print(f"Running {k}-fold cross-validation for {self.model_name} in streaming mode")
                    
                    # loop folds
                    for fold_idx in tqdm(range(k), desc="CV Folds"):
                        val_keys = list(folds[fold_idx])
                        train_keys = [km for i, f in enumerate(folds) if i != fold_idx for km in f]
    
                        # Build loaders
                        train_ds = ParquetMonthBatches(root, feature_names, month_keys=train_keys,
                                                    batch_size=batch_size, shuffle_months=True, shuffle_within_month=True)
                        val_ds   = ParquetMonthBatches(root, feature_names, month_keys=val_keys,
                                                    batch_size=batch_size, shuffle_months=False, shuffle_within_month=False)
    
                        train_loader = DataLoader(train_ds, batch_size=None)
                        val_loader   = DataLoader(val_ds, batch_size=None)
    
                        # Create a fresh model for this fold
                        m = Model(
                            data_x=None, data_y=None,
                            model_name=self.model_name,
                            task=self.__task,
                            parquet_batches_data={ "train_loader": train_loader, "val_loader": val_loader },
                        )
    
                        # Train using the ML streaming method
                        m.train_streaming_ml(
                            batch_accumulation_size=batch_size,
                            log_every=50,
                        )
    
                        # Evaluate on held-out months
                        rmse, r2 = _eval_loader_ml(m, val_loader)
                        if rmse is None or r2 is None:
                            continue
    
                        fold_ids.append(fold_idx + 1)
                        r2_results.append(r2)
                        rmse_results.append(rmse)
    
                    # Summarize
                    df = pd.DataFrame({"Fold": fold_ids, "R2": r2_results, "RMSE": rmse_results})
                    if not df.empty:
                        df.loc["Mean"] = {"Fold": "Mean", "R2": df["R2"].mean(), "RMSE": df["RMSE"].mean()}
                    return df
    
            # ---------- IN-MEMORY PATH FOR TABFORMER ----------
            if self.model_name == 'tabformer':
                # Keep existing in-memory KFold code for tabformer
                import numpy as np
                import torch
                import torch.nn as nn
                from torch.utils.data import DataLoader, TensorDataset
                from sklearn.model_selection import KFold
                from sklearn.metrics import r2_score, mean_squared_error
                import pandas as pd
                from tqdm.auto import tqdm
    
                device = getattr(self, "device", torch.device("cpu"))
                X_all = self.x
                y_all = self.y
    
                kf = KFold(n_splits=k, shuffle=True, random_state=42)
                fold_ids, r2_results, rmse_results = [], [], []
    
                for fold_num, (train_idx, test_idx) in enumerate(tqdm(kf.split(X_all), total=k, desc="CV Folds"), start=1):
                    X_train, X_test = X_all[train_idx], X_all[test_idx]
                    y_train, y_test = y_all[train_idx], y_all[test_idx]
    
                    X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
                    y_train_t = torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1)
                    X_test_t  = torch.as_tensor(X_test,  dtype=torch.float32)
                    y_test_t  = torch.as_tensor(y_test,  dtype=torch.float32).unsqueeze(1)
    
                    train_ds = TensorDataset(X_train_t, y_train_t)
                    test_ds  = TensorDataset(X_test_t,  y_test_t)
    
                    train_loader = DataLoader(train_ds, batch_size=self.__batch_size, shuffle=True)
                    test_loader  = DataLoader(test_ds,  batch_size=64,               shuffle=False)
    
                    self.__model = FTTransformer(num_features=X_train.shape[1])
                    self.__model.to(device)
    
                    optimizer = torch.optim.Adam(self.__model.parameters(), lr=1e-3)
                    criterion = nn.MSELoss()
    
                    epoch_bar = tqdm(range(epochs), desc=f"Fold {fold_num} - Epochs", leave=False, position=0)
                    for epoch in epoch_bar:
                        self.__model.train()
                        batch_losses = []
                        batch_bar = tqdm(train_loader, desc=f"Train fold {fold_num} {epoch+1}/{self.__epochs}", leave=False, position=1)
                        running = 0.0
                        for i, (bx, by) in enumerate(batch_bar, 1):
                            bx = bx.to(device); by = by.to(device)
                            optimizer.zero_grad()
                            out = self.__model(bx)
                            loss = criterion(out, by)
                            loss.backward()
                            optimizer.step()
                            li = loss.item()
                            batch_losses.append(li); running += li
                            if i % 10 == 0 or i == len(train_loader):
                                batch_bar.set_postfix(avg_batch_loss=running / i)
                        epoch_bar.set_postfix(avg_train_batch_mse=np.mean(batch_losses))
    
                    # Evaluate
                    self.__model.eval()
                    test_preds, test_targets = [], []
                    with torch.no_grad():
                        test_bar = tqdm(test_loader, desc=f"Eval-Test fold {fold_num}", leave=False, position=1)
                        for bx, by in test_bar:
                            bx = bx.to(device); by = by.to(device)
                            pb = self.__model(bx)
                            test_preds.append(pb.detach().cpu()); test_targets.append(by.detach().cpu())
    
                    y_test_pred = torch.cat(test_preds, dim=0).numpy().reshape(-1)
                    y_test_true = torch.cat(test_targets, dim=0).numpy().reshape(-1)
    
                    mse  = mean_squared_error(y_test_true, y_test_pred)
                    rmse = float(np.sqrt(mse))
                    r2   = r2_score(y_test_true, y_test_pred)
    
                    fold_ids.append(fold_num); r2_results.append(r2); rmse_results.append(rmse)
    
                df = pd.DataFrame({"Fold": fold_ids, "R2": r2_results, "RMSE": rmse_results})
                df.loc["Mean"] = {"Fold": "Mean", "R2": df["R2"].mean(), "RMSE": df["RMSE"].mean()}
                return df
    
            # ---------- IN-MEMORY PATH FOR CLASSIC ML MODELS ----------
            else:
                # Use different metrics based on task type (classification vs regression)
                from sklearn.model_selection import cross_val_score
                import numpy as np
                result = DataFrame()
                
                # Add fold numbers
                result.add_column('Folds', [i for i in range(1, k+1)])
                
                if self.__task == 'c':  # Classification task
                    # Stratified K-Fold with shuffling for classification
                    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

                    accuracy_results = cross_val_score(self.__model, self.x, self.y, cv=skf, scoring='accuracy')
                    result.add_column('Accuracy', accuracy_results)
                    
                    unique_classes = np.unique(self.y)
                    if len(unique_classes) == 2:  # Binary classification
                        precision_results = cross_val_score(self.__model, self.x, self.y, cv=skf, scoring='precision')
                        recall_results = cross_val_score(self.__model, self.x, self.y, cv=skf, scoring='recall')
                        f1_results = cross_val_score(self.__model, self.x, self.y, cv=skf, scoring='f1')
                        roc_auc_results = cross_val_score(self.__model, self.x, self.y, cv=skf, scoring='roc_auc')
                        
                        result.add_column('Precision', precision_results)
                        result.add_column('Recall', recall_results)
                        result.add_column('F1', f1_results)
                        result.add_column('ROC_AUC', roc_auc_results)
                        
                    else:  # Multiclass classification
                        precision_results = cross_val_score(self.__model, self.x, self.y, cv=skf, scoring='precision_weighted')
                        recall_results = cross_val_score(self.__model, self.x, self.y, cv=skf, scoring='recall_weighted')
                        f1_results = cross_val_score(self.__model, self.x, self.y, cv=skf, scoring='f1_weighted')
                        
                        result.add_column('Precision', precision_results)
                        result.add_column('Recall', recall_results)
                        result.add_column('F1', f1_results)
                
                else:  # Regression task (keep original code)
                    # K-Fold with shuffling for regression
                    kf = KFold(n_splits=k, shuffle=True, random_state=42)

                    r2_results = cross_val_score(self.__model, self.x, self.y, cv=kf, scoring='r2')
                    try:
                        rmse_neg = cross_val_score(self.__model, self.x, self.y, cv=kf, scoring='neg_root_mean_squared_error')
                        rmse_results = -rmse_neg
                    except ValueError:
                        mse_neg = cross_val_score(self.__model, self.x, self.y, cv=kf, scoring='neg_mean_squared_error')
                        rmse_results = np.sqrt(-mse_neg)
                    result.add_column('R2', r2_results)
                    result.add_column('RMSE', rmse_results)

                # Add mean row
                cols = result.get_columns_names()
                numeric_cols = [c for c in cols[1:] if result.get_column(c).dtype != 'object']
                mean_row = {c: None for c in cols}
                mean_row['Folds'] = 'Mean'
                for c in numeric_cols:
                    mean_row[c] = float(result.get_column(c).mean())
                result.add_row(mean_row)
                
                result.show()

    def best_model(self):
        pass
        
    
    # Coerce pandas to numpy float32 for PyTorch
    def _to_float32_ndarray(self, X):
        if isinstance(X, pd.DataFrame):
            # remember feature names once
            try:
                self.feature_names_ = list(X.columns)
            except Exception:
                pass
            try:
                return X.to_numpy(dtype=np.float32, copy=False)
            except Exception:
                # fallback: keep only numeric cols if mixed types
                return X.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32, copy=False)
        return np.asarray(X, dtype=np.float32)
    
    def _to_float32_1d(self, y):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = np.asarray(y)
        return np.asarray(y, dtype=np.float32).reshape(-1)
    
    
    def features_importance_catboost(
        self,
        *,
        target_col: str = "y",
        exclude_cols: list[str] | None = None,
        importance_type: str = "PredictionValuesChange",  # ['PredictionValuesChange','LossFunctionChange','ShapValues','InternalFeatureImportance']
        sample_rows: int = 200_000,                       # sample from parquet to build a Pool for importance
        top_k: int = 30,
        save_csv: bool = True,
        savefig: bool = True,
        figure_name: str = "catboost_feature_importance.png",
        show: bool = False,
    ):
        """
        Compute CatBoost feature importance for huge parquet datasets by sampling rows across row-groups.

        Requirements:
          - self.model_name == 'catboost' and a trained CatBoost model in self.__model.
          - self.single_parquet_path points to a single parquet file (huge file approach).

        Args:
          importance_type: CatBoost fstr type. Defaults to 'PredictionValuesChange'. 'ShapValues' is heavier.
          sample_rows: Max number of rows to sample from parquet for importance computation.
          top_k: Number of top features to plot.
        """
        import os
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
        import pyarrow as pa
        from catboost import Pool
        from tqdm.auto import tqdm

        if self.model_name.lower() != "catboost":
            raise ValueError("features_importance_catboost is available only for CatBoost models.")
        if self.__model is None:
            raise ValueError("No CatBoost model found. Train the model before computing feature importance.")
        if not getattr(self, "single_parquet_path", None):
            raise ValueError("single_parquet_path is not set. This method targets the huge parquet file approach.")

        # Discover features from parquet schema (reuse cached feature_names_ if present)
        pf = pq.ParquetFile(self.single_parquet_path, memory_map=True)
        schema = pf.schema_arrow
        columns = list(schema.names)
        if target_col not in columns:
            raise ValueError(f"Target column '{target_col}' not found in parquet.")

        drop_meta = {"date", "year", "month"}
        feature_cols = [c for c in columns if c not in {target_col, *drop_meta}]
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        if not feature_cols:
            raise ValueError("No feature columns found after exclusions.")

        # Cache names for later use
        self.feature_names_ = feature_cols

        # Sampling across row-groups
        total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
        target_sample = min(int(sample_rows), int(total_rows))
        if target_sample <= 0:
            raise ValueError("sample_rows must be > 0 and less than total rows.")

        # Decide per-row-group budget
        rg_sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
        # proportional allocation
        per_rg_take = [max(0, int(np.floor(target_sample * (n / max(1, total_rows))))) for n in rg_sizes]
        # ensure at least some rows if rounding down to zeros
        shortfall = target_sample - sum(per_rg_take)
        # distribute the remainder greedily
        if shortfall > 0:
            order = np.argsort([-n for n in rg_sizes])  # bigger RG first
            for idx in order:
                if shortfall <= 0:
                    break
                per_rg_take[idx] += 1
                shortfall -= 1

        X_parts, y_parts = [], []
        for rg_idx in tqdm(range(pf.num_row_groups), desc="Sampling for importance", leave=False):
            take_n = per_rg_take[rg_idx]
            if take_n <= 0:
                continue
            tbl = pf.read_row_group(rg_idx, columns=feature_cols + [target_col])
            n = tbl.num_rows
            if n == 0:
                continue
            # random sample indices without replacement
            if take_n >= n:
                sub_tbl = tbl
            else:
                rng = np.random.default_rng(42 + rg_idx)
                idx = rng.choice(n, size=take_n, replace=False)
                sub_tbl = tbl.take(pa.array(idx, type=pa.int64()))
            # Arrow -> NumPy
            try:
                X_cols = [sub_tbl[c].to_numpy(zero_copy_only=False) for c in feature_cols]
                X = np.column_stack(X_cols).astype(np.float32, copy=False)
                y = sub_tbl[target_col].to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1)
            except Exception:
                pdf = sub_tbl.to_pandas()
                X = pdf[feature_cols].to_numpy(dtype=np.float32, copy=False)
                y = pdf[target_col].to_numpy(dtype=np.float32, copy=False).reshape(-1)
            # keep only finite rows
            finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
            if not np.any(finite):
                continue
            X_parts.append(X[finite])
            y_parts.append(y[finite])

        if not X_parts:
            raise RuntimeError("No rows collected for importance computation (all filtered or empty).")

        X_s = np.vstack(X_parts)
        y_s = np.concatenate(y_parts)
        # Final guard
        if X_s.shape[0] == 0:
            raise RuntimeError("Empty sample after filtering; cannot compute importance.")

        # Build Pool for importance
        pool = Pool(X_s, y_s, feature_names=feature_cols)

        # Compute importance
        importance_type = str(importance_type)
        importances = self.__model.get_feature_importance(pool, type=importance_type)
        importances = np.asarray(importances).reshape(-1)

        if importances.size != len(feature_cols):
            # For SHAP values, CatBoost returns [n_objects, n_features+1] (last is expected_value)
            if importance_type.lower() == "shapvalues":
                # take mean absolute SHAP across objects, ignore bias term
                shap_vals = importances.reshape(-1, len(feature_cols) + 1)
                importances = np.mean(np.abs(shap_vals[:, :-1]), axis=0)
            else:
                raise RuntimeError(f"Unexpected importance shape for type={importance_type}: {importances.shape}")

        df_imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
        df_imp.sort_values("importance", ascending=False, inplace=True)
        df_imp.reset_index(drop=True, inplace=True)

        # Save CSV
        if save_csv:
            out_dir = self.output_metrics_dir
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "catboost_feature_importance.csv")
            df_imp.to_csv(csv_path, index=False)
            print(f"✅ Saved feature importance to: {csv_path}")

        # Plot
        if savefig or show:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                sns.set_theme(style="whitegrid")
                top = df_imp.head(int(top_k))
                plt.figure(figsize=(max(8, min(18, 0.4 * len(top))), 5))
                ax = sns.barplot(data=top, x="feature", y="importance", palette="Spectral")
                ax.set_xlabel("Features")
                ax.set_ylabel("Importance")
                ax.set_title(f"CatBoost Feature Importance ({importance_type})")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                if savefig:
                    out_dir = getattr(self, "output_metrics_dir", "./results_monitor")
                    os.makedirs(out_dir, exist_ok=True)
                    fig_path = os.path.join(out_dir, figure_name)
                    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
                    print(f"✅ Saved figure to: {fig_path}")
                if show:
                    plt.show()
                plt.close()
            except Exception as e:
                print(f"Plotting skipped: {e}")

        return df_imp
    
    
    def pdp_ice_xai(
        self,
        *,
        features: list[str] | None = None,   # feature names to explain; None -> all features (or first top_k)
        top_k: int = 20,                     # if features is None, limit to first top_k features
        target_col: str = "y",
        exclude_cols: list[str] | None = None,
        sample_rows: int = 5000,             # sample rows for PDP computation
        grid_points: int = 20,               # number of grid points per feature
        ice_count: int = 50,                 # how many instances to plot ICE for
        output_dir: str | None = None,
        show: bool = False,
        class_index: int | None = None,      # for classification: which class prob to use
        random_state: int = 42,
        chunk_size_pred: int = 10000,        # chunk size for prediction to limit RAM
    ) -> dict:
        """
        Compute PDP and ICE for selected features and export artifacts suitable for XAI studies.

        Outputs (under output_dir/pdp_ice):
          - pdp_ice_<feature>.csv     (columns: value, pdp, ice_0..ice_{k-1})
          - pdp_ice_<feature>.png     (PDP line + ICE curves)
        Returns a dict mapping features -> {'csv': path, 'png': path}
        """
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(random_state)
        out_dir = output_dir or getattr(self, "output_metrics_dir", "./results_monitor")
        pdp_dir = os.path.join(out_dir, "pdp_ice")
        os.makedirs(pdp_dir, exist_ok=True)

        # ---------- Resolve data sample (X_s, feature_names) ----------
        def _feature_names_from_array(X, fallback_prefix="f"):
            if hasattr(self, "feature_names_") and self.feature_names_:
                return list(self.feature_names_)
            if hasattr(X, "columns"):
                try:
                    return list(X.columns)
                except Exception:
                    pass
            return [f"{fallback_prefix}{i:03d}" for i in range(X.shape[1])]

        X_s = None
        feature_names = None

        if getattr(self, "x", None) is not None and self.single_parquet_path is None:
            X_all = self.x
            n = getattr(X_all, "shape", (len(X_all),))[0]
            take = min(max(1, int(sample_rows)), n)
            idx = rng.choice(n, size=take, replace=False)
            if hasattr(X_all, "iloc"):
                X_s_df = X_all.iloc[idx]
                feature_names = _feature_names_from_array(X_all)
                X_s = X_s_df.to_numpy(dtype=np.float32, copy=False)
            else:
                X_s = np.asarray(X_all, dtype=np.float32)[idx]
                feature_names = _feature_names_from_array(X_all)
        elif getattr(self, "single_parquet_path", None):
            # Sample from single parquet file
            import pyarrow.parquet as pq
            import pyarrow as pa

            pf = pq.ParquetFile(self.single_parquet_path, memory_map=True)
            schema = pf.schema_arrow
            cols = list(schema.names)
            if target_col not in cols:
                raise ValueError(f"Target column '{target_col}' not found in parquet file.")
            drop_meta = {"date", "year", "month"}
            feature_cols = [c for c in cols if c not in {target_col, *drop_meta}]
            if exclude_cols:
                feature_cols = [c for c in feature_cols if c not in exclude_cols]
            if not feature_cols:
                raise RuntimeError("No feature columns after exclusions.")

            total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
            take = min(max(1, int(sample_rows)), int(total_rows))

            # proportional per-row-group sampling
            rg_sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
            per_rg_take = [max(0, int(np.floor(take * (n / max(1, total_rows))))) for n in rg_sizes]
            shortfall = take - sum(per_rg_take)
            if shortfall > 0:
                order = np.argsort([-n for n in rg_sizes])
                for idx_rg in order:
                    if shortfall <= 0:
                        break
                    per_rg_take[idx_rg] += 1
                    shortfall -= 1

            X_parts = []
            for rg in range(pf.num_row_groups):
                n = rg_sizes[rg]
                if n <= 0 or per_rg_take[rg] <= 0:
                    continue
                tbl = pf.read_row_group(rg, columns=feature_cols)
                k = min(per_rg_take[rg], n)
                idx_local = rng.choice(n, size=k, replace=False)
                sub_tbl = tbl.take(pa.array(idx_local, type=pa.int64()))
                try:
                    X_cols = [sub_tbl[c].to_numpy(zero_copy_only=False) for c in feature_cols]
                    X_np = np.column_stack(X_cols).astype(np.float32, copy=False)
                except Exception:
                    pdf = sub_tbl.to_pandas()
                    X_np = pdf[feature_cols].to_numpy(dtype=np.float32, copy=False)
                finite = np.isfinite(X_np).all(axis=1)
                if np.any(finite):
                    X_parts.append(X_np[finite])
            if not X_parts:
                raise RuntimeError("No finite rows collected for PDP/ICE sampling.")
            X_s = np.vstack(X_parts)
            feature_names = list(feature_cols)
            self.feature_names_ = feature_names
        else:
            raise RuntimeError("No data available. Provide in-memory X/y or single_parquet_path.")

        # ---------- Choose features to explain ----------
        feat_used = list(feature_names)
        if features:
            missing = [f for f in features if f not in feat_used]
            if missing:
                print(f"⚠️ Missing features ignored: {missing}")
            feat_used = [f for f in features if f in feat_used]
        else:
            # limit to first top_k to keep runtime bounded
            feat_used = feat_used[: int(top_k)]

        # ---------- Predictor wrapper ----------
        def _predict_np(Xnp: np.ndarray) -> np.ndarray:
            # chunk to control memory
            preds = []
            if (self.model_name or "").lower() == "tabformer":
                import torch
                self.__model.eval()
                device = getattr(self, "device", torch.device("cpu"))
                with torch.no_grad():
                    for start in range(0, Xnp.shape[0], chunk_size_pred):
                        end = min(start + chunk_size_pred, Xnp.shape[0])
                        xb = torch.as_tensor(Xnp[start:end], dtype=torch.float32, device=device)
                        pb = self.__model(xb).detach().cpu().numpy().reshape(-1)
                        preds.append(pb)
            else:
                # sklearn/catboost/xgboost path
                # handle classification prob if requested
                has_proba = hasattr(self.__model, "predict_proba")
                for start in range(0, Xnp.shape[0], chunk_size_pred):
                    end = min(start + chunk_size_pred, Xnp.shape[0])
                    xb = Xnp[start:end]
                    if self.__task == 'c' and has_proba and class_index is not None:
                        proba = self.__model.predict_proba(xb)
                        if proba.ndim == 2 and class_index < proba.shape[1]:
                            preds.append(proba[:, class_index])
                        else:
                            preds.append(self.__model.predict(xb).reshape(-1))
                    else:
                        preds.append(np.asarray(self.__model.predict(xb)).reshape(-1))
            return np.concatenate(preds, axis=0)

        # ---------- Compute PDP/ICE per feature ----------
        artifacts = {}
        n_samples = X_s.shape[0]
        ice_n = min(max(1, int(ice_count)), n_samples)
        ice_rows_idx = rng.choice(n_samples, size=ice_n, replace=False)
        X_ice_base = X_s[ice_rows_idx].copy()

        for fname in feat_used:
            fi = feature_names.index(fname)
            col_vals = X_s[:, fi]
            # robust grid from percentiles (avoid extreme outliers)
            lo = np.nanpercentile(col_vals, 1.0)
            hi = np.nanpercentile(col_vals, 99.0)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = float(np.nanmin(col_vals)), float(np.nanmax(col_vals))
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    # skip non-informative feature
                    continue
            grid = np.linspace(lo, hi, int(grid_points), dtype=np.float32)

            # PDP: mean prediction over X_s with feature set to grid value
            pdp_vals = []
            # ICE: for each selected instance, vary feature along grid
            ice_matrix = np.empty((grid.shape[0], ice_n), dtype=np.float32)

            # Prepare a working buffer to avoid repeated allocs (for PDP)
            X_work = X_s.copy()
            for gi, gv in enumerate(grid):
                # PDP over all rows
                X_work[:, fi] = gv
                y_pred = _predict_np(X_work)
                pdp_vals.append(float(np.mean(y_pred)))

                # ICE over selected rows
                X_ice = X_ice_base.copy()
                X_ice[:, fi] = gv
                ice_pred = _predict_np(X_ice)
                ice_matrix[gi, :] = ice_pred.astype(np.float32, copy=False)

            pdp_vals = np.asarray(pdp_vals, dtype=np.float32)

            # ---------- Save CSV ----------
            df_csv = pd.DataFrame({"value": grid, "pdp": pdp_vals})
            # append ICE columns
            for j in range(ice_n):
                df_csv[f"ice_{j}"] = ice_matrix[:, j]
            csv_path = os.path.join(pdp_dir, f"pdp_ice_{fname}.csv")
            df_csv.to_csv(csv_path, index=False)

            # ---------- Plot ----------
            try:
                fig, ax = plt.subplots(figsize=(7, 5))
                # ICE curves (thin, transparent)
                for j in range(min(ice_n, 100)):  # cap plotted ICE to 100 for readability
                    ax.plot(grid, ice_matrix[:, j], color="#999999", alpha=0.2, linewidth=1)
                # PDP line (bold)
                ax.plot(grid, pdp_vals, color="#d62728", linewidth=2.5, label="PDP")
                ax.set_title(f"PDP + ICE: {fname}")
                ax.set_xlabel(fname)
                ax.set_ylabel("Prediction")
                ax.legend(loc="best")
                fig.tight_layout()
                png_path = os.path.join(pdp_dir, f"pdp_ice_{fname}.png")
                fig.savefig(png_path, dpi=300, bbox_inches="tight")
                if show:
                    plt.show()
                plt.close(fig)
            except Exception:
                png_path = None

            artifacts[fname] = {"csv": csv_path, "png": png_path}

        print(f"PDP/ICE artifacts saved to: {pdp_dir}")
        return artifacts
    
    
    def shap_xai(
        self,
        *,
        target_col: str = "y",
        exclude_cols: list[str] | None = None,
        sample_rows: int = 5000,          # rows to explain
        background_size: int = 200,       # background size for SHAP
        top_k: int = 20,                  # number of top features for plots
        output_dir: str | None = None,
        save_raw: bool = True,            # save SHAP matrix as .npy
        show: bool = False,               # show figures
        random_state: int = 42,
    ) -> dict:
        """
        Compute SHAP explanations and export artifacts for XAI studies.

        Supports:
          - CatBoost (native ShapValues or SHAP TreeExplainer)
          - XGBoost / sklearn tree models (SHAP TreeExplainer)
          - TabFormer (PyTorch) via SHAP GradientExplainer

        Outputs (saved under output_dir/shap):
          - mean_abs_shap.csv            (mean |SHAP| per feature)
          - shap_summary_bar.png         (bar chart of top_k features)
          - shap_summary_dot.png         (summary dot plot)
          - shap_dependence_<feature>.png (per-feature dependence plots, top_k)
          - shap_values.npy              (optional; raw SHAP values)
          - shap_force_first.html        (force plot for first sample, best-effort)

        Returns:
          dict with generated file paths.
        """
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # Ensure output dir
        out_dir = output_dir or getattr(self, "output_metrics_dir", "./results_monitor")
        shap_dir = os.path.join(out_dir, "shap")
        os.makedirs(shap_dir, exist_ok=True)

        # Import SHAP
        try:
            import shap
        except Exception as e:
            raise RuntimeError("SHAP is not installed. Install with: pip install shap") from e

        rng = np.random.default_rng(random_state)

        # -------- Resolve features (X_s, y_s, feature_names) --------
        def _feature_names_from_array(X, fallback_prefix="f"):
            if hasattr(self, "feature_names_") and self.feature_names_:
                return list(self.feature_names_)
            # if pandas
            if hasattr(X, "columns"):
                try:
                    return list(X.columns)
                except Exception:
                    pass
            # fallback
            return [f"{fallback_prefix}{i:03d}" for i in range(X.shape[1])]

        X_s = None
        y_s = None
        feature_names = None

        if getattr(self, "x", None) is not None and getattr(self, "y", None) is not None and self.single_parquet_path is None:
            # In-memory path (robust for pandas and numpy)
            X_all = self.x
            y_all = self.y
            n = getattr(X_all, "shape", (len(X_all),))[0]
            take = min(max(1, int(sample_rows)), n)
            idx = rng.choice(n, size=take, replace=False)

            # Row sampling that works for pandas and numpy
            if hasattr(X_all, "iloc"):
                X_s = X_all.iloc[idx]
            else:
                X_s = X_all[idx]

            if y_all is not None:
                if hasattr(y_all, "iloc"):
                    y_s = y_all.iloc[idx]
                else:
                    y_s = y_all[idx]

            feature_names = _feature_names_from_array(X_all)
        elif getattr(self, "single_parquet_path", None):
            # Sample from single parquet file
            import pyarrow.parquet as pq
            import pyarrow as pa

            pf = pq.ParquetFile(self.single_parquet_path, memory_map=True)
            schema = pf.schema_arrow
            cols = list(schema.names)
            if target_col not in cols:
                raise ValueError(f"Target column '{target_col}' not found in parquet file.")
            drop_meta = {"date", "year", "month"}
            feature_cols = [c for c in cols if c not in {target_col, *drop_meta}]
            if exclude_cols:
                feature_cols = [c for c in feature_cols if c not in exclude_cols]
            if not feature_cols:
                raise RuntimeError("No feature columns after exclusions.")

            total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
            take = min(max(1, int(sample_rows)), int(total_rows))

            # proportional sampling per row-group
            rg_sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
            per_rg_take = [max(0, int(np.floor(take * (n / max(1, total_rows))))) for n in rg_sizes]
            shortfall = take - sum(per_rg_take)
            if shortfall > 0:
                order = np.argsort([-n for n in rg_sizes])
                for idx_rg in order:
                    if shortfall <= 0:
                        break
                    per_rg_take[idx_rg] += 1
                    shortfall -= 1

            X_parts, y_parts = [], []
            for rg in range(pf.num_row_groups):
                n = rg_sizes[rg]
                if n <= 0 or per_rg_take[rg] <= 0:
                    continue
                tbl = pf.read_row_group(rg, columns=feature_cols + [target_col])
                k = min(per_rg_take[rg], n)
                # sample without replacement
                idx_local = rng.choice(n, size=k, replace=False)
                sub_tbl = tbl.take(pa.array(idx_local, type=pa.int64()))
                # Arrow -> NumPy (fast path)
                try:
                    X_cols = [sub_tbl[c].to_numpy(zero_copy_only=False) for c in feature_cols]
                    X_np = np.column_stack(X_cols).astype(np.float32, copy=False)
                    y_np = sub_tbl[target_col].to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1)
                except Exception:
                    pdf = sub_tbl.to_pandas()
                    X_np = pdf[feature_cols].to_numpy(dtype=np.float32, copy=False)
                    y_np = pdf[target_col].to_numpy(dtype=np.float32, copy=False).reshape(-1)
                good = np.isfinite(X_np).all(axis=1) & np.isfinite(y_np)
                if np.any(good):
                    X_parts.append(X_np[good])
                    y_parts.append(y_np[good])

            if not X_parts:
                raise RuntimeError("No finite rows collected for SHAP sampling.")
            X_s = np.vstack(X_parts)
            y_s = np.concatenate(y_parts)
            feature_names = list(feature_cols)
            # cache names
            self.feature_names_ = feature_names
        else:
            raise RuntimeError("No data available. Provide in-memory X/y or single_parquet_path.")

        # Ensure float32
        if hasattr(X_s, "to_numpy"):
            X_s = X_s.to_numpy(dtype=np.float32, copy=False)
        else:
            X_s = np.asarray(X_s, dtype=np.float32)
        if y_s is not None and hasattr(y_s, "to_numpy"):
            y_s = y_s.to_numpy(dtype=np.float32, copy=False)
        elif y_s is not None:
            y_s = np.asarray(y_s, dtype=np.float32)

        # Background subset for SHAP
        bg_n = min(max(1, int(background_size)), X_s.shape[0])
        bg_idx = rng.choice(X_s.shape[0], size=bg_n, replace=False)
        X_bg = X_s[bg_idx]

        # -------- Build explainer and compute SHAP --------
        model_type = (self.model_name or "").lower()
        shap_values = None
        expected_value = None
        multiclass = False

        try:
            if model_type == "catboost":
                # Try native CatBoost ShapValues (fast, robust)
                try:
                    from catboost import Pool, CatBoostRegressor, CatBoostClassifier
                    if isinstance(self.__model, CatBoostRegressor):
                        sv = self.__model.get_feature_importance(Pool(X_s, y_s), type="ShapValues")
                        shap_values = sv[:, :-1]
                        expected_value = sv[:, -1].mean() if sv.shape[1] == shap_values.shape[1] + 1 else None
                    elif isinstance(self.__model, CatBoostClassifier):
                        # For binary classification CatBoost returns shap per class if requested via SHAP lib.
                        # Use SHAP TreeExplainer for consistent behavior.
                        explainer = shap.TreeExplainer(self.__model)
                        sv = explainer.shap_values(X_s)
                        # sv could be list[class] or array; handle both
                        if isinstance(sv, list):
                            multiclass = True
                            # Aggregate mean(|SHAP|) across classes later; for plots pick class 1 if binary
                            if len(sv) == 2:
                                shap_values = np.asarray(sv[1])
                                expected_value = np.asarray(explainer.expected_value)[1] if hasattr(explainer, "expected_value") else None
                            else:
                                # aggregate across classes for global importance
                                shap_values = np.mean(np.abs(np.stack(sv, axis=0)), axis=0) * np.sign(np.stack(sv, axis=0)).mean(axis=0)
                                expected_value = np.mean(explainer.expected_value) if hasattr(explainer, "expected_value") else None
                        else:
                            shap_values = np.asarray(sv)
                            expected_value = getattr(explainer, "expected_value", None)
                    else:
                        # Unknown catboost type -> fallback to TreeExplainer
                        explainer = shap.TreeExplainer(self.__model)
                        sv = explainer.shap_values(X_s)
                        shap_values = np.asarray(sv if not isinstance(sv, list) else sv[-1])
                        expected_value = getattr(explainer, "expected_value", None)
                except Exception:
                    # Fallback to TreeExplainer
                    explainer = shap.TreeExplainer(self.__model)
                    sv = explainer.shap_values(X_s)
                    shap_values = np.asarray(sv if not isinstance(sv, list) else sv[-1])
                    expected_value = getattr(explainer, "expected_value", None)

            elif model_type in ("xgboost", "random_forest", "decision_tree", "gradient_boosting"):
                explainer = shap.TreeExplainer(self.__model)
                sv = explainer.shap_values(X_s)
                # list -> multiclass
                if isinstance(sv, list):
                    multiclass = True
                    # pick last class for plots; aggregate for global
                    shap_values = np.asarray(sv[-1])
                    expected_value = np.asarray(explainer.expected_value)[-1] if hasattr(explainer, "expected_value") else None
                else:
                    shap_values = np.asarray(sv)
                    expected_value = getattr(explainer, "expected_value", None)

            elif model_type == "tabformer":
                # SHAP GradientExplainer for PyTorch
                import torch
                self.__model.eval()
                device = getattr(self, "device", torch.device("cpu"))
                model = self.__model

                # Background and samples as tensors on device
                X_bg_t = torch.as_tensor(X_bg, dtype=torch.float32, device=device)
                X_s_t = torch.as_tensor(X_s, dtype=torch.float32, device=device)

                # SHAP GradientExplainer expects (model, inputs) or model callable
                # For 1-output regression, this works
                explainer = shap.GradientExplainer(model, X_bg_t)
                sv = explainer.shap_values(X_s_t)
                # GradientExplainer returns np.array
                shap_values = np.asarray(sv)
                # expected value: compute with model over background
                with torch.no_grad():
                    f_bg = model(X_bg_t).detach().cpu().numpy().reshape(-1)
                expected_value = float(np.mean(f_bg))

            else:
                # Generic fallback using KernelExplainer (slower)
                # Use a simple prediction function
                def f_pred(x):
                    try:
                        return self.__model.predict(x).reshape(-1)
                    except Exception:
                        return np.asarray(self.__model.predict(x)).reshape(-1)
                explainer = shap.KernelExplainer(f_pred, shap.sample(X_bg, min(100, X_bg.shape[0])))
                sv = explainer.shap_values(X_s, nsamples="auto")
                shap_values = np.asarray(sv)
                expected_value = getattr(explainer, "expected_value", None)

        except Exception as e:
            raise RuntimeError(f"Failed to compute SHAP values: {e}")

        if shap_values is None or shap_values.size == 0:
            raise RuntimeError("Empty SHAP values.")

        # Coerce to 2D [n_samples, n_features]
        if isinstance(shap_values, list):
            shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            # handle [classes, samples, features] -> choose last class
            shap_values = shap_values[-1]
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(-1, 1)

        # -------- Global importance (mean |SHAP|) and CSV --------
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        order = np.argsort(-mean_abs)
        feature_names = list(feature_names)
        mean_abs_df = pd.DataFrame({
            "feature": [feature_names[i] for i in order],
            "mean_abs_shap": mean_abs[order]
        })
        mean_abs_csv = os.path.join(shap_dir, "mean_abs_shap.csv")
        mean_abs_df.to_csv(mean_abs_csv, index=False)

        # Optionally save raw shap matrix
        shap_values_npy = None
        if save_raw:
            shap_values_npy = os.path.join(shap_dir, "shap_values.npy")
            np.save(shap_values_npy, shap_values)

        # -------- Plots --------
        artifacts = {
            "mean_abs_csv": mean_abs_csv,
            "shap_values_npy": shap_values_npy,
            "summary_bar_png": None,
            "summary_dot_png": None,
            "dependence_pngs": [],
            "force_html": None,
        }

        # 1) Summary bar plot (top_k)
        try:
            top = min(top_k, len(feature_names))
            fig = plt.figure()
            shap.summary_plot(
                shap_values,
                features=X_s,
                feature_names=feature_names,
                plot_type="bar",
                max_display=top,
                show=show
            )
            bar_path = os.path.join(shap_dir, "shap_summary_bar.png")
            plt.gcf().savefig(bar_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            artifacts["summary_bar_png"] = bar_path
        except Exception:
            plt.close("all")

        # 2) Summary dot plot
        try:
            fig = plt.figure()
            shap.summary_plot(
                shap_values,
                features=X_s,
                feature_names=feature_names,
                max_display=min(top_k, len(feature_names)),
                show=show
            )
            dot_path = os.path.join(shap_dir, "shap_summary_dot.png")
            plt.gcf().savefig(dot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            artifacts["summary_dot_png"] = dot_path
        except Exception:
            plt.close("all")

        # 3) Dependence plots for top_k features
        try:
            for i in order[: min(top_k, len(order))]:
                fname = feature_names[i]
                fig = plt.figure()
                shap.dependence_plot(
                    ind=i,
                    shap_values=shap_values,
                    features=X_s,
                    feature_names=feature_names,
                    interaction_index="auto",
                    show=show
                )
                dep_path = os.path.join(shap_dir, f"shap_dependence_{fname}.png")
                plt.gcf().savefig(dep_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                artifacts["dependence_pngs"].append(dep_path)
        except Exception:
            plt.close("all")

        # 4) Force plot for first sample (HTML), best-effort
        try:
            # Try new SHAP API first (Explanation object)
            explanation = None
            try:
                explanation = shap.Explanation(
                    values=shap_values,
                    base_values=expected_value if expected_value is not None else 0.0,
                    data=X_s,
                    feature_names=feature_names
                )
                # new API
                fhtml = os.path.join(shap_dir, "shap_force_first.html")
                # Use first sample
                fp = shap.plots.force(explanation[0], show=False)
                shap.save_html(fhtml, fp)
                artifacts["force_html"] = fhtml
            except Exception:
                # old API
                fhtml = os.path.join(shap_dir, "shap_force_first.html")
                fp = shap.force_plot(
                    expected_value if expected_value is not None else 0.0,
                    shap_values[0, :],
                    X_s[0, :],
                    feature_names=feature_names,
                )
                shap.save_html(fhtml, fp)
                artifacts["force_html"] = fhtml
        except Exception:
            pass

        print(f"SHAP artifacts saved to: {shap_dir}")
        return artifacts
    
    
    def features_importance(self, 
                            features_nbr=10,
                            method='cb',
                            savefig=False,
                            figure_name='output.png',
                            rotation_angle_x_labels=0,
                            x_label='Features',
                            y_label='Importance scores (%)',
                            **kwargs
                            ):
        """ show a bar chart of features importance and return a dataframe of the results

        Args:
            features_nbr (int, optional): _description_. Defaults to 10.
            savefig (bool, optional): _description_. Defaults to False.
            figure_name (str, optional): _description_. Defaults to 'output.png'.

        Returns:
            _type_: _description_
        """    
        
        if method == 'cb':
            from catboost import CatBoostRegressor, CatBoostClassifier
            
            if self.__task == 'r':
                
                model = CatBoostRegressor(**kwargs)

                model.fit(self.x, self.y)
                
                importances = model.get_feature_importance()
                feature_names = self.x.columns

                # Create a DataFrame for better visualization
                feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

                # Sort the DataFrame by importance
                feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

                print(feature_importance_df)


                feature_imp = pd.Series(importances, index=feature_names)
                
                data = DataFrame()
                data.add_column('Importance', feature_imp)
                data.sort(by_column_name_list=['Importance'], ascending=False)
                # old version
                #feature_imp.nlargest(10).plot(kind='barh')
                
                sns.set_theme(style="whitegrid")
                # Initialize the matplotlib figure
                f, ax = plt.subplots()
 
                # Plot the total crashes
                #sns.set_color_codes("pastel")
                sns.barplot(x=feature_imp.nlargest(features_nbr).index, y=feature_imp.nlargest(features_nbr),
                            palette="Spectral")

                # Add a legend and informative axis label
                plt.xlabel(x_label, fontsize=15)
                plt.ylabel(y_label, fontsize=15)
                
                if rotation_angle_x_labels != 0:
                    plt.xticks(fontsize=11, rotation=rotation_angle_x_labels, ha='right')
                else:
                    plt.xticks(fontsize=11)
                    
                plt.tight_layout()  # Adjust layout to make room for rotated labels 
                plt.yticks(fontsize=11)
                
                if savefig is True:
                    plt.savefig(figure_name, dpi=300, bbox_inches='tight')
                    plt.close(f)
                else:
                    plt.show()
            else:
                etr_model = ExtraTreesClassifier()
                etr_model.fit(self.x,self.y)
                feature_imp = pd.Series(etr_model.feature_importances_,index=[i for i in range(self.x.shape[1])])
                feature_imp.nlargest(10).plot(kind='barh')
                data = DataFrame()
                data.add_column('Importance', feature_imp)
                data.sort(by_column_name_list=['Importance'], ascending=False)
                # old version
                #feature_imp.nlargest(10).plot(kind='barh')
                
                sns.set_theme(style="whitegrid")
                # Initialize the matplotlib figure
                f, ax = plt.subplots()

                # Plot the total crashes
                #sns.set_color_codes("pastel")
                sns.barplot(x=feature_imp.nlargest(features_nbr).index, y=feature_imp.nlargest(features_nbr),
                            palette="Spectral")

                # Add a legend and informative axis label
                plt.xlabel('Features', fontsize=15)
                plt.ylabel('Importance score', fontsize=15)
                plt.xticks(fontsize=11)
                plt.yticks(fontsize=11)
                if savefig is True:
                    plt.savefig(figure_name, dpi=300, bbox_inches='tight')
                    plt.close(f)
                else:
                    plt.show()
                
                """model = self.__model # or XGBRegressor
                plot_importance(model, importance_type = 'gain') # other options available
                plt.show()
                # if you need a dictionary 
                model.get_booster().get_score(importance_type = 'gain')"""
            return data.get_dataframe() 
        
        
        elif method == 'dt':
            
            from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
            
            if self.__task == 'r':
                
                etr_model = ExtraTreesRegressor()
                etr_model.fit(self.x,self.y)
                feature_imp = pd.Series(etr_model.feature_importances_,index=self.x.columns)
                
                data = DataFrame()
                data.add_column('Importance', feature_imp)
                data.sort(by_column_name_list=['Importance'], ascending=False)
                
                
                data.show()

                sns.set_theme(style="whitegrid")
                # Initialize the matplotlib figure
                f, ax = plt.subplots()
 
                # Plot the total crashes
                #sns.set_color_codes("pastel")
                sns.barplot(x=feature_imp.nlargest(features_nbr).index, y=feature_imp.nlargest(features_nbr),
                            palette="Spectral")

                # Add a legend and informative axis label
                plt.xlabel(x_label, fontsize=15)
                plt.ylabel(y_label, fontsize=15)
                
                if rotation_angle_x_labels != 0:
                    plt.xticks(fontsize=11, rotation=rotation_angle_x_labels, ha='right')
                else:
                    plt.xticks(fontsize=11)
                    
                plt.tight_layout()  # Adjust layout to make room for rotated labels 
                plt.yticks(fontsize=11)
                
                if savefig is True:
                    plt.savefig(figure_name, dpi=300, bbox_inches='tight')
                    plt.close(f)
                else:
                    plt.show()
            else:
                etr_model = ExtraTreesClassifier()
                etr_model.fit(self.x,self.y)
                feature_imp = pd.Series(etr_model.feature_importances_,index=[i for i in range(self.x.shape[1])])
                feature_imp.nlargest(10).plot(kind='barh')
                data = DataFrame()
                data.add_column('Importance', feature_imp)
                data.sort(by_column_name_list=['Importance'], ascending=False)
                # old version
                #feature_imp.nlargest(10).plot(kind='barh')
                
                sns.set_theme(style="whitegrid")
                # Initialize the matplotlib figure
                f, ax = plt.subplots()

                # Plot the total crashes
                #sns.set_color_codes("pastel")
                sns.barplot(x=feature_imp.nlargest(features_nbr).index, y=feature_imp.nlargest(features_nbr),
                            palette="Spectral")

                # Add a legend and informative axis label
                plt.xlabel('Features', fontsize=15)
                plt.ylabel('Importance score', fontsize=15)
                plt.xticks(fontsize=11)
                plt.yticks(fontsize=11)
                if savefig is True:
                    plt.savefig(figure_name, dpi=300, bbox_inches='tight')
                    plt.close(f)
                else:
                    plt.show()
                
                """model = self.__model # or XGBRegressor
                plot_importance(model, importance_type = 'gain') # other options available
                plt.show()
                # if you need a dictionary 
                model.get_booster().get_score(importance_type = 'gain')"""
            return data.get_dataframe()    
        
        
        elif method == 'rf':
            pass
        
    def dt_text_representation(self):
        return tree.export_text(self.__model)
    
    def plot_dt_representation(self, viz_type='graph_viz'):
        import graphviz
        
        if viz_type == 'graph_viz':
            
            # DOT data
            dot_data = tree.export_graphviz(self.__model, out_file=None, 
                                            feature_names=self.x.columns.values,  
                                            class_names=self.y.name,
                                            filled=True)

            # Draw graph
            graph = graphviz.Source(dot_data, format="png") 
            return graph
            
        elif viz_type == 'matplotlib':
            fig = plt.figure(figsize=(25,20))
            _ = tree.plot_tree(self.__model, 
                            feature_names=self.x.columns.values,  
                            class_names=self.y.name,
                            filled=True)
            fig.savefig("decistion_tree.png")
            plt.show()
            
    def viz_reporter(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.__y_test, linewidth=3, label='ground truth')
        plt.plot(self.__y_pred, linewidth=3, label='predicted')
        plt.legend(loc='best')
        plt.xlabel('X')
        plt.ylabel('target value')
        
    def fine_tune(self, dict_params=None, n_trials=10):
        import optuna
        
        def objective(trial):
            if self.model_name  == "svm":
                kernels = ['linear', 'poly', 'rbf', 'sigmoid']
                svm_c = trial.suggest_float("svm_c", 1e-10, 1e10, log=True)
                svm_kernel = trial.suggest_categorical("svm_kernel", kernels)
                self.__model = svm.SVR(C=svm_c, 
                                                kernel=svm_kernel, 
                                                gamma="auto")
            elif self.model_name  == "xb":
                boosters = ['gbtree', 'gblinear', 'dart']
                xb_eta = trial.suggest_float("xb_eta", 0.1, 1, step=0.01)
                xb_max_depth = trial.suggest_int("xb_max_depth", 2, 32, log=True)
                xb_booster = trial.suggest_categorical("xb_booster", boosters)
                self.__model = XGBRegressor(
                    eta=xb_eta,
                    max_depth=xb_max_depth,
                    booster=xb_booster,
                )
            elif self.model_name  == "rf":
                rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
                rf_nbr_trees = trial.suggest_int("rf_nbr_trees", 100, 1000, log=True)
                self.__model = RandomForestRegressor(
                    max_depth=rf_max_depth,
                    n_estimators=rf_nbr_trees,
                )
            return self.cross_validation(3)['RMSE'].mean()
        
        if dict_params is None:
            # minimize or maximize
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)
            print(study.best_trial)
        data = DataFrame(study.trials_dataframe(), data_type='df')
        data.drop_columns(['number', 'datetime_start', 'datetime_complete', 'duration', 'state'])
        
        return data.get_dataframe()
        
    def compare_models_cv(self, cv=5):
        """
        Perform cross-validation to compare regression models using R² and RMSE.

        Parameters:
            cv (int): Number of cross-validation folds.

        Returns:
            results (dict): Dictionary of models with mean R² and RMSE scores.
        """
        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.metrics import make_scorer, mean_squared_error, r2_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore")

        if self.__task != 'r':
            raise ValueError("Comparative CV is currently implemented for regression only.")

        models_dict = {
            'LinearRegression': __import__('sklearn.linear_model').linear_model.LinearRegression(),
            'RandomForest': __import__('sklearn.ensemble').ensemble.RandomForestRegressor(),
            'GradientBoosting': __import__('sklearn.ensemble').ensemble.GradientBoostingRegressor(),
            'SVR': __import__('sklearn.svm').svm.SVR(),
            'KNN': __import__('sklearn.neighbors').neighbors.KNeighborsRegressor(),
            'XGBoost': __import__('xgboost').XGBRegressor(),
            'CatBoost': __import__('catboost').CatBoostRegressor(verbose=0),
            'AdaBoost': __import__('sklearn.ensemble').ensemble.AdaBoostRegressor(),
            'DecisionTree': __import__('sklearn.tree').tree.DecisionTreeRegressor()
        }

        results = {}

        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        all_results = []

        for name, model in models_dict.items():
            pipeline = make_pipeline(StandardScaler(), model)

            r2_scores = cross_val_score(pipeline, self.x, self.y, cv=kf, scoring='r2')
            rmse_scorer = make_scorer(lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False), greater_is_better=False)
            rmse_scores = -cross_val_score(pipeline, self.x, self.y, cv=kf, scoring=rmse_scorer)

            for fold_idx, (r2, rmse) in enumerate(zip(r2_scores, rmse_scores), 1):
                all_results.append({
                    'Model': name,
                    'Fold': fold_idx,
                    'R2': round(r2, 4),
                    'RMSE': round(rmse, 4)
                })

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)

        # Pivot Fold-wise R² and RMSE tables
        r2_table = results_df.pivot(index='Fold', columns='Model', values='R2').round(3)
        rmse_table = results_df.pivot(index='Fold', columns='Model', values='RMSE').round(3)

        # Summary table with mean R² and RMSE
        summary_df = results_df.groupby('Model').agg({'R2': 'mean', 'RMSE': 'mean'}).round(4)
        summary_df = summary_df.sort_values(by=['R2', 'RMSE'], ascending=[False, True])

        # Display
        print("📊 R² per Fold:")
        print(r2_table)
        print("\n📉 RMSE per Fold:")
        print(rmse_table)
        print("\n🔍 Summary (Sorted by Best R² and Lowest RMSE):")
        print(summary_df)

        return {
            'fold_r2_table': r2_table,
            'fold_rmse_table': rmse_table,
            'summary': summary_df,
            'raw': results_df
        }

    
    def plot_model_history_from_csv(
        self,
        csv_path: str,
        title: str = "Training history",
        save_path: str | None = None,
        show: bool = True,
        dpi: int = 150,
    ):
        """
        Plot training history from a CSV.

        Default expects columns like:
          - epoch, train_loss, val_loss, val_rmse, val_r2

        If model_name == 'catboost', it will plot iteration-wise training loss (RMSE)
        and validation RMSE/R² from a CSV produced by CatBoost iteration export, e.g.:
          iteration, train_rmse, train_r2, val_rmse, val_r2
        """
        import pandas as pd
        import matplotlib.pyplot as plt

        # Read and normalize columns
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # Helper to coerce numeric if present
        def coerce_numeric(cols: list[str]):
            for col in cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # Try seaborn if available
        try:
            import seaborn as sns
            sns.set_theme(style="whitegrid")
            use_sns = True
        except Exception:
            use_sns = False

        # Pick axis variable (iteration for catboost if present)
        x = None
        x_label = "Step"
        if "iteration" in df.columns and df["iteration"].notna().any():
            x = df["iteration"]
            x_label = "Iteration"
        elif "epoch" in df.columns and df["epoch"].notna().any():
            x = df["epoch"]
            x_label = "Epoch"
        else:
            x = range(1, len(df) + 1)

        # CatBoost-specialized plotting
        if self.model_name.lower() == "catboost":
            # Accept flexible column names
            train_loss_candidates = ["train_loss", "learn_loss", "loss", "train_rmse", "learn_rmse"]
            val_rmse_candidates = ["val_rmse", "test_rmse", "validation_rmse"]
            val_r2_candidates = ["val_r2", "test_r2", "validation_r2"]

            # Coerce likely numeric columns
            coerce_numeric(["iteration", "epoch"] + train_loss_candidates + val_rmse_candidates + val_r2_candidates)

            def first_present(cands):
                for c in cands:
                    if c in df.columns and df[c].notna().any():
                        return c
                return None

            tr_loss_col = first_present(train_loss_candidates)
            val_rmse_col = first_present(val_rmse_candidates)
            val_r2_col = first_present(val_r2_candidates)

            # Determine number of iterations for chart style
            # Prefer iteration column; fall back to non-null count of any plotted series
            if "iteration" in df.columns and df["iteration"].notna().any():
                n_points = int(df["iteration"].dropna().shape[0])
            else:
                counts = [
                    df[tr_loss_col].dropna().shape[0] if tr_loss_col else 0,
                    df[val_rmse_col].dropna().shape[0] if val_rmse_col else 0,
                    df[val_r2_col].dropna().shape[0] if val_r2_col else 0,
                ]
                n_points = int(max(counts) if counts else len(df))
            use_lines = True #n_points > 20  # if iterations > 20, prefer line chart

            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)

            # Row 1: Training progress (loss/RMSE)
            ax1 = axes[0]
            if tr_loss_col is not None:
                if use_lines:
                    if use_sns:
                        sns.lineplot(x=x, y=df[tr_loss_col], ax=ax1, linewidth=2, label=tr_loss_col.upper())
                    else:
                        ax1.plot(x, df[tr_loss_col], linewidth=2, label=tr_loss_col.upper())
                else:
                    if use_sns:
                        sns.scatterplot(x=x, y=df[tr_loss_col], ax=ax1, s=25, label=tr_loss_col.upper())
                    else:
                        ax1.scatter(x, df[tr_loss_col], s=25, label=tr_loss_col.upper())
            ax1.set_ylabel("Train loss")
            if tr_loss_col is not None:
                ax1.legend(loc="best")
            ax1.grid(True, alpha=0.3)

            # Row 2: Validation RMSE and R²
            ax2 = axes[1]
            r2_axis = ax2.twinx()
            plotted_any = False

            if val_rmse_col is not None:
                if use_lines:
                    if use_sns:
                        sns.lineplot(x=x, y=df[val_rmse_col], ax=ax2, color="#1f77b4", linewidth=2, label="Val RMSE")
                    else:
                        ax2.plot(x, df[val_rmse_col], color="#1f77b4", linewidth=2, label="Val RMSE")
                else:
                    if use_sns:
                        sns.scatterplot(x=x, y=df[val_rmse_col], ax=ax2, color="#1f77b4", s=25, label="Val RMSE")
                    else:
                        ax2.scatter(x, df[val_rmse_col], color="#1f77b4", s=25, label="Val RMSE")
                ax2.set_ylabel("Val RMSE", color="#1f77b4")
                ax2.tick_params(axis='y', labelcolor="#1f77b4")
                ax2.grid(True, alpha=0.3)
                plotted_any = True

            if val_r2_col is not None:
                r2_color = "#2ca02c"
                if use_lines:
                    r2_axis.plot(x, df[val_r2_col], color=r2_color, linewidth=2, label="Val R²")
                else:
                    r2_axis.scatter(x, df[val_r2_col], color=r2_color, s=25, label="Val R²")
                r2_axis.set_ylabel("Val R²", color=r2_color)
                r2_axis.tick_params(axis='y', labelcolor=r2_color)
                r2_axis.set_ylim(0.0, 1.0)
                plotted_any = True

            # Combine legends for bottom row
            if plotted_any:
                h_left, l_left = ax2.get_legend_handles_labels()
                h_right, l_right = r2_axis.get_legend_handles_labels()
                if h_left or h_right:
                    ax2.legend(h_left + h_right, l_left + l_right, loc="best")

            axes[-1].set_xlabel(x_label)
            fig.suptitle(title or "CatBoost training progress", fontsize=14)

            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
                print(f"Training History Monitoring Figure saved to: {save_path}")
            if show:
                plt.show()
            plt.close(fig)
            return

        # -------------- Generic plotting (non-CatBoost) --------------
        # Coerce common numeric columns
        coerce_numeric(["epoch", "train_loss", "val_loss", "val_rmse", "val_r2"])

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)

        # Row 1: Losses
        ax1 = axes[0]
        plotted_any = False
        if "train_loss" in df.columns and df["train_loss"].notna().any():
            if use_sns:
                sns.lineplot(x=x, y=df["train_loss"], ax=ax1, marker="o", linewidth=2, label="Train loss")
            else:
                ax1.plot(x, df["train_loss"], marker="o", linewidth=2, label="Train loss")
            plotted_any = True
        if "val_loss" in df.columns and df["val_loss"].notna().any():
            if use_sns:
                sns.lineplot(x=x, y=df["val_loss"], ax=ax1, marker="o", linewidth=2, label="Val loss")
            else:
                ax1.plot(x, df["val_loss"], marker="o", linewidth=2, label="Val loss")
            plotted_any = True
        ax1.set_ylabel("Loss")
        if plotted_any:
            ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Row 2: RMSE and R² (twin axis)
        ax2 = axes[1]
        r2_axis = ax2.twinx()

        if "val_rmse" in df.columns and df["val_rmse"].notna().any():
            if use_sns:
                sns.lineplot(x=x, y=df["val_rmse"], ax=ax2, color="#1f77b4", marker="o", linewidth=2, label="Val RMSE")
            else:
                ax2.plot(x, df["val_rmse"], color="#1f77b4", marker="o", linewidth=2, label="Val RMSE")
            ax2.set_ylabel("Val RMSE", color="#1f77b4")
            ax2.tick_params(axis='y', labelcolor="#1f77b4")
            ax2.grid(True, alpha=0.3)

        if "val_r2" in df.columns and df["val_r2"].notna().any():
            r2_color = "#2ca02c"
            r2_axis.plot(x, df["val_r2"], color=r2_color, marker="o", linewidth=2, label="Val R²")
            r2_axis.set_ylabel("Val R²", color=r2_color)
            r2_axis.tick_params(axis='y', labelcolor=r2_color)
            r2_axis.set_ylim(0.0, 1.0)

        # Legends for bottom row (combine handles)
        handles_left, labels_left = ax2.get_legend_handles_labels()
        handles_right, labels_right = r2_axis.get_legend_handles_labels()
        if handles_left or handles_right:
            ax2.legend(handles_left + handles_right, labels_left + labels_right, loc="best")

        axes[-1].set_xlabel(x_label)
        fig.suptitle(title, fontsize=14)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Training History Monitoring Figure saved to: {save_path}")
        if show:
            plt.show()
        plt.close(fig)    
    
    @staticmethod
    def r2_keras(y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) ) 
