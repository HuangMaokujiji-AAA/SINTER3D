#!/usr/bin/env python3
"""Download and prepare four DLPFC Visium slices as AnnData files."""

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path


BASE_URL = "https://huggingface.co/datasets/han-shu/st_datasets/resolve/main/DLPFC"
SLICES = {
    "151673": "fd2a6e8e4622337bc2c3a79c5f908fe3c7bd0212ed7922dc157de37a5fac172b",
    "151674": "720efac7a847413d3dbc41baf1da1128656641b6bf4bfdd4fcb76f0e0f64686c",
    "151675": "146c723413d162f42ec1e0fcfa0bf5f74de0231ebcb8e166c24870ff3c30b358",
    "151676": "ebfa7afb77ac7c5faea8e2d7890169c69b77e6744d74e893bbcab8ea076dcc1e",
}
LABEL_COLUMNS = ("layer_guess_reordered", "layer", "ground_truth", "spatialLIBD")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, expected_sha256):
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "SINTER3D-data-preparer/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        received = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            received += len(block)
            if total:
                print(f"\r  {destination.name}: {received / total:6.1%}", end="", flush=True)
    print()
    actual = sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink()
        raise ValueError(
            f"Checksum failed for {destination.name}: expected {expected_sha256}, got {actual}"
        )
    os.replace(temporary, destination)


def has_lowres_image(adata):
    spatial = adata.uns.get("spatial", {})
    return any("lowres" in item.get("images", {}) for item in spatial.values())


def prepare(path):
    try:
        import anndata as ad
        import numpy as np
    except ImportError as error:
        raise SystemExit("Install the project requirements before running this script") from error

    adata = ad.read_h5ad(path)
    changed = False

    if "cluster" not in adata.obs:
        source = next((name for name in LABEL_COLUMNS if name in adata.obs), None)
        if source is None:
            raise ValueError(
                f"{path.name} has no cluster label. Available obs columns: "
                + ", ".join(adata.obs.columns)
            )
        adata.obs["cluster"] = adata.obs[source]
        changed = True
        print(f"  cluster <- obs[{source!r}]")

    labels = adata.obs["cluster"].astype("string")
    keep = labels.notna() & labels.fillna("").str.lower().ne("nan")
    if not keep.all():
        adata = adata[keep.to_numpy(dtype=bool)].copy()
        changed = True
        print(f"  removed {(~keep).sum()} spots without a layer label")
    adata.obs["cluster"] = adata.obs["cluster"].astype(str).astype("category")

    if "spatial" not in adata.obsm:
        coordinate_columns = next(
            (
                columns
                for columns in (
                    ("pxl_col_in_fullres", "pxl_row_in_fullres"),
                    ("imagecol", "imagerow"),
                )
                if all(column in adata.obs for column in columns)
            ),
            None,
        )
        if coordinate_columns is None:
            raise ValueError(f"{path.name} has no obsm['spatial'] coordinates")
        adata.obsm["spatial"] = adata.obs[list(coordinate_columns)].to_numpy(dtype=float)
        changed = True

    coordinates = np.asarray(adata.obsm["spatial"])
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"Invalid obsm['spatial'] coordinates in {path.name}: {coordinates.shape}")
    if not has_lowres_image(adata):
        raise ValueError(f"{path.name} has no low-resolution image in uns['spatial']")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"Empty AnnData object in {path.name}")

    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
        changed = True

    if changed:
        temporary = path.with_name(path.stem + ".prepared.h5ad")
        adata.write_h5ad(temporary, compression="gzip")
        os.replace(temporary, path)

    print(
        f"Ready: {path.name}: {adata.n_obs} spots x {adata.n_vars} genes; "
        f"labels={list(adata.obs['cluster'].cat.categories)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/private_sdb/huangyingjie/data/SINTER3D"),
    )
    parser.add_argument("--force", action="store_true", help="Download files again")
    parser.add_argument("--check-only", action="store_true", help="Only validate existing files")
    args = parser.parse_args()

    destination = args.data_root / "DLPFC"
    destination.mkdir(parents=True, exist_ok=True)

    for slice_id, expected_sha256 in SLICES.items():
        path = destination / f"DLPFC_{slice_id}.h5ad"
        if not args.check_only and (args.force or not path.is_file()):
            print(f"Downloading {path.name}")
            download(f"{BASE_URL}/{path.name}?download=true", path, expected_sha256)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        prepare(path)

    print(f"Directory: {destination}")


if __name__ == "__main__":
    main()
