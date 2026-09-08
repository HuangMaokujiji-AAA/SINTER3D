#!/usr/bin/env python3
"""Download and prepare the GSE144136 snRNA-seq reference for DLPFC."""

import argparse
import csv
import gzip
import os
import shutil
import urllib.request
from pathlib import Path


BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE144nnn/GSE144136/suppl"
FILES = (
    "GSE144136_GeneBarcodeMatrix_Annotated.mtx",
    "GSE144136_CellNames.csv",
    "GSE144136_GeneNames.csv",
)


def download(url, destination):
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
    os.replace(temporary, destination)


def decompress(archive, destination):
    temporary = destination.with_name(destination.name + ".part")
    with gzip.open(archive, "rb") as source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    os.replace(temporary, destination)


def csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def matrix_shape(path):
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if not line.startswith("%"):
                rows, columns, nonzero = map(int, line.split())
                return rows, columns, nonzero
    raise ValueError(f"Cannot find Matrix Market dimensions in {path}")


def validate(directory):
    matrix = directory / FILES[0]
    cells = directory / FILES[1]
    genes = directory / FILES[2]
    missing = [str(path) for path in (matrix, cells, genes) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prepared files:\n" + "\n".join(missing))

    n_genes, n_cells, nonzero = matrix_shape(matrix)
    cell_rows = csv_rows(cells)
    gene_rows = csv_rows(genes)
    if (n_genes, n_cells) != (gene_rows, cell_rows):
        raise ValueError(
            "Dimension mismatch: matrix is "
            f"{n_genes} genes x {n_cells} cells, but CSV files contain "
            f"{gene_rows} genes and {cell_rows} cells"
        )
    print(f"Ready: {n_genes} genes x {n_cells} nuclei, {nonzero} non-zero counts")
    print(f"Directory: {directory}")


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

    destination = args.data_root / "spatialLIBD" / "scRNA"
    destination.mkdir(parents=True, exist_ok=True)

    if not args.check_only:
        for filename in FILES:
            output = destination / filename
            if output.is_file() and not args.force:
                print(f"Using existing {output.name}")
                continue
            archive = destination / f"{filename}.gz"
            if args.force or not archive.is_file():
                print(f"Downloading {archive.name}")
                download(f"{BASE_URL}/{archive.name}", archive)
            print(f"Extracting {archive.name}")
            decompress(archive, output)

    validate(destination)


if __name__ == "__main__":
    main()
