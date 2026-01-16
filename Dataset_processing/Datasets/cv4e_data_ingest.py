from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
from beartype import beartype


@dataclass(frozen=True, slots=True)
class IngestPaths:
    wavs_dpath: Path
    selections_dpath: Path
    metadata_xlsx_fpath: Path
    out_csv_fpath: Path


@beartype
def get_file_id(nest_id: str, trial: int) -> str:
    nest_id = nest_id.strip()
    return f"{nest_id}_Trial_{trial}_trim"


@beartype
def strip_leading_25(nest_id: str) -> str:
    nest_id = nest_id.strip()
    if not nest_id.startswith("25"):
        return nest_id
    return nest_id[2:]


@beartype
def parse_wav_file_id(wav_fpath: Path) -> str | None:
    # Example stem: "AZ02 Trial 1_trim"
    stem = wav_fpath.stem
    m = re.fullmatch(r"(?P<nest>[A-Za-z0-9]+)\s+Trial\s+(?P<trial>\d+)_trim", stem)
    if m is None:
        return None
    nest_id = m.group("nest")
    trial = int(m.group("trial"))
    return get_file_id(nest_id, trial)


@beartype
def parse_selection_file_id(selection_fpath: Path) -> str | None:
    # Example: "AZ02 Trial 1_trim.Table.1.selections" (+ optional .txt)
    stem = selection_fpath.name
    stem = re.sub(r"\.txt$", "", stem)
    stem = re.sub(r"\.selections$", "", stem)

    # Remove Raven table suffix
    stem = re.sub(r"\.Table\.\d+$", "", stem)

    m = re.fullmatch(r"(?P<nest>[A-Za-z0-9]+)\s+Trial\s+(?P<trial>\d+)_trim", stem)
    if m is None:
        return None
    nest_id = m.group("nest")
    trial = int(m.group("trial"))
    return get_file_id(nest_id, trial)


@beartype
def read_metadata(metadata_xlsx_fpath: Path) -> pd.DataFrame:
    df = pd.read_excel(metadata_xlsx_fpath)

    keep_cols = [
        "NestID",
        "Site",
        "Date",
        "Time",
        "Model",
        "Stage",
        "NestAge",
        "NestlingAge",
        "Trial",
        "Weather",
    ]
    df = df.loc[:, keep_cols].copy()

    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    df["NestID"] = df["NestID"].astype(str).map(strip_leading_25)
    df["Trial"] = pd.to_numeric(df["Trial"], errors="raise").astype(int)

    df["file_id"] = [get_file_id(n, t) for n, t in zip(df["NestID"], df["Trial"], strict=True)]

    df = df.drop_duplicates(subset=["file_id"], keep="first").reset_index(drop=True)

    return df


@beartype
def scan_wavs(wavs_dpath: Path) -> pd.DataFrame:
    wav_fpaths = sorted(p for p in wavs_dpath.rglob("*.wav") if p.is_file())

    rows: list[dict] = []
    for wav_fpath in wav_fpaths:
        file_id = parse_wav_file_id(wav_fpath)
        if file_id is None:
            continue
        rows.append(
            {
                "file_id": file_id,
                "wav_fpath": str(wav_fpath.resolve()),
                "wav_fname": wav_fpath.name[:-4] + ".WAV" if wav_fpath.name.lower().endswith(".wav") else wav_fpath.name,
            }
        )

    return pd.DataFrame(rows).drop_duplicates(subset=["file_id"], keep="first")


@beartype
def read_selections(selections_dpath: Path) -> pd.DataFrame:
    selection_fpaths = sorted(p for p in selections_dpath.rglob("*selections*") if p.is_file())

    dfs: list[pd.DataFrame] = []
    for selection_fpath in selection_fpaths:
        file_id = parse_selection_file_id(selection_fpath)
        if file_id is None:
            continue

        df = pd.read_csv(selection_fpath, sep="\t", engine="python").copy()
        df["selection_fpath"] = str(selection_fpath.resolve())
        df["file_id"] = file_id

        rename_map = {
            "Begin Time (s)": "begin_time_s",
            "End Time (s)": "end_time_s",
            "Low Freq (Hz)": "low_freq_hz",
            "High Freq (Hz)": "high_freq_hz",
            "Delta Time (s)": "delta_time_s",
            "Type": "call_type",
        }
        df = df.rename(columns=rename_map)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)

    out["Channel"] = pd.to_numeric(out["Channel"], errors="coerce")
    out["channel_is_one"] = out["Channel"].eq(1)
    out = out.loc[out["channel_is_one"]].reset_index(drop=True)

    for col in ["begin_time_s", "end_time_s", "low_freq_hz", "high_freq_hz", "delta_time_s"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


@beartype
def make_analysis_table(paths: IngestPaths) -> pd.DataFrame:
    meta_df = read_metadata(paths.metadata_xlsx_fpath)
    wav_df = scan_wavs(paths.wavs_dpath)
    sel_df = read_selections(paths.selections_dpath)

    if sel_df.empty:
        raise RuntimeError(f"No selections found under selections_dpath={paths.selections_dpath}")

    out = sel_df.merge(meta_df, on="file_id", how="left", validate="many_to_one")
    out = out.merge(wav_df, on="file_id", how="left", validate="many_to_one")

    out["has_wav"] = out["wav_fpath"].notna()
    out["has_metadata"] = out["Site"].notna()

    sort_cols = ["file_id", "begin_time_s", "end_time_s"]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    return out


@beartype
def setup_write_csv(paths: IngestPaths) -> None:
    df = make_analysis_table(paths)
    paths.out_csv_fpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.out_csv_fpath, index=False)

    n_rows = len(df)
    n_missing_wav = int((~df["has_wav"]).sum())
    n_missing_meta = int((~df["has_metadata"]).sum())
    print(f"Wrote {n_rows} rows to {paths.out_csv_fpath}")
    print(f"Missing wav path rows: {n_missing_wav}")
    print(f"Missing metadata rows: {n_missing_meta}")


if __name__ == "__main__":
    paths = IngestPaths(
        wavs_dpath=Path(r"C:\Users\sllaw\OneDrive\Documents\OSU Postdoc\Data\Recordings\C4VE\Trimmed"),
        selections_dpath=Path(r"C:\Users\sllaw\OneDrive\Documents\OSU Postdoc\Data\Recordings\C4VE\Selections"),
        metadata_xlsx_fpath=Path(r"C:\Users\sllaw\OneDrive\Documents\OSU Postdoc\Data\Recordings\Recordings_metadata.xlsx"),
        out_csv_fpath=Path(r"C:\Users\sllaw\OneDrive\Documents\OSU Postdoc\Data\Recordings\C4VE\derived\cv4e_calls_channel1.csv"),
    )
    setup_write_csv(paths)
