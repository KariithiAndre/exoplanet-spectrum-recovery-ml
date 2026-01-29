#!/usr/bin/env python
"""
Download exoplanet data from NASA Exoplanet Archive.

This script fetches:
- Confirmed exoplanet parameters
- Transmission spectroscopy measurements
- Stellar properties
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm


NASA_EXOPLANET_ARCHIVE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def query_exoplanet_archive(
    query: str,
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Query the NASA Exoplanet Archive TAP service.
    
    Args:
        query: ADQL query string
        output_format: Output format (csv, json, votable)
        
    Returns:
        DataFrame with query results
    """
    params = {
        "query": query,
        "format": output_format,
    }
    
    response = requests.get(NASA_EXOPLANET_ARCHIVE_URL, params=params)
    response.raise_for_status()
    
    from io import StringIO
    return pd.read_csv(StringIO(response.text))


def download_confirmed_planets(output_path: Path) -> pd.DataFrame:
    """Download confirmed exoplanet parameters."""
    print("Downloading confirmed exoplanet catalog...")
    
    query = """
    SELECT 
        pl_name, hostname, sy_snum, sy_pnum,
        discoverymethod, disc_year,
        pl_orbper, pl_orbpererr1, pl_orbpererr2,
        pl_rade, pl_radeerr1, pl_radeerr2,
        pl_bmasse, pl_bmasseerr1, pl_bmasseerr2,
        pl_eqt, pl_eqterr1, pl_eqterr2,
        st_teff, st_tefferr1, st_tefferr2,
        st_rad, st_raderr1, st_raderr2,
        st_mass, st_masserr1, st_masserr2,
        sy_dist, sy_disterr1, sy_disterr2,
        sy_vmag, sy_kmag
    FROM ps
    WHERE default_flag = 1
    ORDER BY disc_year DESC
    """
    
    df = query_exoplanet_archive(query)
    df.to_csv(output_path / "confirmed_planets.csv", index=False)
    print(f"  Downloaded {len(df)} confirmed planets")
    
    return df


def download_transmission_targets(output_path: Path) -> pd.DataFrame:
    """Download planets suitable for transmission spectroscopy."""
    print("Downloading transmission spectroscopy targets...")
    
    # Planets with measured radii and equilibrium temperatures
    # that are likely amenable to transmission spectroscopy
    query = """
    SELECT 
        pl_name, hostname,
        pl_rade, pl_bmasse, pl_eqt, pl_trandep,
        st_teff, st_rad, st_mass, sy_kmag,
        pl_orbper, pl_orbsmax
    FROM ps
    WHERE default_flag = 1
      AND pl_rade IS NOT NULL
      AND pl_eqt IS NOT NULL
      AND pl_trandep IS NOT NULL
    ORDER BY pl_trandep DESC
    """
    
    df = query_exoplanet_archive(query)
    
    # Calculate transmission spectroscopy metric (TSM) approximation
    # TSM ∝ R_p^2 * T_eq / (M_p * R_*^2) * 10^(-K/5)
    df["tsm_approx"] = (
        df["pl_rade"] ** 2 * df["pl_eqt"] / 
        (df["pl_bmasse"].fillna(df["pl_rade"] ** 2.06)) *  # Mass-radius relation fallback
        10 ** (-df["sy_kmag"].fillna(10) / 5)
    )
    df = df.sort_values("tsm_approx", ascending=False)
    
    df.to_csv(output_path / "transmission_targets.csv", index=False)
    print(f"  Downloaded {len(df)} transmission targets")
    
    return df


def main(args):
    """Main download function."""
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download catalogs
        planets = download_confirmed_planets(output_path)
        transmission = download_transmission_targets(output_path)
        
        # Summary statistics
        print("\n=== Download Summary ===")
        print(f"Total confirmed planets: {len(planets)}")
        print(f"Transmission targets: {len(transmission)}")
        print(f"Discovery methods: {planets['discoverymethod'].nunique()}")
        print(f"Years covered: {planets['disc_year'].min():.0f} - {planets['disc_year'].max():.0f}")
        
        # Top transmission targets
        print("\nTop 10 Transmission Spectroscopy Targets:")
        top_targets = transmission.head(10)[["pl_name", "pl_rade", "pl_eqt", "tsm_approx"]]
        print(top_targets.to_string(index=False))
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NASA Exoplanet Archive data")
    
    parser.add_argument("--output-dir", type=str, default="./data/catalogs",
                        help="Output directory for downloaded data")
    
    args = parser.parse_args()
    main(args)
