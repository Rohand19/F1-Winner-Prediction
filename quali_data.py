#!/usr/bin/env python3
"""
F1 Qualifying Data Generator
Generate fictional qualifying data for F1 race predictions.
"""

import argparse
import sys
import csv
import random
import os
import pandas as pd
from datetime import datetime

# Default driver names and approximate qualifying times (in seconds)
DEFAULT_DRIVERS = {
    "Lando Norris": {"code": "NOR", "base_time": 75.1, "variance": 0.3},
    "Oscar Piastri": {"code": "PIA", "base_time": 75.2, "variance": 0.3},
    "Max Verstappen": {"code": "VER", "base_time": 75.5, "variance": 0.3},
    "George Russell": {"code": "RUS", "base_time": 75.5, "variance": 0.3},
    "Yuki Tsunoda": {"code": "TSU", "base_time": 75.7, "variance": 0.3},
    "Alex Albon": {"code": "ALB", "base_time": 75.7, "variance": 0.3},
    "Charles Leclerc": {"code": "LEC", "base_time": 75.7, "variance": 0.3},
    "Lewis Hamilton": {"code": "HAM", "base_time": 75.5, "variance": 0.3},
    "Pierre Gasly": {"code": "GAS", "base_time": 75.9, "variance": 0.3},
    "Carlos Sainz": {"code": "SAI", "base_time": 76.0, "variance": 0.3},
    "Isack Hadjar": {"code": "HAD", "base_time": 76.1, "variance": 0.4},
    "Fernando Alonso": {"code": "ALO", "base_time": 76.2, "variance": 0.3},
    "Lance Stroll": {"code": "STR", "base_time": 76.3, "variance": 0.3},
    "Jack Doohan": {"code": "DOO", "base_time": 76.3, "variance": 0.4},
    "Gabriel Bortoleto": {"code": "BOR", "base_time": 76.4, "variance": 0.4},
    "Kimi Antonelli": {"code": "ANT", "base_time": 76.4, "variance": 0.4},
    "Nico Hulkenberg": {"code": "HUL", "base_time": 76.5, "variance": 0.3},
    "Liam Lawson": {"code": "LAW", "base_time": 76.5, "variance": 0.4},
    "Esteban Ocon": {"code": "OCO", "base_time": 76.6, "variance": 0.3},
    "Ollie Bearman": {"code": "BEA", "base_time": 76.6, "variance": 0.4}
}

# Different circuit base times (approximate)
CIRCUIT_BASE_TIMES = {
    "Australian Grand Prix": 75.0,
    "Monaco Grand Prix": 73.0,
    "British Grand Prix": 86.0,
    "Italian Grand Prix": 80.0,
    "Singapore Grand Prix": 92.0,
    "Japanese Grand Prix": 89.0,
    "United States Grand Prix": 93.0,
    "Brazilian Grand Prix": 69.0,
    "Abu Dhabi Grand Prix": 82.0,
    "Hungarian Grand Prix": 76.0,
    "Belgian Grand Prix": 106.0,
    "Spanish Grand Prix": 78.0,
    "Canadian Grand Prix": 74.0,
    "Austrian Grand Prix": 65.0,
    "Dutch Grand Prix": 71.0,
}

def normalize_quali_times(quali_times, circuit_name="Australian Grand Prix"):
    """Adjust qualifying times for a specific circuit"""
    base_time = CIRCUIT_BASE_TIMES.get(circuit_name, 75.0)
    australian_base = 75.0
    
    scale_factor = base_time / australian_base
    
    for driver, data in quali_times.items():
        data["base_time"] = data["base_time"] * scale_factor
    
    return quali_times

def generate_qualifying_data(circuit_name, seed=None, randomize=True):
    """Generate fictional qualifying data for the specified circuit"""
    if seed is not None:
        random.seed(seed)
    
    # Copy the default drivers and adjust times for the circuit
    quali_times = DEFAULT_DRIVERS.copy()
    quali_times = normalize_quali_times(quali_times, circuit_name)
    
    # Generate times with some random variation
    data = []
    for driver, info in quali_times.items():
        variation = random.uniform(-info["variance"], info["variance"]) if randomize else 0
        quali_time = round(info["base_time"] + variation, 3)
        
        data.append({
            "Driver": driver,
            "DriverCode": info["code"],
            "QualifyingTime (s)": quali_time
        })
    
    # Sort by qualifying time
    data = sorted(data, key=lambda x: x["QualifyingTime (s)"])
    
    return pd.DataFrame(data)

def save_qualifying_data(data, circuit_name, year=2025):
    """Save qualifying data to a CSV file"""
    try:
        # Create a data directory if it doesn't exist
        data_dir = 'data'
        if not os.path.exists(data_dir):
            print(f"Creating data directory: {data_dir}")
            os.makedirs(data_dir)
            
        circuit_short = circuit_name.lower().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(data_dir, f"qualifying_{circuit_short}_{year}_{timestamp}.csv")
        
        data.to_csv(filename, index=False)
        print(f"Qualifying data saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving qualifying data: {e}")
        return None

def display_qualifying_results(data):
    """Display qualifying results in a formatted way"""
    print("\n=== Qualifying Results ===\n")
    for i, (_, row) in enumerate(data.iterrows(), 1):
        print(f"{i}. {row['Driver']} ({row['DriverCode']}) - {row['QualifyingTime (s)']:.3f}s")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="F1 Qualifying Data Generator")
    parser.add_argument("--circuit", type=str, default="Australian Grand Prix",
                        help="Circuit name (default: 'Australian Grand Prix')")
    parser.add_argument("--year", type=int, default=2025,
                        help="Year of qualifying (default: 2025)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: None)")
    parser.add_argument("--no-randomize", action="store_true",
                        help="Disable randomization of qualifying times")
    parser.add_argument("--show-circuits", action="store_true",
                        help="Show list of available circuits")
    
    args = parser.parse_args()
    
    if args.show_circuits:
        print("Available circuits:")
        for circuit in sorted(CIRCUIT_BASE_TIMES.keys()):
            print(f"- {circuit}")
        return 0
    
    print(f"Generating qualifying data for {args.year} {args.circuit}")
    
    # Generate qualifying data
    quali_data = generate_qualifying_data(
        args.circuit, 
        seed=args.seed,
        randomize=not args.no_randomize
    )
    
    # Display results
    display_qualifying_results(quali_data)
    
    # Save to CSV
    output_file = save_qualifying_data(quali_data, args.circuit, args.year)
    
    if output_file:
        print(f"\nTo use this qualifying data with the prediction model, run:")
        print(f"python main.py --race-name \"{args.circuit}\" --quali-data {output_file}")
    else:
        print("\nFailed to save qualifying data. Please check permissions and try again.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
