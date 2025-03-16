import pandas as pd

def time_to_seconds(time_str):
    """Converts a time string (MM:SS.sss) to seconds."""
    if time_str is None or time_str == "DNS":
        return None
    minutes, seconds = map(float, time_str.split(':'))
    return minutes * 60 + seconds

qualifying_data_raw = pd.DataFrame({
    "Driver": ["Norris", "Piastri", "Verstappen", "Russell", "Tsunoda", "Albon", "Leclerc", "Hamilton", "Gasly", "Sainz", "Hadjar", "Alonso", "Stroll", "Doohan", "Bortoleto", "Antonelli", "Hulkenberg", "Lawson", "Ocon", "Bearman"],
    "Q1 (s)": ["1:15.912", "1:16.062", "1:16.018", "1:15.971", "1:16.225", "1:16.245", "1:16.029", "1:16.213", "1:16.328", "1:16.360", "1:16.354", "1:16.288", "1:16.369", "1:16.315", "1:16.516", "1:16.525", "1:16.579", "1:17.094", "1:17.147", "DNS"],
    "Q2 (s)": ["1:15.415", "1:15.468", "1:15.565", "1:15.798", "1:16.009", "1:16.017", "1:15.827", "1:15.919", "1:16.112", "1:15.931", "1:16.175", "1:16.453", "1:16.483", "1:16.863", "1:17.520", None, None, None, None, None],
    "Q3 (s)": ["1:15.096", "1:15.180", "1:15.481", "1:15.546", "1:15.670", "1:15.737", "1:15.755", "1:15.973", "1:15.980", "1:16.062", None, None, None, None, None, None, None, None, None, None],
})

# Convert all time strings to seconds
qualifying_data_raw["Q1 (s)"] = qualifying_data_raw["Q1 (s)"].apply(time_to_seconds)
qualifying_data_raw["Q2 (s)"] = qualifying_data_raw["Q2 (s)"].apply(time_to_seconds)
qualifying_data_raw["Q3 (s)"] = qualifying_data_raw["Q3 (s)"].apply(time_to_seconds)

# Create a new DataFrame for mapped qualifying times
result_data = []
for index, row in qualifying_data_raw.iterrows():
    driver = row["Driver"]
    # Use the best time from Q3 if available, otherwise Q2, otherwise Q1
    if pd.notna(row["Q3 (s)"]):
        result_data.append({"Driver": driver, "QualifyingTime (s)": row["Q3 (s)"], "Session": "Q3"})
    elif pd.notna(row["Q2 (s)"]):
        result_data.append({"Driver": driver, "QualifyingTime (s)": row["Q2 (s)"], "Session": "Q2"})
    elif pd.notna(row["Q1 (s)"]):
        result_data.append({"Driver": driver, "QualifyingTime (s)": row["Q1 (s)"], "Session": "Q1"})
    else:
        # If no time available at all (e.g., DNS in all sessions)
        result_data.append({"Driver": driver, "QualifyingTime (s)": None, "Session": "None"})

qualifying_processed = pd.DataFrame(result_data)

# Add driver full names mapping to match prediction.py format
driver_full_names = {
    "Norris": "Lando Norris", 
    "Piastri": "Oscar Piastri", 
    "Verstappen": "Max Verstappen", 
    "Russell": "George Russell", 
    "Tsunoda": "Yuki Tsunoda", 
    "Albon": "Alex Albon", 
    "Leclerc": "Charles Leclerc", 
    "Hamilton": "Lewis Hamilton", 
    "Gasly": "Pierre Gasly", 
    "Sainz": "Carlos Sainz", 
    "Hadjar": "Isack Hadjar", 
    "Alonso": "Fernando Alonso", 
    "Stroll": "Lance Stroll", 
    "Doohan": "Jack Doohan", 
    "Bortoleto": "Gabriel Bortoleto", 
    "Antonelli": "Kimi Antonelli", 
    "Hulkenberg": "Nico Hulkenberg", 
    "Lawson": "Liam Lawson", 
    "Ocon": "Esteban Ocon", 
    "Bearman": "Ollie Bearman"
}

# Add full driver names and driver codes
qualifying_processed["FullName"] = qualifying_processed["Driver"].map(driver_full_names)

# Create proper driver code mapping
driver_code_mapping = {
    "Norris": "NOR", "Piastri": "PIA", "Verstappen": "VER", 
    "Russell": "RUS", "Tsunoda": "TSU", "Albon": "ALB", 
    "Leclerc": "LEC", "Hamilton": "HAM", "Gasly": "GAS", 
    "Sainz": "SAI", "Hadjar": "HAD", "Alonso": "ALO", 
    "Stroll": "STR", "Doohan": "DOO", "Bortoleto": "BOR", 
    "Antonelli": "ANT", "Hulkenberg": "HUL", "Lawson": "LAW", 
    "Ocon": "OCO", "Bearman": "BEA"
}

qualifying_processed["DriverCode"] = qualifying_processed["Driver"].map(driver_code_mapping)

# Print detailed information about each driver's qualifying time
print("=== Detailed Qualifying Times ===")
for _, row in qualifying_processed.iterrows():
    print(f"{row['FullName']} ({row['DriverCode']}): {row['QualifyingTime (s)']:.3f}s - Best from {row['Session']}")

print("\n=== Qualifying Results for Integration with prediction.py ===")
print(qualifying_processed[["FullName", "QualifyingTime (s)", "DriverCode"]])

# Save this data to a CSV file for easy import into prediction.py
qualifying_processed[["FullName", "QualifyingTime (s)", "DriverCode"]].to_csv("qualifying_times.csv", index=False)
print("\nQualifying times saved to 'qualifying_times.csv' for easy import into prediction.py")