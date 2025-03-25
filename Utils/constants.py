# Feature name mappings for user-friendly display
rename_mapping = {
    "temperature_2m_mean": "Mean Temperature",
    "apparent_temperature_mean": "Feels-Like Temperature",
    "precipitation_sum": "Total Precipitation",
    "daylight_duration": "Daylight Duration",
    "wind_speed_10m_max": "Max Wind Speed"
}
# Inverse mapping to recover original names when needed
inverse_rename_mapping = {v: k for k, v in rename_mapping.items()}
