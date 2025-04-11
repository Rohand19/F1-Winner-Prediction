import requests
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class WeatherService:
    """Service to fetch real-time and forecast weather data for F1 race locations."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the weather service.
        
        Args:
            api_key: Optional API key for weather service. If not provided, will try to read from environment variable.
        """
        self.api_key = api_key or os.getenv('WEATHER_API_KEY')
        if not self.api_key:
            logger.warning("No API key provided for weather service. Using default weather data.")
        
        # Cache for weather data to avoid redundant API calls
        self._cache = {}
        self._cache_expiry = {}
        self.cache_duration = timedelta(hours=1)  # Weather data refreshed hourly
    
    def get_current_weather(self, location: str) -> Dict[str, Any]:
        """
        Get current weather for a location.
        
        Args:
            location: City or circuit name
            
        Returns:
            Dictionary containing weather data
        """
        # Check cache first
        cache_key = f"current_{location}"
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached weather data for {location}")
            return self._cache[cache_key]
        
        # Location mapping for circuits
        location_mapping = {
            "Bahrain": "Sakhir,Bahrain",
            "Jeddah": "Jeddah,Saudi Arabia",
            "Melbourne": "Melbourne,Australia",
            "Shanghai": "Shanghai,China",
            "Miami": "Miami,United States",
            "Imola": "Imola,Italy",
            "Monaco": "Monte Carlo,Monaco",
            "Montreal": "Montreal,Canada",
            "Barcelona": "Barcelona,Spain",
            "Spielberg": "Spielberg,Austria",
            "Silverstone": "Silverstone,United Kingdom",
            "Budapest": "Budapest,Hungary",
            "Spa": "Spa,Belgium",
            "Zandvoort": "Zandvoort,Netherlands",
            "Monza": "Monza,Italy",
            "Baku": "Baku,Azerbaijan",
            "Singapore": "Singapore,Singapore",
            "Austin": "Austin,United States",
            "Mexico City": "Mexico City,Mexico",
            "Sao Paulo": "Sao Paulo,Brazil",
            "Las Vegas": "Las Vegas,United States",
            "Lusail": "Lusail,Qatar",
            "Yas Marina": "Abu Dhabi,United Arab Emirates",
        }
        
        # Map circuit name to location if needed
        query_location = location_mapping.get(location, location)
        
        if not self.api_key:
            return self._get_default_weather(location)
        
        try:
            # Using OpenWeatherMap API as an example
            # You could replace this with any weather API service
            url = f"https://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": query_location,
                "appid": self.api_key,
                "units": "metric"  # Use metric units (Celsius)
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                weather_data = {
                    "track_temp": data["main"]["temp"] + 5,  # Track temp typically higher than air
                    "air_temp": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"] * 3.6,  # Convert m/s to km/h
                    "rain_chance": self._calculate_rain_chance(data),
                    "changing_conditions": self._predict_changing_conditions(data),
                    "weather_description": data["weather"][0]["description"],
                    "timestamp": datetime.now().isoformat(),
                    "source": "OpenWeatherMap API",
                }
                
                # Cache the results
                self._cache[cache_key] = weather_data
                self._cache_expiry[cache_key] = datetime.now() + self.cache_duration
                
                logger.info(f"Fetched live weather data for {location}: {weather_data['air_temp']}°C, {weather_data['humidity']}% humidity")
                return weather_data
            else:
                logger.error(f"Error fetching weather data: {response.status_code}, {response.text}")
                return self._get_default_weather(location)
                
        except Exception as e:
            logger.error(f"Exception while fetching weather data for {location}: {str(e)}")
            return self._get_default_weather(location)
    
    def get_race_weather_forecast(self, location: str, race_date: datetime) -> Dict[str, Any]:
        """
        Get weather forecast for a specific race date.
        
        Args:
            location: City or circuit name
            race_date: Date and time of the race
            
        Returns:
            Dictionary containing forecast weather data
        """
        # Check cache first
        cache_key = f"forecast_{location}_{race_date.strftime('%Y%m%d')}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        # Implementation for forecast would be similar to current weather
        # but using a forecast API endpoint
        # For now, return default weather
        return self._get_default_weather(location)
    
    def _calculate_rain_chance(self, weather_data: Dict[str, Any]) -> float:
        """Calculate probability of rain based on weather conditions."""
        # Extract relevant data from the weather API response
        weather_id = weather_data["weather"][0]["id"]
        humidity = weather_data["main"]["humidity"]
        
        # Weather condition codes from OpenWeatherMap
        # 2xx: Thunderstorm, 3xx: Drizzle, 5xx: Rain, 6xx: Snow, 7xx: Atmosphere, 800: Clear, 80x: Clouds
        
        # Already raining
        if weather_id < 700:
            # Thunderstorm, drizzle, rain, or snow
            return min(1.0, 0.8 + humidity / 500)  # Higher chance with higher humidity
        
        # Cloudy conditions
        elif weather_id >= 801:
            # Convert cloud cover to probability (higher clouds = higher chance)
            cloud_factor = (weather_id - 800) / 4 if weather_id <= 804 else 0.8
            humidity_factor = humidity / 100  # Higher humidity = higher chance
            
            # Combine factors with weights
            return min(0.8, cloud_factor * 0.7 + humidity_factor * 0.3)
        
        # Clear sky
        else:
            # Very low chance but still possible if humidity is very high
            return max(0.0, (humidity - 70) / 200) if humidity > 70 else 0.0
    
    def _predict_changing_conditions(self, weather_data: Dict[str, Any]) -> bool:
        """Predict if weather conditions are likely to change during the race."""
        if "wind" not in weather_data:
            return False
            
        # Higher wind speeds often indicate changing conditions
        if weather_data["wind"]["speed"] > 20:  # 20 m/s ~ 72 km/h
            return True
            
        # Specific weather conditions that suggest instability
        weather_id = weather_data["weather"][0]["id"]
        unstable_conditions = [200, 210, 220, 230, 300, 310, 500, 520, 531]  # Various unstable weather types
        
        return weather_id in unstable_conditions or weather_id % 100 < 20  # Weather IDs ending in low numbers often indicate changing conditions
    
    def _get_default_weather(self, location: str) -> Dict[str, Any]:
        """Get default weather for a location when API is unavailable."""
        # Default weather data based on typical conditions for known circuits
        default_weather = {
            "Bahrain": {"track_temp": 45.0, "air_temp": 35.0, "humidity": 40.0, "rain_chance": 0.05},
            "Jeddah": {"track_temp": 40.0, "air_temp": 32.0, "humidity": 60.0, "rain_chance": 0.02},
            "Melbourne": {"track_temp": 30.0, "air_temp": 22.0, "humidity": 55.0, "rain_chance": 0.30},
            "Shanghai": {"track_temp": 30.0, "air_temp": 25.0, "humidity": 60.0, "rain_chance": 0.25},
            "Miami": {"track_temp": 50.0, "air_temp": 33.0, "humidity": 70.0, "rain_chance": 0.35},
            "Imola": {"track_temp": 28.0, "air_temp": 22.0, "humidity": 65.0, "rain_chance": 0.40},
            "Monaco": {"track_temp": 32.0, "air_temp": 24.0, "humidity": 50.0, "rain_chance": 0.20},
            "Barcelona": {"track_temp": 40.0, "air_temp": 25.0, "humidity": 45.0, "rain_chance": 0.15},
            "Spa": {"track_temp": 25.0, "air_temp": 18.0, "humidity": 75.0, "rain_chance": 0.60},
            "Monza": {"track_temp": 35.0, "air_temp": 28.0, "humidity": 50.0, "rain_chance": 0.10},
            "Singapore": {"track_temp": 38.0, "air_temp": 30.0, "humidity": 85.0, "rain_chance": 0.50},
        }
        
        # Get default data for this location, or use a general default
        weather = default_weather.get(location, {
            "track_temp": 30.0, 
            "air_temp": 25.0, 
            "humidity": 60.0, 
            "rain_chance": 0.20
        })
        
        # Add other required fields
        weather["changing_conditions"] = weather.get("rain_chance", 0) > 0.3
        weather["wind_speed"] = 10.0  # Default 10 km/h
        weather["weather_description"] = "Default weather data (API not available)"
        weather["timestamp"] = datetime.now().isoformat()
        weather["source"] = "Default data"
        
        logger.info(f"Using default weather data for {location}")
        return weather
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key in self._cache and key in self._cache_expiry:
            if datetime.now() < self._cache_expiry[key]:
                return True
        return False 