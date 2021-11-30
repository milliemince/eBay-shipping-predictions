from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="geoapiExercises")
zipcode1 = "61000"
location = geolocator.geocode(zipcode1)
country = str(location).split(',')[-1].strip()
print(country)