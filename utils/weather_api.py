import requests 
import os
API_KEY = "68e845aa6644456b5cfa53e52de61ba3"   # replace with your key

#def get_weather(city):
 #   url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
  #  response = requests.get(url).json()
   # if response.get("cod") != 200:
    #    return {"error": response.get("message","No data found")}
    #return {
     #   "city": response["name"],
      #  "temp": response["main"]["temp"],
       # "humidity": response["main"]["humidity"],
        #"weather": response["weather"][0]["description"],
        #"wind": response["wind"]["speed"]
    #}



def get_forecast(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()
    if response.get("cod") != "200":
        return {"error": response.get("message", "No forecast data found")}

    forecast_data = []
    for entry in response["list"]:
        forecast_data.append({
            "datetime": entry["dt_txt"],
            "temp": entry["main"]["temp"],
            "weather": entry["weather"][0]["description"]
        })
    return forecast_data
