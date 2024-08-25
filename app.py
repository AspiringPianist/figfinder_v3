from flask import Flask, request, jsonify, render_template
from calendar_utils import authenticate_google_calendar, get_upcoming_events, create_group_calendar, add_event_to_calendar, clear_calendar, get_calendar_events
from group_utils import Group, User
from googleapiclient.discovery import build
from datetime import datetime
from flask import abort
from flask_cors import CORS
from group_utils import Group
import openai, json, requests
from pydantic import BaseModel, Field
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from datetime import timedelta
from serpapi import GoogleSearch
from typing import List, Optional

class Location(BaseModel):
    name: str = Field(description="Name of the location")
    place_id: str = Field(description="Unique identifier for the place")
    rating: Optional[float] = Field(description="Average rating of the location")
    reviews: Optional[int] = Field(description="Number of reviews for the location")
    price: Optional[str] = Field(description="Price level indicator (e.g., $, $$, $$$)")
    type: str = Field(description="Primary type or category of the location")
    address: str = Field(description="Full address of the location")
    open_state: Optional[str] = Field(description="Current open/closed status of the location")
    phone: Optional[str] = Field(description="Contact phone number for the location")
    website: Optional[str] = Field(description="Official website URL of the location")
    description: Optional[str] = Field(description="Brief description or overview of the location")

class TravelPlan(BaseModel):
    locations: List[Location] = Field(description="List of locations included in the travel plan")
    itinerary: List[str] = Field(description="Day-by-day itinerary for the travel plan")

class Suggestions(BaseModel):
    travel_plan: TravelPlan = Field(description="Comprehensive travel plan including locations and itinerary")

parser = PydanticOutputParser(pydantic_object=Suggestions)

OPENWEATHER_API_KEY = "be2b26c47fb0359ee0369eb2d7f84067"
SERPAPI_API_KEY = "d8be108b46854add4fcddb16e3d168da47c031553bbbe82ac62a387c71333199"

# def extract_locations(ai_output):
#     prompt = f"Extract the city names and brief descriptions from this text: {ai_output}. Format the output as a JSON list of objects with 'name' field."
    
#     client = openai.OpenAI(
#         api_key="d546f9f2469f46799b08a638d01fbd98",
#         base_url="https://api.aimlapi.com/",
#     )

#     chat_completion = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant that extracts location information."},
#             {"role": "user", "content": prompt},
#         ],
#         temperature=0.3,
#         max_tokens=200,
#     )

#     locations_json = chat_completion.choices[0].message.content
#     parsed_locations = Locations.parse_raw(locations_json)
#     return parsed_locations.locations

def verify_group_exists(group_id):
    try:
        Group.get_group_by_id(group_id)
        return True
    except ValueError:
        return False


app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/api/schedule/analyze', methods=['POST'])
def analyze_availability():
    data = request.json
    with open('groups_database.json', 'r') as f:
        groups_data = json.load(f)
    group_id = data.get('groupId')
    preferences = data.get('preferences')

    # Fetch group data
    group_info = groups_data.get(group_id)
    if not group_info:
        return jsonify({"error": "Group not found"}), 404

    # Construct the prompt for the AI
    system_content = "You are an AI assistant that analyzes group availability and preferences for travel planning."
    user_content = f"Analyze the availability for group '{group_info['name']}' with travel dates {group_info['travel_dates']} and the following preferences: {preferences}"


    client = openai.OpenAI(
        api_key="d546f9f2469f46799b08a638d01fbd98",
        base_url="https://api.aimlapi.com/",
    )
    
    chat_completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    
    analysis = chat_completion.choices[0].message.content
    
    return jsonify({
        "analysis": {
            "optimalSlots": analysis
        }
    })

parser = PydanticOutputParser(pydantic_object=Suggestions)

def get_lat_lon(city_name):
    geocoding_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={OPENWEATHER_API_KEY}"
    response = requests.get(geocoding_url)
    data = response.json()
    if data:
        return data[0]['lat'], data[0]['lon']
    return None, None

@app.route('/api/schedule/suggest', methods=['POST'])
def generate_suggestions():
    data = request.json
    group_id = data.get('groupId')
    preferences = data.get('preferences')
    location = data.get('location', 'New York, NY')

    with open('groups_database.json', 'r') as f:
        groups_data = json.load(f)

    group_info = groups_data.get(group_id)
    if not group_info:
        return jsonify({"error": "Group not found"}), 404

    start_date = datetime.fromisoformat(group_info['travel_dates'][0])
    end_date = datetime.fromisoformat(group_info['travel_dates'][1])

    # SerpAPI Google Maps search
    params = {
        "engine": "google_maps",
        "q": preferences,
        "ll": "@40.7455096,-74.0083012,14z",  # Example coordinates for New York
        "type": "search",
        "api_key": SERPAPI_API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    local_results = results.get("local_results", [])

    locations = [Location(**result) for result in local_results]

    # Generate itinerary using AI
    location_descriptions = "\n".join([f"{loc.name}: {loc.description}" for loc in locations])
    prompt = PromptTemplate(
        template="Generate a detailed travel itinerary for a group trip from {start_date} to {end_date} based on these preferences: {preferences}. Consider the following places:\n{locations}\nProvide a day-by-day itinerary.",
        input_variables=["start_date", "end_date", "preferences", "locations"]
    )

    client = openai.OpenAI(
        api_key="d546f9f2469f46799b08a638d01fbd98",
        base_url="https://api.aimlapi.com/",
    )

    chat_completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {"role": "system", "content": "You are a travel planner creating detailed itineraries."},
            {"role": "user", "content": prompt.format(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                preferences=preferences,
                locations=location_descriptions
            )},
        ],
        temperature=0.7,
        max_tokens=512,
    )

    itinerary = chat_completion.choices[0].message.content.split('\n')

    travel_plan = TravelPlan(locations=locations, itinerary=itinerary)
    suggestions = Suggestions(travel_plan=travel_plan)

    return jsonify(suggestions.dict())



@app.route('/api/connect-calendar', methods=['POST'])
def connect_calendar():
    try:
        creds = authenticate_google_calendar()
        if creds:
            return jsonify({"message": "Calendar connected successfully."}), 200
        else:
            return jsonify({"error": "Failed to connect calendar."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/api/calendar/providers', methods=['GET'])
def get_calendar_providers():
    pass


@app.route('/api/groups/create', methods=['POST'])
def create_group():
    data = request.json
    travel_dates = data['travelDates'].split('/')
    new_group = Group(data['name'], data['description'], travel_dates)
    return jsonify({"groupId": new_group.id, "message": "Group created successfully."}), 201

@app.route('/api/groups/join', methods=['POST'])
def join_group():
    data = request.json
    group_id = data['groupId']
    invitation_code = data['invitationCode']
    
    if group_id == invitation_code and verify_group_exists(group_id):
        group = Group.get_group_by_id(group_id)
        user = User(data['name'], data['email'])
        group.add_member(user)
        return jsonify({"message": "Joined group successfully."}), 200
    else:
        return jsonify({"error": "Invalid group ID or invitation code."}), 400


@app.route('/api/groups/<group_id>/add-member', methods=['POST'])
def add_member_to_group(group_id):
    data = request.json
    group = Group.get_group_by_id(group_id)  # Assume this method exists
    user = User(data['name'], data['email'])
    group.add_member(user)
    return jsonify({"message": f"User {user.name} added to the group."}), 200

@app.route('/api/groups/<group_id>/remove-member', methods=['POST'])
def remove_member_from_group(group_id):
    data = request.json
    group = Group.get_group_by_id(group_id)  # Assume this method exists
    user = User(data['name'], data['email'])
    group.remove_member(user)
    return jsonify({"message": f"User {user.name} removed from the group."}), 200

@app.route('/api/groups/<group_id>/info', methods=['GET'])
def get_group_info(group_id):
    try:
        group = Group.get_group_by_id(group_id)
        return jsonify(group.get_group_info()), 200
    except ValueError:
        return jsonify({"error": "Group not found"}), 404


@app.route('/api/groups/<group_id>/free-slots', methods=['GET'])
def get_free_slots(group_id):
    group = Group.get_group_by_id(group_id)  # Assume this method exists
    min_duration = int(request.args.get('min_duration', 30))
    free_slots = group.find_free_slots(min_duration)
    return jsonify({"free_slots": [{"start": slot[0].isoformat(), "end": slot[1].isoformat()} for slot in free_slots]}), 200

@app.route('/api/groups/<group_id>/add-activity', methods=['POST'])
def add_group_activity(group_id):
    data = request.json
    group = Group.get_group_by_id(group_id)  # Assume this method exists
    start_time = datetime.fromisoformat(data['start_time'])
    end_time = datetime.fromisoformat(data['end_time'])
    group.add_group_activity(data['activity_name'], start_time, end_time)
    return jsonify({"message": "Group activity added successfully."}), 200

@app.route('/api/calendar/events', methods=['GET'])
def get_events():
    group_id = request.args.get('groupId')
    group = Group.get_group_by_id(group_id)
    if group:
        events = get_calendar_events(group.service, group.id)
        return jsonify({"events": events}), 200
    else:
        return jsonify({"error": "Group not found"}), 404
    
from datetime import datetime, timezone

@app.route('/api/calendar/availability', methods=['GET'])
def get_calendar_availability():
    group_id = request.args.get('groupId')
    date_range = request.args.get('dateRange')
    
    group = Group.get_group_by_id(group_id)
    start_date, end_date = date_range.split('/')
    
    free_slots = group.find_free_slots()
    suggested_time = (
        datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc),
        datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    )
    
    is_available = any(start <= suggested_time[0] and suggested_time[1] <= end for start, end in free_slots)
    
    return jsonify({
        "isAvailable": is_available,
        "suggestedTime": {
            "start": suggested_time[0].isoformat(),
            "end": suggested_time[1].isoformat()
        }
    }), 200



if __name__ == '__main__':
    app.run(debug=True)
