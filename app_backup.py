from flask import Flask, request, jsonify
from calendar_utils import authenticate_google_calendar, get_upcoming_events, create_group_calendar, add_event_to_calendar, clear_calendar, get_calendar_events
from group_utils import Group, User
from googleapiclient.discovery import build
from datetime import datetime
from flask import abort
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # This enables CORS for all routes


@app.route('/api/auth/connect-calendar', methods=['POST'])
def connect_calendar():
    creds = authenticate_google_calendar()
    if creds:
        return jsonify({"message": "Calendar connected successfully."}), 200
    else:
        return jsonify({"error": "Failed to connect calendar."}), 400

@app.route('/api/groups/create', methods=['POST'])
def create_group():
    data = request.json
    new_group = Group(data['name'], data['description'], data['travel_dates'])
    return jsonify({"groupId": new_group.id, "message": "Group created successfully."}), 201


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


if __name__ == '__main__':
    app.run(debug=True)
