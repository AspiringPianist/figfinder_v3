import json
import os

DATABASE_FILE = 'groups_database.json'

def load_database():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_database(data):
    with open(DATABASE_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def get_group(group_id):
    database = load_database()
    return database.get(group_id)

def save_group(group):
    database = load_database()
    database[group.id] = {
        'name': group.name,
        'description': group.description,
        'travel_dates': [date.isoformat() for date in group.travel_dates],
        'members': [member.name for member in group.members]
    }
    save_database(database)

def delete_group(group_id):
    database = load_database()
    if group_id in database:
        del database[group_id]
        save_database(database)
        return True
    return False
