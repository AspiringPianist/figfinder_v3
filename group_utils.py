from datetime import datetime, timedelta, timezone
from dateutil.parser import parse
from calendar_utils import get_upcoming_events
from calendar_utils import create_group_calendar, add_event_to_calendar, clear_calendar, authenticate_google_calendar
from googleapiclient.discovery import build
import json
from googleapiclient.errors import HttpError
from database import get_group, save_group, delete_group
from datetime import datetime, timezone

class Group:
    def __init__(self, name, description, travel_dates):
        self.name = name
        self.description = description
        self.travel_dates = self.parse_travel_dates(travel_dates)
        self.service = build('calendar', 'v3', credentials=authenticate_google_calendar())
        self.members = []
        self.id = self.create_unified_calendar()
        save_group(self)


    def parse_travel_dates(self, travel_dates):
        if isinstance(travel_dates[0], str):
            return (
                datetime.strptime(travel_dates[0], '%Y-%m-%d').replace(tzinfo=timezone.utc),
                datetime.strptime(travel_dates[1], '%Y-%m-%d').replace(tzinfo=timezone.utc)
            )
        elif isinstance(travel_dates[0], datetime):
            return (
                travel_dates[0].replace(tzinfo=timezone.utc),
                travel_dates[1].replace(tzinfo=timezone.utc)
            )
        else:
            raise ValueError("Invalid travel_dates format. Expected str or datetime.datetime.")
        
    def create_unified_calendar(self):
        calendar_info = {
            'summary': f'{self.name} Unified Calendar',
            'description': json.dumps({
                'group_description': self.description,
                'travel_dates': [date.isoformat() for date in self.travel_dates]
            }),
            'timeZone': 'UTC'
        }
        return create_group_calendar(self.service, calendar_info)

    @classmethod
    def get_group_by_id(cls, group_id):
        group_data = get_group(group_id)
        if group_data:
            group = cls.__new__(cls)
            group.id = group_id
            group.name = group_data['name']
            group.description = group_data['description']
            group.travel_dates = [datetime.fromisoformat(date) for date in group_data['travel_dates']]
            group.members = [User(name, '') for name in group_data['members']]  # Assuming User class exists
            group.service = build('calendar', 'v3', credentials=authenticate_google_calendar())
            return group
        raise ValueError("Group not found")

        
        
    def update_unified_calendar(self):
        if not self.id:
            print("Unified calendar hasn't been created yet.")
            return

        clear_calendar(self.service, self.id)

        for member in self.members:
            for event in member.calendar_events:
                new_event = {
                    'summary': f"{member.name}: {event[2]}",
                    'start': {'dateTime': event[0]},
                    'end': {'dateTime': event[1]},
                    'transparency': 'opaque'  # This makes the event show as "busy"
                }
                add_event_to_calendar(self.service, self.id, new_event)

        # Add a "free" event for the entire travel period
        free_event = {
            'summary': 'Group Free Time',
            'start': {'dateTime': self.travel_dates[0].isoformat()},
            'end': {'dateTime': self.travel_dates[1].isoformat()},
            'transparency': 'transparent'  # This makes the event show as "free"
        }
        add_event_to_calendar(self.service, self.id, free_event)
        save_group(self)


    def add_member(self, user):
        if self.id:
            self.update_unified_calendar()
        if user not in self.members:
            self.members.append(user)
            print(f"User {user.name} added to the group {self.name}.")
        else:
            print(f"User {user.name} is already a member of the group {self.name}.")
        save_group(self)


    def remove_member(self, user):
        if user in self.members:
            self.members.remove(user)
            if self.id:
                self.update_unified_calendar()
            print(f"User {user.name} removed from the group {self.name}.")
        save_group(self)

    def get_group_info(self):
        return {
            "name": self.name,
            "description": self.description,
            "travel_dates": self.travel_dates,
            "members": [member.name for member in self.members]
        }

    def find_free_slots(self, min_duration=30):
        busy_times = []
        for member in self.members:
            busy_times.extend([(parse(event[0]), parse(event[1])) 
                            for event in member.calendar_events])

        busy_times.sort(key=lambda x: x[0])

        free_slots = []
        min_duration = timedelta(minutes=min_duration)

        current_time = self.travel_dates[0]
        end_time = self.travel_dates[1]

        while current_time < end_time:
            day_end = current_time.replace(hour=23, minute=59, second=59)
            day_busy_times = [t for t in busy_times if t[0].date() == current_time.date()]

            if not day_busy_times:
                if day_end - current_time >= min_duration:
                    free_slots.append((current_time, day_end))
            else:
                if day_busy_times[0][0] - current_time >= min_duration:
                    free_slots.append((current_time, day_busy_times[0][0]))

                for i in range(len(day_busy_times) - 1):
                    gap_start = day_busy_times[i][1]
                    gap_end = day_busy_times[i+1][0]
                    if gap_end - gap_start >= min_duration:
                        free_slots.append((gap_start, gap_end))

                if day_end - day_busy_times[-1][1] >= min_duration:
                    free_slots.append((day_busy_times[-1][1], day_end))

            current_time = day_end + timedelta(seconds=1)

        # Merge continuous free slots
        merged_free_slots = []
        for slot in free_slots:
            if not merged_free_slots or slot[0] - merged_free_slots[-1][1] > timedelta(seconds=1):
                merged_free_slots.append(slot)
            else:
                merged_free_slots[-1] = (merged_free_slots[-1][0], max(merged_free_slots[-1][1], slot[1]))

        return merged_free_slots
    
    def add_group_activity(self, activity_name, start_time, end_time):
        if not self.id:
            print("Unified calendar hasn't been created yet.")
            return

        event = {
            'summary': f"Group Activity: {activity_name}",
            'start': {'dateTime': start_time.isoformat(), 'timeZone': 'UTC'},
            'end': {'dateTime': end_time.isoformat(), 'timeZone': 'UTC'},
            'transparency': 'opaque'
        }

        event_id = add_event_to_calendar(self.service, self.id, event)
        if event_id:
            print(f"Group activity '{activity_name}' added to the calendar.")
        else:
            print("Failed to add group activity to the calendar.")

        save_group(self)
        
    def choose_and_add_activity(self):
        free_slots = self.find_free_slots()
        if not free_slots:
            print("No free slots available.")
            return

        print("Available free slots:")
        for i, slot in enumerate(free_slots):
            print(f"{i+1}. From {slot[0]} to {slot[1]}")

        choice = int(input("Choose a slot number: ")) - 1
        if 0 <= choice < len(free_slots):
            activity_name = input("Enter the group activity name: ")
            self.add_group_activity(activity_name, free_slots[choice][0], free_slots[choice][1])
        else:
            print("Invalid choice.")


class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.token_file = f"token_{name.lower()}.json"
        self.calendar_events = self.fetch_calendar_events()

    def authenticate(self):
        return authenticate_google_calendar(self.token_file)

    def fetch_calendar_events(self):
        creds = self.authenticate()
        service = build('calendar', 'v3', credentials=creds)
        return get_upcoming_events(service)


# Example usage
if __name__ == '__main__':
    alice = User("Alice", "unnathch@gmail.com")

    japan_trip = Group("Japan Trip", "A week-long trip to Japan", ("2024-8-23", "2024-8-30"))

    japan_trip.add_member(alice)
    
    japan_trip.create_unified_calendar()
    japan_trip.choose_and_add_activity()
    print(japan_trip.get_group_info())
    print(f"Unified Calendar ID: {japan_trip.calendar_id}")
