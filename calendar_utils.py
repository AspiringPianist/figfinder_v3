# calendar/utils.py
import os.path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from googleapiclient.errors import HttpError
from datetime import datetime, timezone


SCOPES = ['https://www.googleapis.com/auth/calendar']

def authenticate_google_calendar(token_file='token.json'):
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
    return creds



def get_calendar_events(service, calendar_id, time_min=None, time_max=None, max_results=None):
    events_result = service.events().list(
        calendarId=calendar_id,
        timeMin=time_min.isoformat() + 'Z' if time_min else None,
        timeMax=time_max.isoformat() + 'Z' if time_max else None,
        maxResults=max_results,
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    events = events_result.get('items', [])

    formatted_events = []
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        end = event['end'].get('dateTime', event['end'].get('date'))
        formatted_events.append({
            'summary': event['summary'],
            'start': start,
            'end': end,
            'id': event['id']
        })

    return formatted_events


def get_upcoming_events(service):
    now = datetime.now().isoformat() + 'Z'  # 'Z' indicates UTC time
    events_result = service.events().list(calendarId='primary', timeMin=now,
                                          maxResults=10, singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])

    event_list = []
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        end = event['end'].get('dateTime', event['end'].get('date'))
        summary = event['summary']
        event_list.append((start, end, summary))
    return event_list


def create_group_calendar(service, calendar_info):
    calendar = calendar_info
    try:
        created_calendar = service.calendars().insert(body=calendar).execute()
        return created_calendar['id']
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

def add_event_to_calendar(service, calendar_id, event):
    try:
        event = service.events().insert(calendarId=calendar_id, body=event).execute()
        return event['id']
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

def clear_calendar(service, calendar_id):
    try:
        events = service.events().list(calendarId=calendar_id).execute()
        for event in events.get('items', []):
            service.events().delete(calendarId=calendar_id, eventId=event['id']).execute()
    except HttpError as error:
        print(f'An error occurred: {error}')

       
def get_user_calendar_events():
    creds = authenticate_google_calendar()
    service = build('calendar', 'v3', credentials=creds)
    return get_upcoming_events(service)



if __name__ == '__main__':
    events = get_upcoming_events()
    for event in events:
        print(event)
