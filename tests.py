import unittest
import requests
import json
from datetime import datetime, timedelta

BASE_URL = 'http://127.0.0.1:5000'  # Adjust if your app runs on a different port

class TestGroupTravelApp(unittest.TestCase):

    def test_connect_calendar(self):
        response = requests.post(f'{BASE_URL}/api/auth/connect-calendar')
        self.assertEqual(response.status_code, 200)
        self.assertIn('message', response.json())

    def test_create_group(self):
        data = {
            'name': 'Test Group',
            'description': 'A test group',
            'travel_dates': ['2024-01-01', '2024-01-07']
        }
        response = requests.post(f'{BASE_URL}/api/groups/create', json=data)
        self.assertEqual(response.status_code, 201)
        self.assertIn('groupId', response.json())

    def test_get_group_info(self):
        group_id = 'dummy_id'  # Use the dummy_id we set up in get_group_by_id
        response = requests.get(f'{BASE_URL}/api/groups/{group_id}/info')
        self.assertEqual(response.status_code, 200)
        self.assertIn('name', response.json())

    def test_add_member_to_group(self):
        group_id = 'dummy_id'
        data = {
            'name': 'Test User',
            'email': 'test@example.com'
        }
        response = requests.post(f'{BASE_URL}/api/groups/{group_id}/add-member', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('message', response.json())

    def test_remove_member_from_group(self):
        group_id = 'dummy_id'
        data = {
            'name': 'Test User',
            'email': 'test@example.com'
        }
        response = requests.post(f'{BASE_URL}/api/groups/{group_id}/remove-member', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('message', response.json())

    def test_get_free_slots(self):
        group_id = 'dummy_id'
        response = requests.get(f'{BASE_URL}/api/groups/{group_id}/free-slots')
        self.assertEqual(response.status_code, 200)
        self.assertIn('free_slots', response.json())

    def test_add_group_activity(self):
        group_id = 'dummy_id'
        data = {
            'activity_name': 'Test Activity',
            'start_time': (datetime.now() + timedelta(days=1)).isoformat(),
            'end_time': (datetime.now() + timedelta(days=1, hours=2)).isoformat()
        }
        response = requests.post(f'{BASE_URL}/api/groups/{group_id}/add-activity', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('message', response.json())

    def test_get_events(self):
        response = requests.get(f'{BASE_URL}/api/calendar/events', params={'groupId': 'dummy_id'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('events', response.json())

if __name__ == '__main__':
    unittest.main()
