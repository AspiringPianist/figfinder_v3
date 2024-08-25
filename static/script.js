const BASE_URL = "http://127.0.0.1:5000";

async function makeRequest(url, method, body = null) {
  const options = {
    method: method,
    headers: {
      "Content-Type": "application/json",
    },
  };
  if (body) {
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  return await response.json();
}

async function connectCalendar() {
  const result = await makeRequest(`${BASE_URL}/api/connect-calendar`, "POST");
  document.getElementById("connectCalendarResult").innerText = JSON.stringify(
    result,
    null,
    2
  );
}

async function getCalendarProviders() {
  const result = await makeRequest(`${BASE_URL}/api/calendar/providers`, "GET");
  document.getElementById("getProvidersResult").innerText = JSON.stringify(
    result,
    null,
    2
  );
}

async function createGroup() {
  const name = document.getElementById("groupName").value;
  const description = document.getElementById("groupDescription").value;
  const startDate = document.getElementById("startDate").value;
  const endDate = document.getElementById("endDate").value;
  const result = await makeRequest(`${BASE_URL}/api/groups/create`, "POST", {
    name: name,
    description: description,
    travelDates: `${startDate}/${endDate}`,
  });
  document.getElementById("createGroupResult").innerText = JSON.stringify(
    result,
    null,
    2
  );
}

async function joinGroup() {
  const groupId = document.getElementById("joinGroupId").value;
  const invitationCode = document.getElementById("invitationCode").value;
  const name = document.getElementById("joinUserName").value;
  const email = document.getElementById("joinUserEmail").value;
  const result = await makeRequest(`${BASE_URL}/api/groups/join`, "POST", {
    groupId: groupId,
    invitationCode: invitationCode,
    name: name,
    email: email,
  });
  document.getElementById("joinGroupResult").innerText = JSON.stringify(
    result,
    null,
    2
  );
}

async function getCalendarEvents() {
  const groupId = document.getElementById("eventsGroupId").value;
  const result = await makeRequest(
    `${BASE_URL}/api/calendar/events?groupId=${groupId}`,
    "GET"
  );
  document.getElementById("getEventsResult").innerText = JSON.stringify(
    result,
    null,
    2
  );
}

async function checkCalendarAvailability() {
  const groupId = document.getElementById("availabilityGroupId").value;
  const startDate = document.getElementById("availabilityStartDate").value;
  const endDate = document.getElementById("availabilityEndDate").value;
  const result = await makeRequest(
    `${BASE_URL}/api/calendar/availability?groupId=${groupId}&dateRange=${startDate}/${endDate}`,
    "GET"
  );
  document.getElementById("checkAvailabilityResult").innerText = JSON.stringify(
    result,
    null,
    2
  );
}

async function analyzeAvailability() {
  const groupId = document.getElementById("analyzeGroupId").value;
  const preferences = {
    preference1: document.getElementById("analyzePreference1").value,
    preference2: document.getElementById("analyzePreference2").value,
    preference3: document.getElementById("analyzePreference3").value,
  };
  const result = await makeRequest(`${BASE_URL}/api/schedule/analyze`, "POST", {
    groupId: groupId,
    preferences: preferences,
  });
  document.getElementById("analyzeResult").innerText = JSON.stringify(
    result,
    null,
    2
  );
}

async function generateSuggestions() {
  const groupId = document.getElementById("suggestGroupId").value;
  const preferences = {
    preference1: document.getElementById("suggestPreference1").value,
    preference2: document.getElementById("suggestPreference2").value,
    preference3: document.getElementById("suggestPreference3").value,
  };
  const result = await makeRequest(`${BASE_URL}/api/schedule/suggest`, "POST", {
    groupId: groupId,
    preferences: preferences,
  });
  document.getElementById("suggestResult").innerText = JSON.stringify(
    result,
    null,
    2
  );
}
