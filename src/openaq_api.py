import logging
import os
import time
import requests

BASE_URL = "https://api.openaq.org/v3"
PAGE_LIMIT = 1000 # Maximum rows the API returns per page
REQUEST_DELAY = 1.1 # Seconds between requests, staying under the 60 requests per minute
REQUEST_TIMEOUT = 90 # Seconds to wait for a response
MAX_ATTEMPTS = 5 # Attempts per page before giving up
RETRY_DELAY = 15 # Seconds added to the wait after each failed attempt

# Temporary errors that are worth trying again
RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504}
RETRY_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.JSONDecodeError,
)

logger = logging.getLogger(__name__)

# Creates a session that sends the API key with every request
def create_session():
    session = requests.Session()
    session.headers.update({
        "X-API-Key": os.environ["OPENAQ_API_KEY"],
        "User-Agent": "AQ Forecaster",
    })

    return session

# Retrieves one page of hourly data, retrying if a temporary error occurs
def get_page(session, sensor_id, datetime_from, datetime_to, page):
    url = f"{BASE_URL}/sensors/{sensor_id}/hours"
    params = {
        "datetime_from": datetime_from,
        "datetime_to": datetime_to,
        "limit": PAGE_LIMIT,
        "page": page,
    }

    for attempt in range(1, MAX_ATTEMPTS + 1):
        time.sleep(REQUEST_DELAY) # Avoid rate limiting

        try:
            response = session.get(url=url, params=params, timeout=REQUEST_TIMEOUT)

            # Errors such as an invalid API key will not fix themselves, so stop right away
            if response.status_code not in RETRY_STATUS_CODES:
                response.raise_for_status()
                return response.json()["results"]

            problem = f"HTTP {response.status_code}"
            error = requests.exceptions.HTTPError(problem, response=response)

        except RETRY_EXCEPTIONS as exception:
            problem = type(exception).__name__
            error = exception

        logger.warning(f"Sensor {sensor_id}, {datetime_from} to {datetime_to}, page {page}: {problem} on attempt {attempt}/{MAX_ATTEMPTS}")

        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_DELAY * attempt)

    raise error

# Retrieves every hourly row for a sensor between two UTC times written as "YYYY-MM-DDTHH:MM:SSZ"
def get_sensor_range(session, sensor_id, datetime_from, datetime_to):
    rows = []
    page = 1

    # A page with fewer rows than the limit is the last one
    while True:
        page_rows = get_page(session, sensor_id, datetime_from, datetime_to, page)
        rows.extend(page_rows)

        if len(page_rows) < PAGE_LIMIT:
            return rows

        page += 1

# Retrieves every hourly row for a sensor in the given year
def get_sensor_year(session, sensor_id, year):
    return get_sensor_range(session, sensor_id, f"{year}-01-01T00:00:00Z", f"{year + 1}-01-01T00:00:00Z")