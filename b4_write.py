import requests
import time
import sys
import json

class ApiResponseError(Exception):
    """Custom exception for API response errors."""
    pass

def move(moveit_url,x, y, z, qx, qy, qz, qw):

    params = {
        "x": x,
        "y": y,
        "z": z,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qw": qw,
    }

    max_retries = 5  # 最大重试次数
    retry_delay = 0  # 重试间隔时间（秒）

    for attempt in range(max_retries):
        try:
            # print(f"Attempt {attempt + 1}/{max_retries} to call {moveit_url}") # Optional debug line
            resp = requests.get(moveit_url, params=params, timeout=10)
            resp.raise_for_status()  # Raises HTTPError for 4xx/5xx responses

            try:
                response_data = resp.json()
            # Using requests.exceptions.JSONDecodeError which is raised by resp.json()
            except requests.exceptions.JSONDecodeError as json_exc:
                error_message = f"Failed to decode JSON response. Content: {resp.text[:200]}" # Truncate for brevity
                raise ApiResponseError(error_message) from json_exc

            api_status = response_data.get("status")
            if api_status == "success":
                print("Status:", resp.status_code)
                print("Response body:")
                # Using json.dumps for potentially better formatting and handling non-ASCII
                print(json.dumps(response_data, indent=2, ensure_ascii=False))
                return True  # Request successful and API status is success
            else:
                # API status is not "success" or status key is missing
                error_message = f"API status was '{api_status}'. Expected 'success'. Response: {json.dumps(response_data, indent=2, ensure_ascii=False)}"
                raise ApiResponseError(error_message)

        except (requests.exceptions.RequestException, ApiResponseError) as exc:
            print(f"[ERROR] Attempt {attempt + 1}/{max_retries} failed: {type(exc).__name__} - {str(exc)}")
            if attempt < max_retries - 1:
                if retry_delay > 0:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                elif retry_delay == 0:
                    print("Retrying immediately...")
            else:
                print(f"[ERROR] All retries failed after {max_retries} attempts. Giving up.", file=sys.stderr)
                return False # Return False from the function on ultimate failure