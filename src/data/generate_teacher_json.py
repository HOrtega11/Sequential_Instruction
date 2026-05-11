
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import APITimeoutError

from prompts.teacher_generation import build_teacher_messages
from src.utils.client import get_client
from src.utils.config import get_config





def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def try_parse_json(text: str) -> Tuple[bool, Any]:
    try:
        return True, json.loads(text)
    except Exception:
        return False, None


def validate_teacher_schema(task_type: str, obj: Any) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if task_type == "json_repair":
        if isinstance(obj, (dict, list)):
            return True, []
        return False, ["json_repair_top_level_must_be_object_or_array"]

    if not isinstance(obj, dict):
        return False, ["top_level_not_object"]

    if task_type == "json_extraction":
        required = {
            "person_name": str,
            "date": str,
            "city": str,
            "event": str,
        }
    elif task_type == "schema_generation":
        required = {
            "product_name": str,
            "price": (int, float),
            "in_stock": bool,
            "tags": list,
        }
    elif task_type == "json_classification":
        required = {
            "label": str,
            "rationale": str,
        }
    elif task_type == "tool_call_arguments":
        required = {
            "origin": str,
            "destination": str,
            "date": str,
            "passengers": int,
        }
    else:
        return False, [f"unknown_task_type:{task_type}"]

    obj_keys = set(obj.keys())
    req_keys = set(required.keys())

    for key in req_keys - obj_keys:
        errors.append(f"missing_key:{key}")
    for key in obj_keys - req_keys:
        errors.append(f"extra_key:{key}")

    for key, expected_type in required.items():
        if key not in obj:
            continue
        if not isinstance(obj[key], expected_type):
            errors.append(f"wrong_type:{key}")

    if task_type == "json_classification" and "label" in obj:
        if obj["label"] not in {"positive", "negative", "neutral"}:
            errors.append("invalid_label_value")

    if task_type == "schema_generation" and "tags" in obj:
        if not isinstance(obj["tags"], list) or not all(isinstance(x, str) for x in obj["tags"]):
            errors.append("wrong_type:tags_elements")

    if task_type == "tool_call_arguments" and "passengers" in obj:
        if not isinstance(obj["passengers"], int):
            errors.append("wrong_type:passengers")
        elif obj["passengers"] < 1:
            errors.append("invalid_value:passengers")

    return len(errors) == 0, errors


def build_json_extraction_examples() -> List[Dict[str, str]]:
    people = [
        "Maria Lopez", "James Carter", "Aisha Khan", "Daniel Kim", "Sofia Martinez",
        "Ethan Walker", "Priya Patel", "Lucas Brown", "Olivia Davis", "Noah Wilson",
        "Mia Hernandez", "Liam Thompson", "Ava Robinson", "Elijah Clark", "Emma Lewis",
        "Benjamin Hall", "Charlotte Allen", "Henry Young", "Amelia King", "Jack Wright",
        "Harper Scott", "Michael Green", "Evelyn Baker", "Alexander Adams", "Abigail Nelson",
        "Samuel Hill", "Ella Ramirez", "David Campbell", "Scarlett Mitchell", "Joseph Perez",
        "Grace Roberts", "Sebastian Turner", "Chloe Phillips", "Matthew Parker", "Lily Evans",
        "Carter Edwards", "Aria Collins", "Julian Stewart", "Nora Sanchez", "Wyatt Morris",
        "Zoe Rogers", "Isaac Reed", "Hannah Cook", "Gabriel Morgan", "Layla Bell",
        "Owen Murphy", "Victoria Bailey", "Nathan Rivera", "Penelope Cooper", "Leo Richardson",
        "Audrey Cox", "Christopher Howard", "Stella Ward", "Andrew Torres", "Claire Peterson",
        "Ryan Gray", "Lucy Ramirez", "Thomas James", "Anna Watson", "Dylan Brooks",
        "Madison Kelly", "Isaiah Sanders", "Brooklyn Price", "John Bennett", "Natalie Wood",
        "Levi Barnes", "Savannah Ross", "Adam Henderson", "Paisley Coleman", "Connor Jenkins",
        "Bella Perry", "Jordan Powell", "Violet Long", "Robert Patterson", "Alice Hughes",
        "Aaron Flores", "Ruby Washington", "Charles Butler", "Naomi Simmons", "Jonathan Foster",
    ]

    events = [
        "cardiology conference", "oncology symposium", "machine learning workshop",
        "public health seminar", "robotics expo", "emergency medicine summit",
        "bioinformatics forum", "surgical skills lab", "data science meetup",
        "neuroscience colloquium", "cloud computing conference", "education panel",
        "cybersecurity roundtable", "startup pitch night", "pediatrics lecture",
        "pharmacology workshop", "ethics debate", "AI policy summit", "design showcase",
        "medical imaging session", "nursing leadership event", "finance workshop",
        "environmental health meeting", "language technology forum", "residency orientation",
        "trauma response drill", "critical care simulation", "genomics seminar",
        "business analytics talk", "research poster session", "innovation challenge",
        "telemedicine panel", "supply chain conference", "sports medicine lecture",
        "campus leadership retreat", "legal medicine discussion", "health informatics meetup",
        "grant writing workshop", "infectious disease briefing", "wellness fair",
        "quality improvement session", "hospital operations review", "deep learning tutorial",
        "graduate recruitment event", "cardiac imaging workshop", "clinical reasoning conference",
        "rural health summit", "epidemiology update", "product launch event", "developer conference",
        "materials science seminar", "virtual reality demo", "disaster preparedness briefing",
        "medical education workshop", "artificial intelligence symposium", "cancer research panel",
        "resident teaching session", "national policy forum", "precision medicine discussion",
        "informatics bootcamp", "global health conference", "simulation debriefing",
        "interprofessional case review", "occupational therapy showcase", "safety committee meeting",
        "vaccination campaign launch", "leadership development seminar", "consumer technology expo",
        "clinical documentation workshop", "biostatistics lecture", "research integrity panel",
        "transplant medicine update", "robot-assisted surgery session", "palliative care forum",
        "stroke care workshop", "critical thinking seminar", "employer networking night",
        "health equity roundtable", "immunology conference", "student research colloquium",
    ]

    cities = [
        "Houston", "Chicago", "San Antonio", "Dallas", "Austin", "Phoenix", "Seattle",
        "Boston", "Denver", "Atlanta", "Miami", "Nashville", "Philadelphia", "Detroit",
        "Portland", "Los Angeles", "San Diego", "New York", "Orlando", "Minneapolis",
        "Salt Lake City", "Charlotte", "Raleigh", "Indianapolis", "Kansas City", "New Orleans",
        "Cleveland", "Pittsburgh", "Columbus", "Sacramento", "St. Louis", "Tampa",
        "Baltimore", "Milwaukee", "Omaha", "Albuquerque", "Las Vegas", "San Jose",
        "Cincinnati", "Madison", "Birmingham", "Memphis", "Louisville", "Richmond",
        "Buffalo", "Tucson", "Boise", "El Paso", "Oklahoma City", "Tulsa",
        "Anchorage", "Fresno", "Reno", "Lubbock", "Waco", "Irving",
        "Arlington", "Plano", "Frisco", "Corpus Christi", "Chattanooga", "Knoxville",
        "Greensboro", "Spokane", "Tacoma", "Fort Worth", "Baton Rouge", "Jackson",
        "Little Rock", "Des Moines", "Grand Rapids", "Toledo", "Akron", "Dayton",
        "Savannah", "Asheville", "Santa Fe", "Modesto", "Berkeley", "Irvine",
    ]

    dates = [
        "January 8, 2025", "January 22, 2025", "February 3, 2025", "February 18, 2025",
        "March 12, 2025", "March 27, 2025", "April 5, 2025", "April 19, 2025",
        "May 2, 2025", "May 16, 2025", "June 10, 2025", "June 24, 2025",
        "July 7, 2025", "July 21, 2025", "August 4, 2025", "August 19, 2025",
        "September 1, 2025", "September 14, 2025", "October 6, 2025", "October 20, 2025",
        "November 9, 2025", "November 23, 2025", "December 11, 2025", "January 15, 2026",
        "February 11, 2026", "March 4, 2026", "April 17, 2026", "May 29, 2026",
        "June 13, 2026", "July 25, 2026", "August 8, 2026", "September 30, 2026",
        "October 14, 2026", "November 18, 2026", "December 2, 2026", "January 26, 2027",
        "February 9, 2027", "March 23, 2027", "April 12, 2027", "May 6, 2027",
        "June 28, 2027", "July 13, 2027", "August 24, 2027", "September 16, 2027",
        "October 7, 2027", "November 21, 2027", "December 5, 2027", "January 19, 2028",
        "February 14, 2028", "March 8, 2028", "April 26, 2028", "May 17, 2028",
        "June 6, 2028", "July 29, 2028", "August 15, 2028", "September 27, 2028",
        "October 10, 2028", "November 30, 2028", "December 18, 2028", "January 11, 2029",
        "February 22, 2029", "March 15, 2029", "April 3, 2029", "May 24, 2029",
        "June 18, 2029", "July 2, 2029", "August 20, 2029", "September 12, 2029",
        "October 28, 2029", "November 6, 2029", "December 19, 2029", "January 31, 2030",
        "February 12, 2030", "March 26, 2030", "April 9, 2030", "May 21, 2030",
        "June 4, 2030", "July 18, 2030", "August 30, 2030", "September 22, 2030",
    ]

    examples = []
    for person, event, city, date in zip(people, events, cities, dates):
        examples.append(
            {
                "task_type": "json_extraction",
                "instruction": (
                    "Extract the person name, date, city, and event from the text and return a valid JSON object."
                ),
                "input": f"{person} attended the {event} in {city} on {date}.",
            }
        )
    return examples


def build_schema_generation_examples() -> List[Dict[str, str]]:
    products = [
        "wireless mouse", "mechanical keyboard", "USB-C hub", "noise-canceling headphones",
        "smartwatch", "laptop stand", "tablet case", "portable charger", "desk lamp",
        "fitness tracker", "Bluetooth speaker", "external SSD", "gaming monitor",
        "webcam", "microphone", "standing desk", "office chair", "air purifier",
        "water bottle", "coffee grinder", "electric toothbrush", "sleep mask",
        "running shoes", "backpack", "travel pillow", "digital thermometer",
        "blood pressure cuff", "stethoscope", "surgical headlight", "scrub top",
        "protein shaker", "yoga mat", "resistance bands", "dumbbell set",
        "cookware set", "rice cooker", "vacuum cleaner", "humidifier",
        "smart plug", "robot vacuum", "book light", "flash drive",
        "wireless router", "graphics tablet", "drawing stylus", "camera tripod",
        "phone charger", "car mount", "bike helmet", "camping lantern",
        "thermos", "monitor arm", "ergonomic mouse", "portable fan",
        "desk organizer", "calendar planner", "whiteboard", "label maker",
        "document scanner", "mini projector", "LED strip lights", "lunch container",
        "medical clipboard", "glucose meter", "pulse oximeter", "face shield",
        "shoe rack", "coat hanger", "window blinds", "picnic blanket",
        "cooler bag", "reusable tote", "flashlight", "multitool",
        "toolbox", "extension cord", "power strip", "wireless presenter",
        "kettle", "blender", "toaster oven", "air fryer",
    ]

    examples = []
    for product in products:
        examples.append(
            {
                "task_type": "schema_generation",
                "instruction": (
                    "Create a valid JSON object with keys: "
                    "product_name (string), price (number), in_stock (boolean), tags (array of strings)."
                ),
                "input": f"Generate an example product record for a {product}.",
            }
        )
    return examples


def build_classification_examples() -> List[Dict[str, str]]:
    positive_texts = [
        "The service was fast and the staff was helpful.",
        "The product arrived early and worked perfectly.",
        "I loved the clear instructions and smooth setup.",
        "The meal was delicious and the portions were generous.",
        "Her presentation was engaging and easy to follow.",
        "The clinic was clean and the nurse was kind.",
        "The software update fixed the issue immediately.",
        "Shipping was quick and the packaging was excellent.",
        "The professor explained the topic very clearly.",
        "Customer support resolved my problem in minutes.",
        "The event was well organized and enjoyable.",
        "This keyboard feels great and types smoothly.",
        "The apartment was spacious and surprisingly quiet.",
        "The workshop was informative and practical.",
        "I was impressed by the battery life.",
        "The doctor listened carefully and answered all my questions.",
        "The app interface is intuitive and responsive.",
        "Check-in was easy and the hotel room was spotless.",
        "The training session was concise and useful.",
        "The book was thoughtful and beautifully written.",
        "The consultant gave excellent advice.",
        "The resident gave a confident and organized presentation.",
        "The headphones sound amazing and fit comfortably.",
        "The new workflow saved us a lot of time.",
        "The image quality is crisp and vibrant.",
        "The commute was easy and the location is ideal.",
        "The tool made the task much easier.",
    ]

    negative_texts = [
        "The order arrived damaged and late.",
        "The instructions were confusing and incomplete.",
        "The room was dirty and smelled bad.",
        "Customer support never answered my emails.",
        "The app crashes every time I open it.",
        "The device stopped working after one day.",
        "The food was cold and bland.",
        "The lecture was disorganized and hard to understand.",
        "The battery drained much faster than advertised.",
        "The package was missing several items.",
        "The appointment started an hour late.",
        "The website kept freezing during checkout.",
        "The chair is uncomfortable and poorly built.",
        "The audio quality is terrible and distorted.",
        "The training material was outdated and unhelpful.",
        "The product feels cheap and flimsy.",
        "The instructions contained several errors.",
        "The shipment was delayed without explanation.",
        "The interface is cluttered and frustrating to use.",
        "The hotel bed was uncomfortable and noisy.",
        "The exam review was rushed and unclear.",
        "The sample data was incomplete and inaccurate.",
        "The report was full of mistakes.",
        "The implementation is buggy and unreliable.",
        "The patient summary omitted critical details.",
        "The monitor had dead pixels right out of the box.",
        "The coffee tasted burnt and watery.",
    ]

    neutral_texts = [
        "The meeting started at 2 PM and ended at 3 PM.",
        "The package contains a charger, cable, and manual.",
        "The room has two windows and one desk.",
        "The report was submitted on Tuesday.",
        "The clinic is located on the third floor.",
        "The laptop comes in silver and black.",
        "The event will be held downtown next month.",
        "The form asks for your name and address.",
        "The article discusses changes in policy.",
        "The battery capacity is 5000 mAh.",
        "The class meets every Monday and Wednesday.",
        "The product is available in three sizes.",
        "The file was uploaded to the shared folder.",
        "The patient returned for follow-up in six weeks.",
        "The office closes at 5 PM.",
        "The system stores records in JSON format.",
        "The conference includes keynote talks and posters.",
        "The shipment left the warehouse yesterday.",
        "The software supports English and Spanish.",
        "The paper compares two forecasting methods.",
        "The user entered an invalid password twice.",
        "The train departs at 7:30 AM.",
        "The chart shows values from January to March.",
        "The building opened in 2018.",
        "The medication was taken with food.",
        "The dashboard displays recent activity.",
    ]

    examples = []

    for text in positive_texts:
        examples.append(
            {
                "task_type": "json_classification",
                "instruction": (
                    "Classify the sentiment as exactly one of: positive, negative, neutral. "
                    "Return valid JSON with keys: label, rationale."
                ),
                "input": text,
            }
        )

    for text in negative_texts:
        examples.append(
            {
                "task_type": "json_classification",
                "instruction": (
                    "Classify the sentiment as exactly one of: positive, negative, neutral. "
                    "Return valid JSON with keys: label, rationale."
                ),
                "input": text,
            }
        )

    for text in neutral_texts:
        examples.append(
            {
                "task_type": "json_classification",
                "instruction": (
                    "Classify the sentiment as exactly one of: positive, negative, neutral. "
                    "Return valid JSON with keys: label, rationale."
                ),
                "input": text,
            }
        )

    return examples


def build_json_repair_examples() -> List[Dict[str, str]]:
    broken_jsons = [
        '{"name": "John", "age": 31, "skills": ["python", "sql",}',
        '{"city": "Houston", "state": "Texas", "zip": 77001',
        '{"patient": "Alice", "heart_rate": 88, "stable": true,, "unit": "ICU"}',
        '{"course": "Biology", "credits": 4 "semester": "Fall"}',
        '{"items": ["pen", "notebook" "eraser"], "count": 3}',
        '{"flight": {"origin": "SAT", "destination": "ORD", "date": "2026-06-10",}}',
        '{"temperature": 98.6, "symptoms": ["cough", "fatigue"], "follow_up": yes}',
        '{"product_name": "Mouse", "price": 25.99, "in_stock": tru}',
        '{"book": {"title": "Dune", "author": "Frank Herbert", "year": 1965,}',
        '{"user": "maria", "roles": ["admin", "editor"], "active": true}}',
        '{"team": "Raptors", "wins": 48, "losses": 34,, "seed": 4}',
        '{"email": "test@example.com", "verified": false, "tags": ["new",]}',
        '{"x": 10, "y": 20, "label": "point"',
        '{"medications": ["aspirin", "metformin"], "allergies": ["penicillin",], "age": 54}',
        '{"device": "router", "ports": 4, "wireless": true "band": "dual"}',
        '{"schedule": {"day": "Monday", "time": "08:00"}',
        '{"title": "Report", "pages": twelve, "status": "draft"}',
        '{"origin": "Austin", "destination": "Denver" "passengers": 2}',
        '{"sensor": "A12", "value": 17.4, "units": "mg/dL",}',
        '{"movie": "Inception", "rating": 9, "genres": ["sci-fi", "thriller",}',
        '{"employee": "Karen", "department": HR, "id": 2045}',
        '{"order_id": 12345, "items": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}],}',
        '{"phone": "(210) 555-1234", "primary": True}',
        '{"window": {"width": 800, "height": 600}, "fullscreen": fals}',
        '{"recipe": "pasta", "ingredients": ["noodles", "tomato sauce"], "servings": 4,,}',
        '{"station": "North", "arrivals": ["7:00", "7:15" "7:30"]}',
        '{"ticket": "A-113", "open": true, "priority": "high"',
        '{"author": "Jane Austen", "works": ["Pride and Prejudice", "Emma"], "born": 1775,,}',
        '{"lab": "CBC", "hemoglobin": 13.2, "platelets": 250000 "units": "per_uL"}',
        '{"campaign": "Spring", "budget": 15000, "active": false, "channels": ["email", "social",]}',
        '{"address": {"street": "123 Main St", "city": "Austin", "state": "TX", "zip": 78701,}}',
        '{"playlist": ["song1", "song2",], "shuffle": true}',
        '{"diagnosis": "hypertension", "stage": 2 "meds": ["lisinopril"]}',
        '{"metrics": {"accuracy": 0.91, "f1": 0.88,}, "split": "test"}',
        '{"fruit": "apple", "count": 5, "color": red}',
        '{"invoice": 9001, "paid": false, "amount": 123.45,, "currency": "USD"}',
        '{"calendar_event": {"title": "Review", "date": "2026-07-01", "attendees": 3}',
        '{"language": "Python", "versions": [3.9, 3.10, 3.11,, 3.12]}',
        '{"score": 87, "passed": true, "comments": "good work",}',
        '{"ward": "ICU", "beds": 20 "occupied": 18}',
        '{"shape": "circle", "radius": 4.5, "area": }',
        '{"priority": "low", "resolved": false, "owner": "Sam", "tags": ["ops", "infra",}',
        '{"conference": "MedTech", "year": 2027, "location": "Boston",,}',
        '{"allergy": "peanut", "severity": severe}',
        '{"filename": "report.pdf", "size_kb": 420, "encrypted": flase}',
        '{"package": {"weight": 2.5, "unit": "kg"}, "fragile": true, "tracking": "ZX123",}',
        '{"login_count": 12, "last_seen": "2026-03-01T12:00:00" "status": "active"}',
        '{"course_code": "CS6263", "students": 28, "online": true,, "room": "A101"}',
        '{"instrument": "guitar", "strings": 6, "electric": false,}',
        '{"summary": "stable patient", "vitals": {"hr": 82, "rr": 16, "bp": "120/76",}}',
        '{"country": "Japan", "capital": "Tokyo", "population_millions": 125.1,,}',
        '{"paper": "Transformers", "citations": [1200, 1500,], "year": 2017}',
        '{"device_id": "X9", "connected": yes, "signal": "strong"}',
        '{"topic": "ethics", "session": 3, "required": true "duration_minutes": 45}',
        '{"album": "Kind of Blue", "tracks": 5, "genre": "jazz",}',
        '{"browser": "Firefox", "version": 123 "platform": "Linux"}',
        '{"inventory": [{"id": 1, "qty": 10}, {"id": 2, "qty": 4},], "warehouse": "W1"}',
        '{"teacher": "Mr. Lee", "subject": "Math", "years_experience": 12,,}',
        '{"patient_id": 8842, "disposition": "home", "followup_days": seven}',
        '{"route": {"from": "SAT", "to": "DFW"}, "duration_min": 65, "direct": true,}',
        '{"module": "alignment", "status": "complete", "tests_passed": 18 "tests_failed": 0}',
        '{"festival": "Lights", "days": ["Friday", "Saturday", "Sunday",], "outdoor": true}',
        '{"sensor_id": "T-44", "reading": 72.4, "units": "F",, "ok": true}',
        '{"policy": "refund", "days": 30, "exceptions": ["final sale" "gift cards"]}',
        '{"hospital": "Memorial", "beds_total": 300, "beds_open": 24,}',
        '{"favorite_color": "blue", "rgb": [0, 0, 255,,]}',
        '{"project": "SeqTune", "phase": "stage2", "owner": "Horus", "active": true,}',
        '{"meeting": "lab sync", "time": "10:30", "room": B201}',
        '{"dosage_mg": 5, "frequency": "daily", "prn": False}',
        '{"monitor": "Dell", "size_inches": 27 "resolution": "1440p"}',
        '{"message": "hello", "urgent": false, "recipients": ["Ana", "Ben",],}',
        '{"temperature_c": 37.0, "fever": false, "notes": "normal exam"',
        '{"semester": "Spring", "credits": 15, "honors": tru, "advisor": "Dr. Smith"}',
        '{"airport": "ORD", "gates": 191, "international": true,}',
        '{"code": "A17", "valid": false, "retries": 3,,}',
        '{"brand": "Acme", "model": "X1", "available": true "warranty_years": 2}',
        '{"article": "forecasting", "peer_reviewed": true, "citations": 42,}',
        '{"unit": "PICU", "patient_count": 14 "nurses_on_shift": 6}',
        '{"exercise": "squat", "sets": 4, "reps": [8, 8, 6, 6,], "weight_lb": 225}',
        '{"username": "alex", "login_attempts": 3, "locked": false,, "last_login": "2026-04-01"}',
    ]

    examples = []
    for broken in broken_jsons:
        examples.append(
            {
                "task_type": "json_repair",
                "instruction": "Repair the malformed JSON and return only valid JSON.",
                "input": broken,
            }
        )
    return examples


def build_tool_call_examples() -> List[Dict[str, str]]:
    routes = [
        ("San Antonio", "Chicago", "June 10, 2026", 2),
        ("Austin", "Denver", "July 14, 2026", 1),
        ("Houston", "Seattle", "August 3, 2026", 3),
        ("Dallas", "Boston", "September 21, 2026", 1),
        ("Phoenix", "Atlanta", "October 5, 2026", 2),
        ("Miami", "New York", "November 8, 2026", 1),
        ("Nashville", "Los Angeles", "December 12, 2026", 4),
        ("Portland", "San Diego", "January 9, 2027", 2),
        ("Philadelphia", "Detroit", "February 16, 2027", 1),
        ("Charlotte", "Orlando", "March 28, 2027", 2),
        ("Raleigh", "Minneapolis", "April 11, 2027", 1),
        ("Indianapolis", "Las Vegas", "May 24, 2027", 3),
        ("Kansas City", "Salt Lake City", "June 7, 2027", 2),
        ("New Orleans", "Cleveland", "July 19, 2027", 1),
        ("Pittsburgh", "Sacramento", "August 27, 2027", 2),
        ("St. Louis", "Tampa", "September 14, 2027", 1),
        ("Baltimore", "Milwaukee", "October 30, 2027", 2),
        ("Omaha", "Albuquerque", "November 17, 2027", 1),
        ("Cincinnati", "Madison", "December 3, 2027", 2),
        ("Birmingham", "Memphis", "January 22, 2028", 1),
        ("Louisville", "Richmond", "February 13, 2028", 2),
        ("Buffalo", "Tucson", "March 6, 2028", 1),
        ("Boise", "El Paso", "April 18, 2028", 3),
        ("Oklahoma City", "Tulsa", "May 29, 2028", 2),
        ("Anchorage", "Fresno", "June 16, 2028", 1),
        ("Reno", "Lubbock", "July 8, 2028", 2),
        ("Waco", "Irving", "August 25, 2028", 1),
        ("Arlington", "Plano", "September 9, 2028", 2),
        ("Frisco", "Corpus Christi", "October 20, 2028", 1),
        ("Chattanooga", "Knoxville", "November 11, 2028", 2),
        ("Greensboro", "Spokane", "December 28, 2028", 1),
        ("Tacoma", "Fort Worth", "January 17, 2029", 2),
        ("Baton Rouge", "Jackson", "February 8, 2029", 1),
        ("Little Rock", "Des Moines", "March 19, 2029", 2),
        ("Grand Rapids", "Toledo", "April 7, 2029", 1),
        ("Akron", "Dayton", "May 23, 2029", 2),
        ("Savannah", "Asheville", "June 12, 2029", 1),
        ("Santa Fe", "Modesto", "July 27, 2029", 2),
        ("Berkeley", "Irvine", "August 18, 2029", 1),
        ("Columbus", "Denver", "September 2, 2029", 2),
        ("Austin", "Chicago", "October 15, 2029", 1),
        ("Houston", "Boston", "November 4, 2029", 2),
        ("Dallas", "Seattle", "December 9, 2029", 1),
        ("Phoenix", "Miami", "January 26, 2030", 2),
        ("San Diego", "Atlanta", "February 14, 2030", 1),
        ("New York", "Portland", "March 10, 2030", 2),
        ("Orlando", "Detroit", "April 21, 2030", 1),
        ("Minneapolis", "Nashville", "May 13, 2030", 2),
        ("Salt Lake City", "Philadelphia", "June 5, 2030", 1),
        ("Charlotte", "Los Angeles", "July 16, 2030", 2),
        ("Raleigh", "San Antonio", "August 29, 2030", 1),
        ("Indianapolis", "Houston", "September 18, 2030", 2),
        ("Kansas City", "Austin", "October 8, 2030", 1),
        ("New Orleans", "Phoenix", "November 22, 2030", 2),
        ("Cleveland", "Dallas", "December 14, 2030", 1),
        ("Pittsburgh", "Miami", "January 6, 2031", 2),
        ("Sacramento", "Boston", "February 25, 2031", 1),
        ("Tampa", "Seattle", "March 17, 2031", 2),
        ("Milwaukee", "Chicago", "April 4, 2031", 1),
        ("Omaha", "Denver", "May 28, 2031", 2),
        ("Albuquerque", "Houston", "June 20, 2031", 1),
        ("Cincinnati", "Atlanta", "July 9, 2031", 2),
        ("Madison", "Portland", "August 31, 2031", 1),
        ("Birmingham", "Orlando", "September 26, 2031", 2),
        ("Memphis", "San Diego", "October 19, 2031", 1),
        ("Louisville", "Philadelphia", "November 7, 2031", 2),
        ("Richmond", "Phoenix", "December 27, 2031", 1),
        ("Buffalo", "Austin", "January 12, 2032", 2),
        ("Tucson", "Dallas", "February 23, 2032", 1),
        ("Boise", "Seattle", "March 30, 2032", 2),
        ("El Paso", "Chicago", "April 15, 2032", 1),
        ("Tulsa", "Boston", "May 6, 2032", 2),
        ("Anchorage", "Miami", "June 24, 2032", 1),
        ("Fresno", "Denver", "July 11, 2032", 2),
        ("Reno", "Houston", "August 5, 2032", 1),
        ("Plano", "San Antonio", "September 14, 2032", 2),
        ("Corpus Christi", "Atlanta", "October 1, 2032", 1),
        ("Knoxville", "New York", "November 16, 2032", 2),
        ("Spokane", "Los Angeles", "December 8, 2032", 1),
        ("San Antonio", "Boston", "January 20, 2033", 2),
    ]

    examples = []
    for origin, destination, date, passengers in routes:
        examples.append(
            {
                "task_type": "tool_call_arguments",
                "instruction": (
                    "Generate valid JSON arguments for a function call "
                    "book_flight(origin, destination, date, passengers)."
                ),
                "input": (
                    f"Book a flight from {origin} to {destination} on {date} "
                    f"for {passengers} passenger{'s' if passengers > 1 else ''}."
                ),
            }
        )
    return examples


def build_prompt_bank(train_per_task: int, eval_per_task: int) -> Dict[str, List[Dict[str, str]]]:
    config = get_config()
    RANDOM_SEED = config.get("data_generation", {}).get("random_seed", 42)
    random.seed(RANDOM_SEED)

    task_builders = {
        "json_extraction": build_json_extraction_examples,
        "schema_generation": build_schema_generation_examples,
        "json_classification": build_classification_examples,
        "json_repair": build_json_repair_examples,
        "tool_call_arguments": build_tool_call_examples,
    }

    train_examples = []
    eval_examples = []

    for task_type, builder in task_builders.items():
        examples = builder()
        random.shuffle(examples)

        needed = train_per_task + eval_per_task
        if len(examples) < needed:
            raise ValueError(
                f"Not enough examples for {task_type}. Need {needed}, found {len(examples)}."
            )

        eval_examples.extend(examples[:eval_per_task])
        train_examples.extend(examples[eval_per_task : eval_per_task + train_per_task])

    random.shuffle(train_examples)
    random.shuffle(eval_examples)

    return {
        "train": train_examples,
        "eval": eval_examples,
    }


def generate_teacher_output(
    client,
    model_name: str,
    example: Dict[str, str],
    temperature: float,
    max_tokens: int,
    max_attempts: int = 5,
) -> str:
    last_output = None
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=build_teacher_messages(example),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content.strip()
            last_output = output

            parsed_ok, parsed_obj = try_parse_json(output)
            if not parsed_ok:
                print(
                    f"Invalid JSON on attempt {attempt}/{max_attempts} "
                    f"for task {example['task_type']}"
                )
                continue

            schema_ok, schema_errors = validate_teacher_schema(example["task_type"], parsed_obj)
            if schema_ok:
                return output

            print(
                f"Schema-invalid JSON on attempt {attempt}/{max_attempts} "
                f"for task {example['task_type']}: {schema_errors}"
            )

        except APITimeoutError as e:
            last_error = e
            print(
                f"Timeout on attempt {attempt}/{max_attempts} "
                f"for task {example['task_type']}"
            )
            time.sleep(2 * attempt)

    if last_output is not None:
        raise ValueError(
            f"Failed to get valid schema-compliant JSON after {max_attempts} attempts "
            f"for task {example['task_type']}. Last output:\n{last_output}"
        )

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        f"Failed to generate valid JSON for task {example['task_type']} after {max_attempts} attempts."
    )


def generate_dataset_split(
    client,
    model_name: str,
    examples: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    split_name: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    accepted: List[Dict[str, str]] = []
    failures: List[Dict[str, Any]] = []

    for idx, example in enumerate(examples, start=1):
        print(f"[{split_name}] Generating {idx}/{len(examples)} for task {example['task_type']}")

        try:
            output = generate_teacher_output(
                client=client,
                model_name=model_name,
                example=example,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            accepted.append(
                {
                    "id": f"{split_name}_{example['task_type']}_{idx}",
                    "instruction": example["instruction"],
                    "input": example["input"],
                    "output": output,
                    "task_type": example["task_type"],
                }
            )

        except Exception as e:
            failures.append(
                {
                    "id": f"{split_name}_{example['task_type']}_{idx}",
                    "instruction": example["instruction"],
                    "input": example["input"],
                    "task_type": example["task_type"],
                    "error": str(e),
                }
            )
            print(
                f"Discarding example after repeated failure "
                f"for task {example['task_type']}: {e}"
            )

    return accepted, failures


def main() -> None:
    config = get_config()
    RANDOM_SEED = config.get("data_generation", {}).get("random_seed", 42)

    client = get_client()
    teacher_model = config["models"]["teacher"]
    temperature = config["generation"]["temperature"]
    max_tokens = config["generation"]["max_tokens"]

    output_path = config["paths"]["json_train"]
    eval_path = config["paths"]["json_eval"]

    data_cfg = config.get("data_generation", {})
    train_per_task = data_cfg.get("train_per_task", 60)
    eval_per_task = data_cfg.get("eval_per_task", 20)

    failures_path = config.get("outputs", {}).get(
        "teacher_failures_path",
        "outputs/json_teacher_generation_failures.json",
    )
    metadata_path = config.get("outputs", {}).get(
        "teacher_metadata_path",
        "outputs/json_teacher_generation_metadata.json",
    )

    prompt_splits = build_prompt_bank(
        train_per_task=train_per_task,
        eval_per_task=eval_per_task,
    )

    train_data, train_failures = generate_dataset_split(
        client=client,
        model_name=teacher_model,
        examples=prompt_splits["train"],
        temperature=temperature,
        max_tokens=max_tokens,
        split_name="train",
    )

    eval_data, eval_failures = generate_dataset_split(
        client=client,
        model_name=teacher_model,
        examples=prompt_splits["eval"],
        temperature=temperature,
        max_tokens=max_tokens,
        split_name="eval",
    )

    save_json(train_data, output_path)
    save_json(eval_data, eval_path)
    save_json(
        {
            "train_failures": train_failures,
            "eval_failures": eval_failures,
        },
        failures_path,
    )

    metadata = {
        "teacher_model": teacher_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "random_seed": RANDOM_SEED,
        "train_per_task_requested": train_per_task,
        "eval_per_task_requested": eval_per_task,
        "train_examples_saved": len(train_data),
        "eval_examples_saved": len(eval_data),
        "train_failures": len(train_failures),
        "eval_failures": len(eval_failures),
        "task_counts_train": {},
        "task_counts_eval": {},
    }

    for row in train_data:
        task = row["task_type"]
        metadata["task_counts_train"][task] = metadata["task_counts_train"].get(task, 0) + 1

    for row in eval_data:
        task = row["task_type"]
        metadata["task_counts_eval"][task] = metadata["task_counts_eval"].get(task, 0) + 1

    save_json(metadata, metadata_path)

    print(f"Saved {len(train_data)} JSON training examples to {output_path}")
    print(f"Saved {len(eval_data)} JSON eval examples to {eval_path}")
    print(f"Saved failure log to {failures_path}")
    print(f"Saved generation metadata to {metadata_path}")


if __name__ == "__main__":
    main()

