# K-Nearest Neighbors + Google Maps Flask App — Step-by-Step Tutorial

## Introduction to K-Nearest Neighbors

**K-Nearest Neighbors (KNN)** is a supervised machine learning algorithm used for both classification and regression. Unlike decision trees or logistic regression, KNN builds no internal model during training — it simply memorises the training data and defers all computation to prediction time. For this reason it is called a **lazy learner** (or **instance-based learner**).

### How it works

To predict a new point:

1. **Measure distance** — compute the distance from the query point to every point in the training set
2. **Rank** — sort all training points by distance, ascending
3. **Select K** — take the `K` closest points (the "neighbors")
4. **Aggregate** — for **classification**, take a majority vote of the neighbors' labels; for **regression**, average their values

There is no fitting step in the usual sense — `.fit()` for a KNN model just stores the training data (and optionally builds a spatial index) so that `.predict()` can search it quickly.

### Choosing K

| Small K (e.g. 1) | Large K |
|---|---|
| Follows the training data very closely | Smooths over local detail |
| Sensitive to noise and outliers — can overfit | Can wash out real local structure — can underfit |
| Decision boundary is jagged | Decision boundary is smooth |

An odd K is usually preferred for binary classification to avoid tied votes.

### Distance metrics — why Haversine matters here

Most introductions to KNN use **Euclidean distance** — a straight line through feature space. That works for generic numeric features, but this project's features are **latitude and longitude**: coordinates on the surface of a sphere. Treating them as flat Euclidean coordinates distorts distance, especially over larger areas, because a degree of longitude covers less real-world distance the further you are from the equator.

The **Haversine formula** instead computes the great-circle distance between two points on a sphere:

```
a = sin²(Δφ/2) + cos(φ₁) · cos(φ₂) · sin²(Δλ/2)
d = 2 · R · atan2(√a, √(1−a))
```

Where `φ` is latitude, `λ` is longitude, and `R` is the Earth's radius. scikit-learn's `NearestNeighbors` and `KNeighborsClassifier` both accept `metric='haversine'`, which is exactly what this app uses to compare geographic points correctly.

> **Caveat worth knowing:** scikit-learn's `haversine` metric expects coordinates in **radians**, not degrees. This tutorial (matching the original app) fits directly on latitude/longitude in degrees without converting via `np.radians()` first. For the small, tightly clustered area used here (a few kilometers across Madrid), the *ordering* of neighbors still comes out correct, but the raw `distance` values returned by the API are not true kilometers — just a consistent ranking value. In a production app spanning a wider area, always convert with `np.radians()` before fitting and multiply the result by the Earth's radius to recover real distances.

### Two flavors used in this app

| Class | Purpose | Used for |
|---|---|---|
| `NearestNeighbors` | Unsupervised — just finds the K closest points to a query, no labels involved | "Show me the 3 closest places to where I clicked" |
| `KNeighborsClassifier` | Supervised — finds the K closest points **and** votes on their labels | "What type of place is this, based on what's around it?" |

### Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| Simple to understand and implement | Slow at prediction time — must scan the whole dataset (no learned shortcuts) |
| No training phase | Sensitive to the scale of features |
| Naturally handles multi-class problems | Sensitive to irrelevant or noisy features |
| Decision boundary adapts to any shape of data | Performance depends heavily on the choice of K and distance metric |

---

## What we're building

A small Flask web app with an interactive Google Map. The map is preloaded with a handful of points of interest around Madrid, each tagged with a category (`restaurant` or `car_workshop`). Clicking anywhere on the map does one of two things, depending on the selected mode:

- **Search mode** — finds the `K` nearest points of interest to the click, using `NearestNeighbors`
- **Classification mode** — predicts which category the clicked location most likely belongs to, using `KNeighborsClassifier`, then shows the confidence per category and the closest points that share the predicted category

All the machine learning happens server-side in Flask; the browser only renders markers and calls two JSON endpoints.

### Files in this tutorial

```
knn-google-maps/
├── app.py                 # Flask backend + KNN endpoints
├── templates/
│   └── index.html         # Google Map frontend
└── .env                   # your Google Maps API key (not committed)
```

---

## Preparation — Environment Setup

Before running any code, install Python and set up an isolated environment.

**Install Python 3.9 or newer** from [python.org](https://www.python.org/downloads/). Verify it is available in your terminal:

```bash
python3 --version
```

> **Windows users:** during installation, tick **"Add Python to PATH"** so the `python` and `pip` commands are available in your terminal.

Then set up an isolated environment:

```bash
# 1. Create and enter your project folder
mkdir knn-google-maps && cd knn-google-maps

# 2. Create a virtual environment (run once)
python3 -m venv venv

# 3. Activate it
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install flask python-dotenv scikit-learn numpy

# 5. When you're done, deactivate
deactivate
```

### Get a Google Maps API key

1. Open the [Google Cloud Console](https://console.cloud.google.com/) and create (or select) a project
2. Enable the **Maps JavaScript API**
3. Create an API key under **APIs & Services → Credentials**
4. (Recommended) restrict the key to the Maps JavaScript API and to your local/dev referrers

Create a file named `.env` in your project folder:

```
GOOGLE_MAPS_KEY=AIzaSy...your_actual_key_here...
```

> Never commit your `.env` file — add it to `.gitignore`. A blank grey map almost always means the key is missing, invalid, or the Maps JavaScript API isn't enabled for it.

---

## Preparation — Imports

```python
# app.py
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
```

- **flask** — the web framework: routes, request parsing, JSON responses, HTML templating
- **os** — reads the API key out of the environment
- **python-dotenv** — loads `.env` into `os.environ` so `os.getenv` can see it
- **sklearn.neighbors** — `NearestNeighbors` (plain search) and `KNeighborsClassifier` (classification)
- **numpy** — builds the coordinate arrays scikit-learn expects

The frontend (`templates/index.html`) needs no Python imports — it loads the Google Maps JavaScript API directly via a `<script>` tag.

---

## How to work through this tutorial — run it after every step

Do **not** save running the app for the end. Every step below finishes with a **Checkpoint**: a command to run and the output you should see. A step that doesn't pass its checkpoint is a step to fix now, while the change that broke it is the only thing you just touched. Debugging eight steps at once is what makes web apps feel hard.

Set yourself up with two terminals:

| Terminal | Purpose |
|---|---|
| **1 — the server** | `python app.py`, started once in Step 1 and left running for the whole tutorial |
| **2 — the client** | `curl` commands for the JSON checkpoints in Steps 3 and 4 |

`app.run(debug=True)` (added in Step 1) enables Flask's **auto-reloader**: every time you save `app.py`, the server restarts itself and you'll see a fresh `Restarting with stat` line in Terminal 1. You never need to stop and restart it manually. Debug mode also renders tracebacks in the browser instead of a bare `500` page.

Changes to `templates/index.html` don't even need a reload — just refresh the browser (the `after_request` cache headers from Step 1 make sure you get the new version).

From Step 5 onward, keep your browser's **DevTools** open (`F12`, or `Cmd+Option+I` on macOS):

- the **Console** tab shows JavaScript errors — a typo there breaks the map silently, with no Python traceback anywhere
- the **Network** tab shows each `fetch` to `/knn_search` and `/knn_classification`, with the exact JSON sent and received

> **Windows users:** in PowerShell, `curl` is an alias for `Invoke-WebRequest` and won't accept the flags below. Write `curl.exe` explicitly (bundled with Windows 10+), or run the checkpoints from Git Bash / WSL.

---

## Step 1 — Project Skeleton & Data Points (`app.py`)

Create the Flask app instance, load the `.env` file, and define `data_points`: a Python list of dictionaries, each with `lat`, `lng`, `title`, `description`, `icon`, and `category`. Include at least 8 points split across two categories (e.g. `restaurant` and `car_workshop`) clustered around a city of your choice.

Also add an `after_request` hook that disables browser caching for every response, and — at the very bottom of the file — the `if __name__ == "__main__"` block that starts the development server with `debug=True`, so the app is runnable from this step onward.

<details>
<summary>Solution</summary>

```python
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np

load_dotenv()

app = Flask(__name__)


@app.after_request
def add_cache_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, public, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# === YOUR CUSTOM DATA POINTS ===
data_points = [
    {"lat": 40.4168, "lng": -3.7038, "title": "Puerta del Sol", "description": "The heart of Madrid", "icon": "https://maps.google.com/mapfiles/ms/icons/red-dot.png", "category": "restaurant"},
    {"lat": 40.4154, "lng": -3.7074, "title": "Plaza Mayor", "description": "Historic square with beautiful architecture", "icon": "https://maps.google.com/mapfiles/ms/icons/red-dot.png", "category": "restaurant"},
    {"lat": 40.4140, "lng": -3.7010, "title": "Casa Botín", "description": "Oldest restaurant in the world", "icon": "https://maps.google.com/mapfiles/ms/icons/red-dot.png", "category": "restaurant"},
    {"lat": 40.4175, "lng": -3.7050, "title": "La Latina District", "description": "Famous for tapas and restaurants", "icon": "https://maps.google.com/mapfiles/ms/icons/red-dot.png", "category": "restaurant"},
    {"lat": 40.4075, "lng": -3.6925, "title": "Retiro Park", "description": "Beautiful park with lake and monuments", "icon": "https://maps.google.com/mapfiles/ms/icons/green-dot.png", "category": "car_workshop"},
    {"lat": 40.4304, "lng": -3.7023, "title": "Santiago Bernabéu Stadium", "description": "Home of Real Madrid", "icon": "https://maps.google.com/mapfiles/ms/icons/green-dot.png", "category": "car_workshop"},
    {"lat": 40.4050, "lng": -3.6900, "title": "AutoZone Madrid", "description": "Car parts and service center", "icon": "https://maps.google.com/mapfiles/ms/icons/green-dot.png", "category": "car_workshop"},
    {"lat": 40.4320, "lng": -3.7000, "title": "Bernabéu Auto Service", "description": "Professional car repair shop", "icon": "https://maps.google.com/mapfiles/ms/icons/green-dot.png", "category": "car_workshop"},
]


if __name__ == "__main__":
    app.run(debug=True)
```

**`data_points`** stands in for a database table in this small app — a real deployment would query this from Postgres, a spreadsheet import, or an API instead of hardcoding it. Every point needs `lat`/`lng` (fed to KNN as features) and `category` (fed to KNN as the label for classification).

**Why the `after_request` cache hook?** During development you'll frequently edit `data_points` and reload the page to see the change. Browsers aggressively cache plain HTML/GET responses; without these headers you might keep seeing a stale list of points even after editing and restarting the server. Setting `Cache-Control: no-store` forces a fresh render every time.

**`if __name__ == "__main__"`** only runs when you execute the file directly (`python app.py`), not when something imports it. `debug=True` turns on the auto-reloader and in-browser tracebacks — never use it in production, since the debugger lets anyone who can reach the page execute code on your server.

</details>

### Checkpoint — the server starts

In Terminal 1, with the virtual environment activated:

```bash
python app.py
```

**Expected output:**

```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
```

Leave this running for the rest of the tutorial. Visiting `http://127.0.0.1:5000` right now correctly gives **404 Not Found** — you haven't defined any routes yet. What you've proved is that Python, Flask, scikit-learn and numpy all import cleanly and the port is free; an `ImportError` or `ModuleNotFoundError` here means the `pip install` from Preparation didn't land in the active virtual environment.

In Terminal 2, confirm your data survived the round trip through Python:

```bash
python -c "import app; print(len(app.data_points), sorted({p['category'] for p in app.data_points}))"
```

**Expected output:**

```
8 ['car_workshop', 'restaurant']
```

Both categories must appear — a `KNeighborsClassifier` trained on a single label can never predict anything else, and Step 4 would look broken for a reason that actually originates here.

---

## Step 2 — Serve the Map Page (`app.py`)

Add the `/` route. It should render `templates/index.html`, passing the Google Maps API key (read from the environment) and the full `data_points` list into the template.

<details>
<summary>Solution</summary>

```python
@app.route("/")
def index():
    return render_template(
        "index.html",
        google_maps_key=os.getenv("GOOGLE_MAPS_KEY"),
        points=data_points
    )
```

**`render_template`** looks for `index.html` inside a `templates/` folder by convention — Flask will not find it anywhere else without extra configuration.

**`os.getenv("GOOGLE_MAPS_KEY")`** reads the key loaded from `.env` by `load_dotenv()` in Step 1. Passing it as a template variable (rather than hardcoding it in the HTML) keeps the key out of source control.

**`points=data_points`** hands the entire Python list to Jinja. The template will later convert it to a JSON array for use in JavaScript — see Step 5.

</details>

### Checkpoint — the route renders and the API key is loaded

The real `templates/index.html` isn't written until Step 5, so create a throwaway placeholder now — you'd otherwise get `jinja2.exceptions.TemplateNotFound`:

```bash
mkdir -p templates
```

```html
<!-- templates/index.html — temporary, replaced in Step 5 -->
<p>Points: {{ points|length }}</p>
<p>Key loaded: {{ 'yes' if google_maps_key else 'NO' }}</p>
<p>First point: {{ points[0].title }}</p>
```

Save `app.py`; Terminal 1 shows `Restarting with stat`. Open `http://127.0.0.1:5000`:

**Expected output:**

```
Points: 8
Key loaded: yes
First point: Puerta del Sol
```

Three separate things just got verified, and it's worth being explicit about which is which, because they fail independently:

- **`Points: 8`** — Flask found `templates/`, and the `data_points` list reached Jinja
- **`Key loaded: yes`** — `load_dotenv()` found your `.env` and `os.getenv` read the variable. If this says **`NO`**, fix it *now*: it is the single most common cause of the blank grey map in Step 5, and it is far easier to diagnose on this page than through the Google Maps API's silent failure. Check that `.env` sits next to `app.py`, that the line reads `GOOGLE_MAPS_KEY=AIza...` with no quotes and no spaces around the `=`, and that you restarted the server after creating it (`load_dotenv()` only runs at import time)
- **`First point: Puerta del Sol`** — attribute access on the dicts works, which is what Step 5's marker loop depends on

---

## Step 3 — Build the `/knn_search` Endpoint (`app.py`)

Add a `POST /knn_search` route. It should:

1. Read `lat`, `lng`, and `k` (default `3`) from the JSON request body
2. Return a `400` JSON error if `lat`/`lng` are missing
3. Build a feature matrix from every point's `lat`/`lng`
4. Fit a `NearestNeighbors` model using the `haversine` metric
5. Query the `k` nearest neighbors to the clicked point
6. Return each neighbor's coordinates, title, description, and distance as JSON

<details>
<summary>Solution</summary>

```python
@app.route("/knn_search", methods=["POST"])
def knn_search():
    data = request.json
    lat = data.get("lat")
    lng = data.get("lng")
    k = data.get("k", 3)
    if lat is None or lng is None:
        return jsonify({"error": "Missing lat/lng"}), 400

    X = np.array([[point["lat"], point["lng"]] for point in data_points])
    knn = NearestNeighbors(n_neighbors=min(k, len(data_points)), metric='haversine')
    knn.fit(X)
    query_point = np.array([[lat, lng]])
    distances, indices = knn.kneighbors(query_point)

    neighbors = []
    for idx, distance in zip(indices[0], distances[0]):
        point = data_points[int(idx)]
        neighbors.append({
            "lat": point["lat"], "lng": point["lng"],
            "title": point["title"], "description": point["description"],
            "distance": float(distance)
        })
    return jsonify({"query": {"lat": lat, "lng": lng}, "neighbors": neighbors})
```

**`min(k, len(data_points))`** guards against a `k` larger than the number of available points — `NearestNeighbors` raises a `ValueError` if you ask for more neighbors than samples exist.

**`knn.fit(X)`** — for `NearestNeighbors` this doesn't "learn" anything in the statistical sense; it just indexes the points so `kneighbors()` can search them efficiently. Refitting on every request is fine at this scale (8 points); a production app with thousands of points would fit once at startup and reuse the model.

**`kneighbors(query_point)`** returns two parallel arrays: `distances` (sorted ascending) and `indices` (the row positions in `X`, and therefore in `data_points`, of each neighbor). Zipping them together lets you look up the original point for each result while keeping its distance.

</details>

### Checkpoint — the search endpoint answers

No frontend is needed to test a JSON endpoint, and you shouldn't wait for one. In Terminal 2, click a point on the map *by hand* — a coordinate just north-east of Puerta del Sol:

```bash
curl -s -X POST http://127.0.0.1:5000/knn_search \
  -H "Content-Type: application/json" \
  -d '{"lat": 40.4180, "lng": -3.7020, "k": 3}' | python -m json.tool
```

**Expected output:**

```json
{
    "neighbors": [
        {
            "description": "The heart of Madrid",
            "distance": 0.0020330501728824894,
            "lat": 40.4168,
            "lng": -3.7038,
            "title": "Puerta del Sol"
        },
        {
            "description": "Famous for tapas and restaurants",
            "distance": 0.002780958249952361,
            "lat": 40.4175,
            "lng": -3.705,
            "title": "La Latina District"
        },
        {
            "description": "Oldest restaurant in the world",
            "distance": 0.004102464103840381,
            "lat": 40.414,
            "lng": -3.701,
            "title": "Casa Botín"
        }
    ],
    "query": {
        "lat": 40.418,
        "lng": -3.702
    }
}
```

Check the shape, not the exact decimals: **three** neighbors for `k=3`, `distance` **ascending**, and the nearest one being the point you'd expect from eyeballing the coordinates. Remember the caveat from the introduction — these distances aren't kilometers, only a consistent ranking.

Now test the error path and the `k` guard, which a mouse click in the browser can never trigger:

```bash
# missing lng -> 400, not a traceback
curl -s -i -X POST http://127.0.0.1:5000/knn_search \
  -H "Content-Type: application/json" -d '{"lat": 40.4180}' | head -1

# k larger than the dataset -> clamped to 8 by min(), not a ValueError
curl -s -X POST http://127.0.0.1:5000/knn_search \
  -H "Content-Type: application/json" \
  -d '{"lat": 40.4180, "lng": -3.7020, "k": 99}' | python -c "import json,sys; print(len(json.load(sys.stdin)['neighbors']), 'neighbors')"
```

**Expected output:**

```
HTTP/1.1 400 BAD REQUEST
8 neighbors
```

A `500` on the first command means your `lat`/`lng` guard is missing or placed after the code that uses them; a `ValueError: Expected n_neighbors <= n_samples` on the second means the `min(k, len(data_points))` clamp isn't there.

---

## Step 4 — Build the `/knn_classification` Endpoint (`app.py`)

Add a `POST /knn_classification` route. It should:

1. Read `lat`, `lng`, and `k` from the JSON request body (all required)
2. Fit a `KNeighborsClassifier` on every point's coordinates (features) and category (labels), using the `haversine` metric
3. Predict the category of the clicked point and its class probabilities
4. Re-run a `NearestNeighbors` search restricted to only the points that share the predicted category, to surface the closest "similar" locations
5. Return the predicted category, per-category confidence, and the similar points as JSON

<details>
<summary>Solution</summary>

```python
@app.route("/knn_classification", methods=["POST"])
def knn_classification():
    data = request.json
    lat = data.get("lat")
    lng = data.get("lng")
    k = data.get("k", 3)
    if lat is None or lng is None or k is None:
        return jsonify({"error": "Missing lat/lng/k"}), 400

    X = np.array([[point["lat"], point["lng"]] for point in data_points])
    y = np.array([point["category"] for point in data_points])

    knn_classifier = KNeighborsClassifier(n_neighbors=min(k, len(data_points)), metric='haversine')
    knn_classifier.fit(X, y)

    query_point = np.array([[lat, lng]])
    predicted_category = knn_classifier.predict(query_point)[0]
    probabilities = knn_classifier.predict_proba(query_point)[0]
    class_names = knn_classifier.classes_

    category_points = [point for point in data_points if point["category"] == predicted_category]
    similar_points = []
    if category_points:
        X_category = np.array([[point["lat"], point["lng"]] for point in category_points])
        category_knn = NearestNeighbors(n_neighbors=min(k, len(category_points)), metric='haversine')
        category_knn.fit(X_category)
        distances, indices = category_knn.kneighbors(query_point)
        for idx, distance in zip(indices[0], distances[0]):
            point = category_points[int(idx)]
            similar_points.append({
                "lat": point["lat"],
                "lng": point["lng"],
                "title": point["title"],
                "description": point["description"],
                "distance": float(distance)
            })

    return jsonify({
        "query": {"lat": lat, "lng": lng},
        "predicted_category": predicted_category,
        "probabilities": {class_names[0]: float(probabilities[0]), class_names[1]: float(probabilities[1])},
        "similar_points": similar_points
    })
```

**`knn_classifier.fit(X, y)`** is the supervised step: unlike `NearestNeighbors`, this model is given labels (`y`) alongside coordinates (`X`), so `predict` can return a category, not just a list of neighbors.

**`predict_proba`** returns the fraction of the `k` nearest neighbors that belong to each class — this *is* KNN's "confidence": if `k=5` and 4 of the 5 nearest points are `restaurant`, the probability for `restaurant` is `0.8`. **`classes_`** gives the label order matching the probability array, since scikit-learn always returns classes sorted alphabetically rather than in the order you defined them.

**Why search again after classifying?** `predict` tells you the most likely category, but the plain nearest neighbors (from Step 3) might include points from *both* categories. Filtering `data_points` down to `category_points` first, then running a second `NearestNeighbors` search only within that filtered set, guarantees the "similar locations" shown to the user actually match the predicted category — e.g. don't show a car workshop as a "similar restaurant."

</details>

### Checkpoint — the classifier predicts, and K actually changes the answer

Same query point as Step 3, now through the classifier:

```bash
curl -s -X POST http://127.0.0.1:5000/knn_classification \
  -H "Content-Type: application/json" \
  -d '{"lat": 40.4180, "lng": -3.7020, "k": 3}' | python -m json.tool
```

**Expected output** (`similar_points` abbreviated — it matches Step 3's three restaurants):

```json
{
    "predicted_category": "restaurant",
    "probabilities": {
        "car_workshop": 0.0,
        "restaurant": 1.0
    },
    "query": {
        "lat": 40.418,
        "lng": -3.702
    },
    "similar_points": [ ... ]
}
```

Deep in restaurant territory, all 3 neighbors vote the same way, so confidence is a flat `1.0`. That's correct but uninformative — it would look identical if `predict_proba` were broken. So probe a point *between* the two clusters, where the vote should actually split:

```bash
for K in 1 3 5; do
  curl -s -X POST http://127.0.0.1:5000/knn_classification \
    -H "Content-Type: application/json" \
    -d "{\"lat\": 40.4120, \"lng\": -3.6960, \"k\": $K}" |
  python -c "import json,sys; d=json.load(sys.stdin); print('k=$K ->', d['predicted_category'], d['probabilities'])"
done
```

**Expected output:**

```
k=1 -> restaurant {'car_workshop': 0.0, 'restaurant': 1.0}
k=3 -> restaurant {'car_workshop': 0.3333333333333333, 'restaurant': 0.6666666666666666}
k=5 -> restaurant {'car_workshop': 0.4, 'restaurant': 0.6}
```

This is the theory from the introduction, visible in your own output: confidence is just the *fraction of the K neighbors* voting for each class, so it can only ever take values `n/K` — `1/1`, `2/3`, `3/5`. As K grows, the query point sees more of the rival cluster and confidence erodes toward a coin flip. Watching that number move as you vary K is the fastest proof that the classifier is genuinely voting rather than returning a constant.

Two things worth confirming in the JSON before moving on, since the frontend depends on both:

- every `similar_points` entry belongs to the **predicted** category — that's the second `NearestNeighbors` search doing its job
- `probabilities` has **both** keys, `restaurant` and `car_workshop`. Step 7 reads them by name, so a missing key surfaces there as `undefined%` rather than as an error here

---

## Step 5 — Frontend Skeleton & Map Markers (`templates/index.html`)

Build the HTML page: a full-screen `#map` div, a floating control panel (`.header`) with a mode toggle (Search / Classification radio buttons) and a `K Neighbors` number input, and a results panel. Then write `initMap()`: create the Google Map centered on your city, inject the Jinja-rendered points as a JavaScript array, and place a marker with an info window for each one.

<details>
<summary>Solution</summary>

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Python + Google Maps</title>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; }
        #map { height: 100vh; width: 100%; }
        .header {
            position: absolute; top: 10px; left: 10px; background: white;
            padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); z-index: 1000;
        }
        .controls { background: white; padding: 12px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); margin-top: 10px; display: flex; gap: 10px; align-items: center; }
        .results { background: white; padding: 12px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); margin-top: 10px; font-size: 12px; max-height: 200px; overflow-y: auto; }
        .result-item { padding: 8px; border-bottom: 1px solid #eee; background: #f9f9f9; }
        .result-item:last-child { border-bottom: none; }
        .result-title { font-weight: bold; color: #333; }
        .result-distance { color: #666; font-size: 11px; }
    </style>
</head>
<body>

<div class="header">
    <h2>Python Flask + Google Maps</h2>
    <p>{{ points|length }} custom data points loaded from Python</p>

    <div class="controls">
        <label>Mode:</label>
        <div style="display: flex; gap: 10px;">
            <label><input type="radio" name="mode" value="search" checked> Search</label>
            <label><input type="radio" name="mode" value="classification"> Classification</label>
        </div>
    </div>

    <div class="controls">
        <label for="kNeighbors">K Neighbors:</label>
        <input type="number" id="kNeighbors" value="3" min="1" max="10">
    </div>

    <div id="results" class="results" style="display: none;">
        <strong id="resultsTitle">KNN Results:</strong>
        <div id="resultsList"></div>
    </div>
    <p style="font-size: 12px; color: #666; margin-top: 10px;">
        <span id="instructionText">Click on the map to find nearest neighbors</span>
    </p>
</div>

<div id="map"></div>

<script>
    let map, clickMarker = null, resultMarkers = [];

    function initMap() {
        map = new google.maps.Map(document.getElementById("map"), {
            zoom: 13,
            center: { lat: 40.4168, lng: -3.7038 },
            mapTypeId: "roadmap"
        });

        const mapPoints = {{ points | tojson | safe }};
        mapPoints.forEach(p => {
            const marker = new google.maps.Marker({
                position: { lat: p.lat, lng: p.lng },
                map: map,
                title: p.title,
                icon: p.icon
            });
            const info = new google.maps.InfoWindow({ content: `<h3>${p.title}</h3><p>${p.description}</p>` });
            marker.addListener("click", () => info.open(map, marker));
        });
    }
</script>

<script async src="https://maps.googleapis.com/maps/api/js?key={{ google_maps_key }}&callback=initMap"></script>
</body>
</html>
```

**`{{ points | tojson | safe }}`** is the bridge between Python and JavaScript: Jinja's `tojson` filter serializes the Python list of dicts to a JSON string, and `safe` tells Jinja not to HTML-escape it (which would otherwise corrupt the JSON's quotes). The result is valid JavaScript — a literal array of objects — spliced directly into the `<script>` block at render time.

**`callback=initMap`** in the Google Maps script tag is how the Maps JavaScript API bootstraps itself: it loads asynchronously (`async`), and once ready, calls the global `initMap` function by name. This is why `initMap` must be defined *before* that script tag runs — the callback wiring happens by string lookup on `window`.

**One marker + one `InfoWindow` per point** — each info window starts closed; the `click` listener opens it only when its own marker is clicked, showing the title and description passed in from Python.

</details>

### Checkpoint — the map draws your points

This replaces the placeholder template from Step 2. No server restart is needed for a template change — just refresh `http://127.0.0.1:5000`.

**Expected output:** a roadmap centred on Madrid, `8 custom data points loaded from Python` in the panel, and **8 markers** — 4 red around the centre, 4 green further out. Clicking a marker opens an info window with its title and description.

Open DevTools and confirm two things before touching Step 6:

- the **Console** tab is clean. `initMap is not a function` means `initMap` is defined after the Maps script tag instead of before it; `Uncaught SyntaxError` in the inline script usually means `| safe` is missing from the `tojson` filter, so Jinja HTML-escaped the quotes into `&#34;` and broke the JavaScript literal
- the **Network** tab shows the request to `maps.googleapis.com` returning **200**. A `403` or `REQUEST_DENIED` is the API key — the key itself, its restrictions, or the Maps JavaScript API not being enabled, all of which are Google Cloud Console problems rather than code problems

A **blank grey map** is the classic symptom here. The distinction that saves you time: if the panel still says `8 custom data points loaded from Python`, then Flask, Jinja and your data are all fine and the fault is purely the key — which your Step 2 checkpoint already ruled out or caught.

---

## Step 6 — Handle Map Clicks & Search Mode (`templates/index.html`)

Add a click listener to the map that checks which radio button is selected and calls the appropriate handler. Implement `performKNNSearch(lat, lng)`: place a temporary marker at the click location, `POST` to `/knn_search` with the coordinates and selected `k`, then render the results with `displaySearchResults`, which should place a numbered marker for each neighbor and list them (with distance) in the results panel.

<details>
<summary>Solution</summary>

```javascript
map.addListener("click", e => {
    const mode = document.querySelector('input[name="mode"]:checked').value;
    if (mode === "search") performKNNSearch(e.latLng.lat(), e.latLng.lng());
    else performKNNClassification(e.latLng.lat(), e.latLng.lng());
});

function performKNNSearch(lat, lng) {
    const k = parseInt(document.getElementById("kNeighbors").value) || 3;
    if (clickMarker) clickMarker.setMap(null);
    clickMarker = new google.maps.Marker({ position: { lat, lng }, map, icon: "http://maps.google.com/mapfiles/ms/icons/yellow-dot.png" });

    fetch("/knn_search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lng, k })
    })
    .then(r => r.json())
    .then(data => displaySearchResults(data.neighbors));
}

function clearResults() {
    resultMarkers.forEach(m => m.setMap(null));
    resultMarkers = [];
    document.getElementById("results").style.display = "none";
}

function displaySearchResults(neighbors) {
    clearResults();
    const list = document.getElementById("resultsList");
    list.innerHTML = "";
    neighbors.forEach((n, i) => {
        const marker = new google.maps.Marker({ position: { lat: n.lat, lng: n.lng }, map, label: String(i+1) });
        resultMarkers.push(marker);

        const div = document.createElement("div");
        div.className = "result-item";
        div.innerHTML = `
            <div class="result-title">${i+1}. ${n.title}</div>
            <div class="result-distance">Distance: ${n.distance.toFixed(4)}°</div>
            <div style="font-size:11px;color:#999;">${n.description}</div>`;
        list.appendChild(div);
    });
    document.getElementById("results").style.display = "block";
}
```

**`e.latLng.lat()` / `e.latLng.lng()`** — the Google Maps click event gives you a `LatLng` object, not plain numbers; these accessor methods extract the coordinates to send to Flask.

**`if (clickMarker) clickMarker.setMap(null)`** removes the previous click marker before placing a new one — `setMap(null)` is the Google Maps API's way of removing a marker from the map (there's no `marker.remove()`).

**`clearResults()`** is called at the start of every new search so that clicking a second location doesn't leave the previous numbered markers behind — it wipes `resultMarkers` and hides the panel before the new results are rendered.

**`label: String(i+1)`** puts a number (1, 2, 3, ...) directly inside the marker icon, matching the numbered list in the results panel so users can visually connect a marker to its row.

</details>

### Checkpoint — clicking the map returns neighbors

Refresh, leave the mode on **Search**, and click somewhere between the red markers near the city centre.

**Expected output:** a yellow marker at your click, **3** numbered markers on the nearest points, and a results panel listing those 3 with ascending distances. Click a second, different spot — the old yellow and numbered markers must **disappear** rather than accumulate; if they pile up, `clearResults()` isn't being called or `setMap(null)` is missing.

Then vary the inputs, since this is the first step where the whole stack is wired together:

- set **K Neighbors** to `1`, click → exactly 1 marker; set it to `8` → 8 markers
- switch to **Classification** and click → nothing happens yet (`performKNNClassification` doesn't exist until Step 7). Expect `Uncaught ReferenceError: performKNNClassification is not defined` in the Console. That's the correct result for this step, not a bug

In the **Network** tab, click the `knn_search` entry. Its **Payload** should show the clicked coordinates and your chosen `k`, and its **Response** should be the same JSON shape you curled in Step 3. This is where a frontend bug separates cleanly from a backend one:

| What you see | Where the fault is |
|---|---|
| No `knn_search` request at all | The click listener — not registered, or the mode radio lookup is wrong |
| Request sent, `400`/`500` returned | The backend, or the payload the frontend built. Terminal 1 has the traceback |
| `200` with correct JSON, nothing on screen | `displaySearchResults` — element IDs or the `results` panel's `display` toggle |

---

## Step 7 — Classification Mode & Mode Toggle (`templates/index.html`)

Implement `performKNNClassification(lat, lng)` (mirrors `performKNNSearch` but calls `/knn_classification`) and `displayClassificationResults(data)`, which should show the predicted category with its confidence percentage per class, then list and mark the `similar_points`, colored by category (red for `restaurant`, green for `car_workshop`). Finally, wire up the mode radio buttons so switching modes updates the instruction text and clears any markers left over from the other mode.

<details>
<summary>Solution</summary>

```javascript
function performKNNClassification(lat, lng) {
    const k = parseInt(document.getElementById("kNeighbors").value) || 3;
    if (clickMarker) clickMarker.setMap(null);
    clickMarker = new google.maps.Marker({ position: { lat, lng }, map, icon: "http://maps.google.com/mapfiles/ms/icons/yellow-dot.png" });

    fetch("/knn_classification", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lng, k })
    })
    .then(r => r.json())
    .then(data => displayClassificationResults(data));
}

function displayClassificationResults(data) {
    clearResults();
    const list = document.getElementById("resultsList");
    list.innerHTML = "";

    const predDiv = document.createElement("div");
    predDiv.className = "result-item";
    predDiv.innerHTML = `
        <div class="result-title" style="color:#2e7d32;">Predicted Category: ${data.predicted_category}</div>
        <div style="font-size:12px;margin:5px 0;">
            <strong>Confidence:</strong><br>
            Restaurant: ${(data.probabilities.restaurant*100).toFixed(1)}%<br>
            Car Workshop: ${(data.probabilities.car_workshop*100).toFixed(1)}%
        </div>`;
    list.appendChild(predDiv);

    if (data.similar_points.length) {
        const header = document.createElement("div");
        header.className = "result-item";
        header.innerHTML = `<div class="result-title">Similar ${data.predicted_category} locations:</div>`;
        list.appendChild(header);

        data.similar_points.forEach((p, i) => {
            const marker = new google.maps.Marker({
                position: { lat: p.lat, lng: p.lng },
                map,
                label: String(i+1),
                icon: data.predicted_category === "restaurant" ?
                    "http://maps.google.com/mapfiles/ms/icons/red-dot.png" :
                    "http://maps.google.com/mapfiles/ms/icons/green-dot.png"
            });
            resultMarkers.push(marker);

            const div = document.createElement("div");
            div.className = "result-item";
            div.innerHTML = `
                <div class="result-title">${i+1}. ${p.title}</div>
                <div class="result-distance">Distance: ${p.distance.toFixed(4)}°</div>
                <div style="font-size:11px;color:#999;">${p.description}</div>`;
            list.appendChild(div);
        });
    }
    document.getElementById("results").style.display = "block";
}

window.addEventListener('load', () => {
    document.querySelectorAll('input[name="mode"]').forEach(r => {
        r.addEventListener('change', () => {
            document.getElementById("instructionText").textContent =
                r.value === "search" ? "Click on the map to find nearest neighbors" : "Click on the map to classify the location";
            document.getElementById("resultsTitle").textContent =
                r.value === "search" ? "KNN Search Results:" : "KNN Classification Results:";
            clearResults();
        });
    });
});
```

**Confidence display** reads directly from the `probabilities` object returned by Step 4's `predict_proba` — since this demo only has two categories, both are hardcoded by name; a more general version would loop over `Object.keys(data.probabilities)` to support any number of classes.

**Marker color mirrors category** — the ternary picks the red or green dot icon to match the icons already used for the preloaded points (Step 1), so a user can visually confirm the "similar locations" really do belong to the predicted category.

**The `change` listener on the mode radios** keeps the UI consistent when switching mode mid-session: it updates the help text so the user knows what a click will do next, and calls `clearResults()` so results from the *previous* mode don't linger on screen looking like they still apply.

</details>

### Checkpoint — classification mode in the browser

Refresh, switch to **Classification**, and click the same in-between spot you curled in Step 4 — roughly `40.4120, -3.6960`, south-east of the centre.

**Expected output:** the panel shows `Predicted Category: restaurant`, `Restaurant: 66.7%` / `Car Workshop: 33.3%` at `K=3`, and red numbered markers on the nearest restaurants. Those percentages are the Step 4 curl output rendered as UI — if the browser disagrees with your curl, the fault is in `displayClassificationResults`, not in the model.

Three behaviours to exercise now that both modes exist:

- **the toggle** — switch Search → Classification and back. Markers and the results panel must clear on every switch, and the instruction text must change. Leftover markers from the other mode are exactly what the `change` listener is there to prevent
- **K flipping the prediction** — with the click point fixed, raise K from `1` to `5` and re-click. Confidence should fall (`100%` → `66.7%` → `60%`) as more of the rival cluster is pulled into the vote. This is the small-K/large-K tradeoff from the introduction, now visible on screen
- **`undefined%`** in the confidence lines means `data.probabilities` lacks a key by that name — likelier a typo in the category strings in `data_points` (Step 1) than a frontend bug

---

## Step 8 — Final Run-Through & Troubleshooting

If you've been running the checkpoints, the app already works — nothing new gets built here. This step is the end-to-end pass you'd do before showing the app to someone else, starting from a **cold start** so you catch anything that only worked because of state left over in a long-running server.

Stop Terminal 1 with `Ctrl+C`, then:

```bash
source venv/bin/activate        # macOS / Linux
python app.py
```

Open `http://127.0.0.1:5000` and walk the whole feature set:

| # | Action | Expected |
|---|---|---|
| 1 | Load the page | 8 markers: 4 red (restaurants), 4 green (car workshops) |
| 2 | Click any marker | Info window with that point's title and description |
| 3 | Search mode, `K=3`, click the centre | Yellow click marker, 3 numbered markers, 3 rows sorted by ascending distance |
| 4 | Click a second spot | Previous markers gone, new ones drawn |
| 5 | Set `K=8`, click | 8 numbered markers |
| 6 | Switch to Classification | Panel clears, instruction text changes |
| 7 | Click deep in the red cluster | `restaurant` at `100%`, similar points all red |
| 8 | Click deep in the green cluster | `car_workshop` at `100%`, similar points all green |
| 9 | Click between the clusters, vary `K` from 1 to 5 | Confidence drops as K grows; near the boundary the prediction can flip |

Both DevTools tabs should stay quiet throughout: no Console errors, and every `knn_search`/`knn_classification` request returning `200`.

**Troubleshooting:**

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError` on startup | The virtual environment isn't activated, or `pip install` ran against a different interpreter — re-run the Preparation install with `venv` active |
| `Address already in use` / `Port 5000 is in use` | An earlier `python app.py` is still running, or (on macOS) **AirPlay Receiver** owns port 5000 — disable it in System Settings → General → AirDrop & Handoff, or run `app.run(debug=True, port=5001)` |
| `jinja2.exceptions.TemplateNotFound: index.html` | `index.html` isn't inside a `templates/` folder next to `app.py` — Flask looks nowhere else |
| Blank grey map, no markers | `GOOGLE_MAPS_KEY` missing/invalid, or the Maps JavaScript API isn't enabled for that key — re-run the Step 2 checkpoint, which reports the key's presence directly |
| Page renders, but no markers and a Console error | JavaScript, not Python — most often `initMap` defined after the Maps script tag, or `| safe` missing from `{{ points | tojson }}` |
| `500` error on click | Flask console will show the traceback — check `k` is a valid number and `data_points` isn't empty |
| `400 Missing lat/lng` on click | The frontend isn't sending the coordinates — check the request Payload in the Network tab against Step 6 |
| Prediction is always the same category | Every point in `data_points` shares one `category`, or the label strings are inconsistent (`"restaurant"` vs `"Restaurant"`) — the Step 1 checkpoint catches this |
| Code edits have no effect | The server didn't reload — confirm `debug=True` and look for `Restarting with stat` in Terminal 1 |
| Stale markers after editing `data_points` | Hard-refresh the browser — the `after_request` cache headers from Step 1 should prevent this, but browser extensions can still cache aggressively |

### Where to go next

- **Fit once, not per request** — both endpoints refit the model on every call. Move the `fit` to module level and see the latency drop; then think about what has to happen when `data_points` changes
- **Fix the radians caveat** — convert with `np.radians()` before fitting and multiply by `6371` to return real kilometers, then label the distances `km` in the UI instead of `°`
- **Generalise the confidence display** — loop over `Object.keys(data.probabilities)` so a third category works without touching the frontend
- **Automate these checkpoints** — the curl commands in Steps 3 and 4 are assertions in disguise. `pip install pytest`, use Flask's `app.test_client()`, and turn each one into a test that runs in a second without a browser
