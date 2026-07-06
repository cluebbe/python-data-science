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

## Step 1 — Project Skeleton & Data Points (`app.py`)

Create the Flask app instance, load the `.env` file, and define `data_points`: a Python list of dictionaries, each with `lat`, `lng`, `title`, `description`, `icon`, and `category`. Include at least 8 points split across two categories (e.g. `restaurant` and `car_workshop`) clustered around a city of your choice.

Also add an `after_request` hook that disables browser caching for every response.

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
```

**`data_points`** stands in for a database table in this small app — a real deployment would query this from Postgres, a spreadsheet import, or an API instead of hardcoding it. Every point needs `lat`/`lng` (fed to KNN as features) and `category` (fed to KNN as the label for classification).

**Why the `after_request` cache hook?** During development you'll frequently edit `data_points` and reload the page to see the change. Browsers aggressively cache plain HTML/GET responses; without these headers you might keep seeing a stale list of points even after editing and restarting the server. Setting `Cache-Control: no-store` forces a fresh render every time.

</details>

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

---

## Step 8 — Run & Test the App

With your `.env` file in place (Step "Preparation — Environment Setup") and the virtual environment activated, start the Flask development server:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser. You should see a map centered on your chosen city with the preloaded markers visible.

Try both modes:

- **Search mode** (default) — click anywhere on the map; you should see `K` numbered markers appear along with a distance-sorted list in the top-left panel
- **Classification mode** — switch the radio button, click again; you should see a predicted category, a confidence percentage for each category, and markers for the closest locations sharing that category

Experiment with the `K Neighbors` input — a small `K` (e.g. 1) makes the classification snap to whichever single point is closest, while a larger `K` averages over more neighbors and can flip the prediction near a category boundary.

**Troubleshooting:**

| Symptom | Likely cause |
|---|---|
| Blank grey map, no markers | `GOOGLE_MAPS_KEY` missing/invalid, or the Maps JavaScript API isn't enabled for that key |
| `500` error on click | Flask console will show the traceback — check `k` is a valid number and `data_points` isn't empty |
| Stale markers after editing `data_points` | Hard-refresh the browser — the `after_request` cache headers from Step 1 should prevent this, but browser extensions can still cache aggressively |
