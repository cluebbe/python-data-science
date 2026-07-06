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
    {"lat": 40.4168, "lng": -3.7038, "title": "Puerta del Sol", "description": "The heart of Madrid \U0001F31E", "icon": "https://maps.google.com/mapfiles/ms/icons/red-dot.png", "category": "restaurant"},
    {"lat": 40.4154, "lng": -3.7074, "title": "Plaza Mayor", "description": "Historic square with beautiful architecture", "icon": "https://maps.google.com/mapfiles/ms/icons/red-dot.png", "category": "restaurant"},
    {"lat": 40.4140, "lng": -3.7010, "title": "Casa Botín", "description": "Oldest restaurant in the world \U0001F37D️", "icon": "https://maps.google.com/mapfiles/ms/icons/red-dot.png", "category": "restaurant"},
    {"lat": 40.4175, "lng": -3.7050, "title": "La Latina District", "description": "Famous for tapas and restaurants", "icon": "https://maps.google.com/mapfiles/ms/icons/red-dot.png", "category": "restaurant"},
    {"lat": 40.4075, "lng": -3.6925, "title": "Retiro Park", "description": "Beautiful park with lake and monuments", "icon": "https://maps.google.com/mapfiles/ms/icons/green-dot.png", "category": "car_workshop"},
    {"lat": 40.4304, "lng": -3.7023, "title": "Santiago Bernabéu Stadium", "description": "Home of Real Madrid ⚽", "icon": "https://maps.google.com/mapfiles/ms/icons/green-dot.png", "category": "car_workshop"},
    {"lat": 40.4050, "lng": -3.6900, "title": "AutoZone Madrid", "description": "Car parts and service center \U0001F527", "icon": "https://maps.google.com/mapfiles/ms/icons/green-dot.png", "category": "car_workshop"},
    {"lat": 40.4320, "lng": -3.7000, "title": "Bernabéu Auto Service", "description": "Professional car repair shop", "icon": "https://maps.google.com/mapfiles/ms/icons/green-dot.png", "category": "car_workshop"},
]


@app.route("/")
def index():
    return render_template(
        "index.html",
        google_maps_key=os.getenv("GOOGLE_MAPS_KEY"),
        points=data_points
    )


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


if __name__ == "__main__":
    app.run(debug=True)
