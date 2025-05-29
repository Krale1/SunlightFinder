
const maptilerKey = "4CHvvrHJ9EnxPMxYto8d";
const shadeMapKey =
    "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6ImtyaXN0aWphbmtyYWxldnNraTRAZ21haWwuY29tIiwiY3JlYXRlZCI6MTc0ODUyNzM1NTM1NSwiaWF0IjoxNzQ4NTI3MzU1fQ.YFiMN3q0ShAoUYa2j9MHhNde_Qj6bidmdQ91HxjCgBQ";
const WEATHER_API_KEY = "8a48baf68292fe3068c6cee9b99cf24b";
const SKOPJE_COORDS = { lat: 41.998, lon: 21.435 };

// Weather update function
function updateWeather() {
    document.getElementById("weatherDesc").textContent = "Loading...";
    document.getElementById("weatherIcon").textContent = "⏳";

    fetch(
        `https://api.openweathermap.org/data/2.5/weather?lat=${SKOPJE_COORDS.lat}&lon=${SKOPJE_COORDS.lon}&units=metric&appid=${WEATHER_API_KEY}`
    )
        .then((res) => {
            if (!res.ok) throw new Error("Weather API response was not ok");
            return res.json();
        })
        .then((data) => {
            document.getElementById("currentTemp").textContent = `${Math.round(
                data.main.temp
            )}°C`;
            document.getElementById("weatherDesc").textContent =
                data.weather[0].description;
            document.getElementById("feelsLike").textContent = `${Math.round(
                data.main.feels_like
            )}°C`;
            document.getElementById("windSpeed").textContent = `${Math.round(
                data.wind.speed * 3.6
            )} km/h`;
            document.getElementById(
                "humidity"
            ).textContent = `${data.main.humidity}%`;
            document.getElementById("uvIndex").textContent = data.uvi || "--";

            // Update weather icon
            const weatherCode = data.weather[0].id;
            let weatherIcon = "⛅";
            if (weatherCode >= 200 && weatherCode < 300) weatherIcon = "⛈️";
            else if (weatherCode >= 300 && weatherCode < 400)
                weatherIcon = "🌧️";
            else if (weatherCode >= 500 && weatherCode < 600)
                weatherIcon = "🌧️";
            else if (weatherCode >= 600 && weatherCode < 700)
                weatherIcon = "❄️";
            else if (weatherCode >= 700 && weatherCode < 800)
                weatherIcon = "🌫️";
            else if (weatherCode === 800) weatherIcon = "☀️";
            else if (weatherCode > 800) weatherIcon = "☁️";
            document.getElementById("weatherIcon").textContent = weatherIcon;

            // Update timestamp
            const updateTime = new Date().toLocaleTimeString();
            document.getElementById(
                "weatherUpdate"
            ).textContent = `Last updated: ${updateTime}`;
            document.getElementById("lastUpdate").textContent =
                new Date().toLocaleString();
        })
        .catch((error) => {
            console.error("Error fetching weather:", error);
            document.getElementById("weatherDesc").textContent =
                "Weather data unavailable";
            document.getElementById("weatherIcon").textContent = "⚠️";
        });
}

// Update weather immediately and then every 30 minutes
updateWeather();
setInterval(updateWeather, 30 * 60 * 1000);

const map = new maplibregl.Map({
    container: "map",
    style: `https://api.maptiler.com/maps/streets/style.json?key=${maptilerKey}`,
    center: [21.435, 41.998],
    zoom: 16,
    pitch: 0,
    bearing: 0,
});

let shadeMap;
let cafeFeatures = [];

map.on("load", async () => {
    // Load cafes GeoJSON file
    const cafes = await fetch("skopje_cafes.geojson").then((res) =>
        res.json()
    );
    cafeFeatures = cafes.features;

    // Add shadow source and layer first
    map.addSource("shadow-source", {
        type: "geojson",
        data: {
            type: "FeatureCollection",
            features: [],
        },
    });

    map.addLayer({
        id: "shadow-layer",
        type: "fill",
        source: "shadow-source",
        paint: {
            "fill-color": "#000",
            "fill-opacity": 0.7,
        },
    });

    // Then add cafes
    map.addSource("cafes", {
        type: "geojson",
        data: cafes,
    });

    map.addLayer({
        id: "cafes-layer",
        type: "circle",
        source: "cafes",
        paint: {
            "circle-radius": 6,
            "circle-color": [
                "case",
                ["==", ["get", "in_sun"], true],
                "#ff6600",
                "#666666",
            ],
            "circle-stroke-width": 2,
            "circle-stroke-color": "#fff",
        },
    });

    // Add sun status symbols
    map.addLayer({
        id: "cafe-symbols",
        type: "symbol",
        source: "cafes",
        layout: {
            "text-field": ["case", ["==", ["get", "in_sun"], true], "🌞", "🌚"],
            "text-size": 14,
            "text-offset": [0, -1.5],
            "text-allow-overlap": true,
        },
    });

    // Popup on click
    map.on("click", "cafes-layer", (e) => {
        const feature = e.features[0];
        const name = feature.properties.name || "Unknown Café";
        const street = feature.properties["addr:street"] || "Unknown street";
        const isInSun = feature.properties.in_sun;
        const sunStatus = isInSun ? "🌞 In sunlight" : "🌚 In shadow";

        const popupHtml = `
          <div class="cafe-popup">
            <h3>${name}</h3>
            <p>${street}</p>
            <div class="sun-status">${sunStatus}</div>
          </div>
        `;

        new maplibregl.Popup()
            .setLngLat(e.lngLat)
            .setHTML(popupHtml)
            .addTo(map);
    });

    map.on("mouseenter", "cafes-layer", () => {
        map.getCanvas().style.cursor = "pointer";
    });

    map.on("mouseleave", "cafes-layer", () => {
        map.getCanvas().style.cursor = "";
    });

    // ShadeMap setup
    const baseDate = new Date();
    baseDate.setHours(12, 0, 0, 0);

    shadeMap = new ShadeMap({
        apiKey: shadeMapKey,
        date: baseDate,
        color: "#111111",
        opacity: 0.7,
        terrainSource: {
            tileSize: 256,
            maxZoom: 15,
            getSourceUrl: ({ x, y, z }) =>
                `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`,
            getElevation: ({ r, g, b }) => r * 256 + g + b / 256 - 32768,
        },
        getFeatures: () => {
            const features = map
                .querySourceFeatures("openmaptiles", {
                    sourceLayer: "building",
                })
                .filter(
                    (f) =>
                        f.properties &&
                        (f.properties.height || f.properties.render_height)
                );

            features.forEach((f) => {
                if (!f.properties.height) f.properties.height = 3.5;
            });

            features.sort((a, b) => a.properties.height - b.properties.height);
            return features;
        },
        debug: (msg) => console.log("[ShadeMap]", msg),
    });

    // Add ShadeMap to map and set up update handler
    shadeMap.addTo(map);

    // Update shadow layer when time changes
    shadeMap.on("update", () => {
        const shadowFeatures = shadeMap.getShadowFeatures();
        map.getSource("shadow-source").setData({
            type: "FeatureCollection",
            features: shadowFeatures,
        });
        // Wait for the next frame to ensure the layer is updated
        requestAnimationFrame(() => {
            updateCafeSunStatus();
        });
    });

    // Function to update cafe sun status
    function updateCafeSunStatus() {
        if (!map.getLayer("shadow-layer")) {
            console.warn("Shadow layer not ready yet");
            return;
        }

        const currentHour = parseInt(
            document.getElementById("timeSlider").value
        );
        const currentDate = new Date();
        currentDate.setHours(currentHour, 0, 0, 0);

        const sunPosition = SunCalc.getPosition(
            currentDate,
            SKOPJE_COORDS.lat,
            SKOPJE_COORDS.lon
        );
        const isDaytime = sunPosition.altitude > 0;

        cafeFeatures.forEach((feature, index) => {
            const [lng, lat] = feature.geometry.coordinates;

            // Check if the point intersects with the shadow layer
            const isInShadow =
                map.queryRenderedFeatures(map.project([lng, lat]), {
                    layers: ["shadow-layer"],
                }).length > 0;

            feature.properties.in_sun = isDaytime && !isInShadow;
        });

        // Update the source data
        map.getSource("cafes").setData({
            type: "FeatureCollection",
            features: cafeFeatures,
        });

        // Update stats
        const sunnyCount = cafeFeatures.filter(
            (f) => f.properties.in_sun
        ).length;
        const shadowCount = cafeFeatures.filter(
            (f) => !f.properties.in_sun
        ).length;
        document.getElementById("sunnyCount").textContent = sunnyCount;
        document.getElementById("shadowCount").textContent = shadowCount;
    }

    // Time slider logic
    const timeSlider = document.getElementById("timeSlider");
    const timeValue = document.getElementById("timeValue");

    timeSlider.addEventListener("input", () => {
        const hour = parseInt(timeSlider.value);
        timeValue.textContent = `${hour.toString().padStart(2, "0")}:00`;
        const newDate = new Date();
        newDate.setHours(hour, 0, 0, 0);
        shadeMap.setDate(newDate);
    });

    // Filter controls
    document.getElementById("showSunny").addEventListener("change", (e) => {
        const visibility = e.target.checked ? "visible" : "none";
        map.setLayoutProperty("cafe-symbols", "visibility", visibility);
        updateCafeSunStatus();
    });

    document
        .getElementById("showShadow")
        .addEventListener("change", (e) => {
            const visibility = e.target.checked ? "visible" : "none";
            map.setLayoutProperty("cafe-symbols", "visibility", visibility);
            updateCafeSunStatus();
        });

    // Search functionality
    document
        .getElementById("searchInput")
        .addEventListener("input", (e) => {
            const searchTerm = e.target.value.toLowerCase();
            const features = map.querySourceFeatures("cafes");

            features.forEach((feature) => {
                const name = feature.properties.name?.toLowerCase() || "";
                const street =
                    feature.properties["addr:street"]?.toLowerCase() || "";
                const isVisible =
                    name.includes(searchTerm) || street.includes(searchTerm);

                map.setFeatureState(
                    { source: "cafes", id: feature.id },
                    { visible: isVisible }
                );
            });
        });

    // Initial update after a short delay to ensure layers are ready
    setTimeout(() => {
        updateCafeSunStatus();
    }, 1000);
});
