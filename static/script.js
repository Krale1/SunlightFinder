const maptilerKey = "4CHvvrHJ9EnxPMxYto8d";
const shadeMapKey =
  "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6ImtyaXN0aWphbmtyYWxldnNraTRAZ21haWwuY29tIiwiY3JlYXRlZCI6MTc0ODUyNzM1NTM1NSwiaWF0IjoxNzQ4NTI3MzU1fQ.YFiMN3q0ShAoUYa2j9MHhNde_Qj6bidmdQ91HxjCgBQ";
const WEATHER_API_KEY = "8a48baf68292fe3068c6cee9b99cf24b";
const SKOPJE_COORDS = { lat: 41.998, lon: 21.435 };

// Add error handling for map initialization
let map;
try {
  map = new maplibregl.Map({
    container: "map",
    style: `https://api.maptiler.com/maps/streets/style.json?key=${maptilerKey}`,
    center: [21.435, 41.998],
    zoom: 16,
    attributionControl: true
  });

  // Add navigation controls
  map.addControl(new maplibregl.NavigationControl(), 'top-right');

  // Add error handling for map load
  map.on('error', (e) => {
    console.error('Map error:', e);
  });

  map.on('load', async () => {
    console.log('Map loaded successfully');
    try {
      const response = await fetch("/cafes.geojson");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const cafes = await response.json();
      console.log('Cafes loaded:', cafes.features.length);
      cafeFeatures = cafes.features;
      document.getElementById("cafeCount").textContent = cafeFeatures.length;

      // Add terrain source
      map.addSource('terrain-source', {
        type: 'raster-dem',
        tiles: [
          'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'
        ],
        tileSize: 256,
        maxzoom: 15
      });

      // Add terrain layer
      //map.addLayer({
        //'id': 'terrain',
        //'type': 'hillshade',
        // 'source': 'terrain-source',
        // 'paint': {
        //   'hillshade-exaggeration': 0.5
        // }
     // });

      map.addSource("shadow-source", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
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
          "circle-color": "#ff6600",
          "circle-stroke-width": 2,
          "circle-stroke-color": "#fff",
        },
      });

      map.addLayer({
        id: "cafe-symbols",
        type: "symbol",
        source: "cafes",
        layout: {
          "text-field": "☕",
          "text-size": 14,
          "text-offset": [0, -1.5],
          "text-allow-overlap": true,
        },
      });

      map.on("click", "cafes-layer", (e) => {
        const feature = e.features[0];
        const name = feature.properties.name || "Unknown Café";
        const street = feature.properties["addr:street"] || "Unknown street";

        const popupHtml = `
          <div class="cafe-popup">
            <h3>${name}</h3>
            <p>${street}</p>
          </div>`;

        new maplibregl.Popup().setLngLat(e.lngLat).setHTML(popupHtml).addTo(map);
      });

      map.on("mouseenter", "cafes-layer", () => map.getCanvas().style.cursor = "pointer");
      map.on("mouseleave", "cafes-layer", () => map.getCanvas().style.cursor = "");

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

      shadeMap.addTo(map);

      shadeMap.on("update", () => {
        const shadowFeatures = shadeMap.getShadowFeatures();
        map.getSource("shadow-source").setData({
          type: "FeatureCollection",
          features: shadowFeatures,
        });
      });

    } catch (error) {
      console.error('Error loading cafes:', error);
    }
  });

} catch (error) {
  console.error('Error initializing map:', error);
}

function updateWeather() {
  document.getElementById("weatherDesc").textContent = "Loading...";
  document.getElementById("weatherIcon").textContent = "⏳";

  fetch(
    `https://api.openweathermap.org/data/2.5/weather?lat=${SKOPJE_COORDS.lat}&lon=${SKOPJE_COORDS.lon}&units=metric&appid=${WEATHER_API_KEY}`
  )
    .then((res) => res.json())
    .then((data) => {
      document.getElementById("currentTemp").textContent = `${Math.round(data.main.temp)}°C`;
      document.getElementById("weatherDesc").textContent = data.weather[0].description;
      document.getElementById("feelsLike").textContent = `${Math.round(data.main.feels_like)}°C`;
      document.getElementById("windSpeed").textContent = `${Math.round(data.wind.speed * 3.6)} km/h`;
      document.getElementById("humidity").textContent = `${data.main.humidity}%`;
      document.getElementById("uvIndex").textContent = data.uvi || "--";

      const weatherCode = data.weather[0].id;
      let weatherIcon = "⛅";
      if (weatherCode >= 200 && weatherCode < 300) weatherIcon = "⛈️";
      else if (weatherCode >= 300 && weatherCode < 600) weatherIcon = "🌧️";
      else if (weatherCode >= 600 && weatherCode < 700) weatherIcon = "❄️";
      else if (weatherCode >= 700 && weatherCode < 800) weatherIcon = "🌫️";
      else if (weatherCode === 800) weatherIcon = "☀️";
      else if (weatherCode > 800) weatherIcon = "☁️";

      document.getElementById("weatherIcon").textContent = weatherIcon;
      document.getElementById("weatherUpdate").textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
      document.getElementById("lastUpdate").textContent = new Date().toLocaleString();
    })
    .catch((error) => {
      console.error('Error fetching weather:', error);
      document.getElementById("weatherDesc").textContent = "Weather data unavailable";
      document.getElementById("weatherIcon").textContent = "⚠️";
    });
}

updateWeather();
setInterval(updateWeather, 30 * 60 * 1000);

const timeSlider = document.getElementById("timeSlider");
const timeValue = document.getElementById("timeValue");

timeSlider.addEventListener("input", () => {
  const hour = parseInt(timeSlider.value);
  timeValue.textContent = `${hour.toString().padStart(2, "0")}:00`;
  const newDate = new Date();
  newDate.setHours(hour, 0, 0, 0);
  if (shadeMap) {
    shadeMap.setDate(newDate);
  }
});

document.getElementById("searchInput").addEventListener("input", (e) => {
  const term = e.target.value.toLowerCase();
  const source = map.getSource("cafes");
  const filtered = cafeFeatures.filter((f) => {
    const name = f.properties.name?.toLowerCase() || "";
    const street = f.properties["addr:street"]?.toLowerCase() || "";
    return name.includes(term) || street.includes(term);
  });
  source.setData({ type: "FeatureCollection", features: filtered });
  document.getElementById("cafeCount").textContent = filtered.length;
});

// Handle the "Find Best Spots" button click
document.getElementById('find-sunlight').addEventListener('click', () => {
    const time = document.getElementById('time').value;
    const date = document.getElementById('date').value;
    
    console.log('Finding best spots for:', date, time);
});

// Set default date to today
document.getElementById('date').valueAsDate = new Date();
