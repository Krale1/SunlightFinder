const maptilerKey = "4CHvvrHJ9EnxPMxYto8d";
const shadeMapKey = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6ImtyaXN0aWphbmtyYWxldnNraTRAZ21haWwuY29tIiwiY3JlYXRlZCI6MTc0ODUyNzM1NTM1NSwiaWF0IjoxNzQ4NTI3MzU1fQ.YFiMN3q0ShAoUYa2j9MHhNde_Qj6bidmdQ91HxjCgBQ";
const WEATHER_API_KEY = "8a48baf68292fe3068c6cee9b99cf24b";
const SKOPJE_COORDS = { lat: 41.998, lon: 21.435 };

let map;
let shadeMap;
let cafeFeatures = [];
let emojiMarkers = [];

map = new maplibregl.Map({
  container: "map",
  style: `https://api.maptiler.com/maps/streets/style.json?key=${maptilerKey}`,
  center: [21.435, 41.998],
  zoom: 16,
  attributionControl: true
});

map.addControl(new maplibregl.NavigationControl(), 'top-right');

map.on('load', async () => {
  const response = await fetch("/cafes.geojson");
  const cafes = await response.json();
  cafeFeatures = cafes.features;
  document.getElementById("cafeCount").textContent = cafeFeatures.length;

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
        .filter((f) => f.properties && (f.properties.height || f.properties.render_height));

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

  updateCafesWithSunlight(parseInt(timeSlider.value));
});

// Add debounce function at the top level
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Replace the timeSlider event listener with a debounced version
const debouncedUpdate = debounce((hour) => {
  timeValue.textContent = `${hour.toString().padStart(2, "0")}:00`;

  const newDate = new Date();
  newDate.setHours(hour, 0, 0, 0);
  if (shadeMap) {
    shadeMap.setDate(newDate);
  }

  updateCafesWithSunlight(hour);
}, 150); // 150ms debounce time

timeSlider.addEventListener("input", () => {
  const hour = parseInt(timeSlider.value);
  debouncedUpdate(hour);
});

// Add a function to create a marker with opacity transition
function createMarkerWithTransition(emoji, lat, lng, popupContent) {
  const el = document.createElement('div');
  el.className = 'emoji-marker';
  el.textContent = emoji;
  el.style.fontSize = '20px';
  el.style.transform = 'translate(-50%, -50%)';
  el.style.position = 'absolute';
  el.style.opacity = '0'; // Start with opacity 0

  const marker = new maplibregl.Marker(el)
    .setLngLat([lng, lat])
    .setPopup(new maplibregl.Popup().setHTML(popupContent))
    .addTo(map);

  // Fade in the marker
  setTimeout(() => {
    el.style.transition = 'opacity 0.3s ease-in-out';
    el.style.opacity = '1';
  }, 50);

  return marker;
}

async function updateCafesWithSunlight(hour) {
  const showSunny = document.getElementById("showSunny").checked;
  const showShadow = document.getElementById("showShadow").checked;
  const isNighttime = hour >= 21 || hour <= 5;

  emojiMarkers.forEach(marker => {
    const el = marker.getElement();
    el.style.transition = 'opacity 0.2s ease-out';
    el.style.opacity = '0';
  });
  setTimeout(async () => {
    emojiMarkers.forEach(marker => marker.remove());
    emojiMarkers = [];

    if (isNighttime) {
      cafeFeatures.forEach((feature) => {
        if (!showShadow) return;
        
        const lat = feature.geometry.coordinates[1];
        const lng = feature.geometry.coordinates[0];
        const popupContent = `<strong>${feature.properties.name}</strong><br>🌚 (Nighttime)`;
        
        const marker = createMarkerWithTransition('🌚', lat, lng, popupContent);
        emojiMarkers.push(marker);
      });
    } else {
      try {
        const cafesPayload = cafeFeatures.map((feature) => ({
          lat: feature.geometry.coordinates[1],
          lng: feature.geometry.coordinates[0],
          name: feature.properties.name || '',
          outdoor: feature.properties.outdoor_seating || 'false',
          orientation: feature.properties.bar_orientation || null
        }));

        const res = await fetch('/predict_batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ hour, cafes: cafesPayload })
        });

        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }

        const predictions = await res.json();
        
        if (!Array.isArray(predictions)) {
          throw new Error('Invalid response format from server');
        }

        predictions.forEach((data, i) => {
          const feature = cafeFeatures[i];
          if (!feature) return;
          
          const lat = feature.geometry.coordinates[1];
          const lng = feature.geometry.coordinates[0];
          const emoji = data.emoji || '❓';
          const isSunny = emoji === '☀️' || emoji === '🌞';
          const displayEmoji = isSunny ? '🌞' : '🌚';

          if ((isSunny && showSunny) || (!isSunny && showShadow)) {
            const popupContent = `<strong>${feature.properties.name}</strong><br>${displayEmoji}`;
            const marker = createMarkerWithTransition(displayEmoji, lat, lng, popupContent);
            emojiMarkers.push(marker);
          }
        });

      } catch (err) {
        console.error('Batch prediction error:', err);
        cafeFeatures.forEach((feature) => {
          const lat = feature.geometry.coordinates[1];
          const lng = feature.geometry.coordinates[0];
          const popupContent = `<strong>${feature.properties.name}</strong><br>⚠️ Error loading data`;
          const marker = createMarkerWithTransition('⚠️', lat, lng, popupContent);
          emojiMarkers.push(marker);
        });
      }
    }
  }, 200);
}

document.getElementById("showSunny").addEventListener("change", () => {
  updateCafesWithSunlight(parseInt(timeSlider.value));
});

document.getElementById("showShadow").addEventListener("change", () => {
  updateCafesWithSunlight(parseInt(timeSlider.value));
});

function updateWeather() {
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
