// Publication color manager with colors for publications from zinepub_metadata.csv
const publicationColorManager = {
    // Colors for publications from zinepub_metadata.csv
    // Format: "publication_name": { color: "#hexcolor", pub_id: "id" }
    publications: {
        "Ain't I A Woman?": { color: "#e74c3c", pub_id: "1" },
        "All She Wrote": { color: "#3498db", pub_id: "2" },
        "All Together Journal": { color: "#2ecc71", pub_id: "3" },
        "At the Foot of the Mountain": { color: "#9b59b6", pub_id: "4" },
        "Atalanta": { color: "#f39c12", pub_id: "5" },
        "Aurora": { color: "#1abc9c", pub_id: "6" },
        "Awake & Move": { color: "#e67e22", pub_id: "7" },
        "Belles Lettres": { color: "#34495e", pub_id: "8" },
        "Berkshire Women's News": { color: "#8e44ad", pub_id: "9" },
        "Between Our Selves": { color: "#27ae60", pub_id: "10" },
        "Big Mamma Rag": { color: "#ff6b35", pub_id: "11" },
        "Bitch": { color: "#9013fe", pub_id: "12" },
        "Blue Lantern": { color: "#4caf50", pub_id: "13" },
        "Bottomfish Blues": { color: "#ff5722", pub_id: "14" },
        "Boxcar": { color: "#607d8b", pub_id: "15" },
        "Breakthrough": { color: "#795548", pub_id: "16" },
        "Catalyst": { color: "#ffc107", pub_id: "17" },
        "Change": { color: "#e91e63", pub_id: "18" },
        "Change is Gonna Come": { color: "#00bcd4", pub_id: "19" },
        "Changing Woman": { color: "#ffeb3b", pub_id: "20" },
        "Clarion": { color: "#673ab7", pub_id: "21" },
        "Common Ground": { color: "#009688", pub_id: "22" },
        "Common Woman": { color: "#ff9800", pub_id: "23" },
        "Connections": { color: "#3f51b5", pub_id: "24" },
        "Coyote Howls": { color: "#8bc34a", pub_id: "25" },
        "Cries from Cassandra": { color: "#f44336", pub_id: "26" },
        "Distaff": { color: "#9c27b0", pub_id: "27" },
        "Do It NOW": { color: "#2196f3", pub_id: "28" },
        "Gold Flower": { color: "#cddc39", pub_id: "29" },
        
        // Default for unknown publications
        "Unknown": { color: "#95a5a6", pub_id: null }
    },

    // Get color by publication name
    getColor: function(publicationName) {
        if (this.publications[publicationName]) {
            return this.publications[publicationName].color;
        }
        return this.publications["Unknown"].color;
    },

    // Get color by pub_id
    getColorById: function(pubId) {
        for (const [name, data] of Object.entries(this.publications)) {
            if (data.pub_id === pubId) {
                return data.color;
            }
        }
        return this.publications["Unknown"].color;
    },

    // Get publication name by pub_id
    getNameById: function(pubId) {
        for (const [name, data] of Object.entries(this.publications)) {
            if (data.pub_id === pubId) {
                return name;
            }
        }
        return "Unknown";
    },

    // Get pub_id by publication name
    getPubId: function(publicationName) {
        if (this.publications[publicationName]) {
            return this.publications[publicationName].pub_id;
        }
        return null;
    },

    // Get all publication colors used in dataset
    getPublicationColors: function(publications) {
        const colors = {};
        publications.forEach(pub => {
            colors[pub] = this.getColor(pub);
        });
        return colors;
    },

    // Populate legend with publication colors
    populateLegend: function(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = '';
        
        // Get unique publications from current data
        const publications = this.getCurrentPublications();
        
        publications.forEach(pub => {
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            legendItem.innerHTML = `
                <span class="legend-color" style="background-color: ${this.getColor(pub)}"></span>
                <span class="legend-label">${pub}</span>
            `;
            container.appendChild(legendItem);
        });
    },

    // Get current publications from global data
    getCurrentPublications: function() {
        if (typeof globalMetadata !== 'undefined' && globalMetadata) {
            return [...new Set(globalMetadata.map(d => d.publication_name).filter(Boolean))];
        }
        return Object.keys(this.publications).filter(name => name !== "Unknown");
    }
};

// Reuse type colors and manager
const reuseTypeColorManager = {
    colors: {
        "minimal_reuse": "#95a5a6",     // Light gray
        "partial_reuse": "#f39c12",     // Orange  
        "moderate_reuse": "#e74c3c",    // Red
        "high_reuse": "#8e44ad",        // Purple
        "extensive_reuse": "#2c3e50"    // Dark blue
    },

    getColor: function(reuseType) {
        return this.colors[reuseType] || "#bdc3c7"; // Default light gray
    },

    populateLegend: function(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = '';
        
        Object.entries(this.colors).forEach(([type, color]) => {
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            legendItem.innerHTML = `
                <span class="legend-color" style="background-color: ${color}"></span>
                <span class="legend-label">${type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
            `;
            container.appendChild(legendItem);
        });
    }
};

// Helper function to get reuse type color (for backward compatibility)
function getReuseTypeColor(reuseType) {
    return reuseTypeColorManager.getColor(reuseType);
}

// Make sure managers are available globally
window.publicationColorManager = publicationColorManager;
window.reuseTypeColorManager = reuseTypeColorManager;