/**
 * EcoSim - Ecological Succession Simulator
 * Core Logic & GPU Accelerators
 */

const CONFIG = {
    gridWidth: 40,
    gridHeight: 30,
    numSpecies: 5,
    speciesNames: ['地衣', '苔藓', '草本', '灌木', '乔木'],
    colors: [
        '#6E7C88', // Lichen
        '#599F2F', // Moss
        '#A3B18A', // Herb (Inferred)
        '#468843', // Shrub
        '#1B4332'  // Tree (Forest)
    ],
    bgSeed: [0.2, 0.1, 0.0005, 0.0001, 0.00005], // 极低的高等种背景输入
    initialSeed: [1.0, 0.2, 0.0, 0.0, 0.0], 
    // Parameters from CSV
    rho: [30.0, 100.0, 50.0, 200.0, 80.0], // 降低草本产种量 (150 -> 50)
    lambda: [0.5, 0.8, 1.5, 3.0, 6.0],
    R_max: 5.0,
    r: [0.3, 0.4, 0.3, 0.6, 0.2], // 降低草本生长率 (0.6 -> 0.3)
    g: [0.15, 0.2, 0.6, 0.4, 0.2],
    s: [0.6, 0.6, 0.2, 0.6, 0.7],
    nu: [0.045, 0.06, 0.18, 0.12, 0.06],
    K_base: [0.3, 0.4, 0.7, 0.8, 0.95], // 高等物种拥有更高的环境容纳量
    alpha: [
        [1.0, 1.15, 2.0, 2.5, 3.3], // 地衣被所有物种强烈抑制（尤其是乔木遮荫）
        [0.4, 1.0, 1.5, 2.0, 2.8], // 苔藓受更高物种抑制
        [0.2, 0.3, 1.0, 1.5, 2.3], // 草本受灌木/乔木抑制
        [0.1, 0.1, 0.2, 1.0, 1.6], // 灌木受乔木抑制
        [0.01, 0.01, 0.05, 0.1, 1.0] // 乔木几乎不受早期演替种的影响
    ],

    soilMult: [
        [1.0, 0.8, 0.6],  // Lichen
        [0.9, 1.0, 0.8],  // Moss
        [0.4, 0.9, 1.0],  // Herb
        [0.1, 0.7, 1.0],  // Shrub
        [0.02, 0.4, 1.0]  // Tree
    ]
};

class EcoSimulator {
    constructor() {
        try {
            this.gpu = typeof GPU.GPU === 'function' ? new GPU.GPU() : new GPU();
        } catch (e) {
            console.error("GPU.js initialization failed, falling back to CPU mode:", e);
            this.gpu = null;
        }
        this.width = CONFIG.gridWidth;
        this.height = CONFIG.gridHeight;
        this.numSpecies = CONFIG.numSpecies;

        this.biomass = new Float32Array(this.width * this.height * this.numSpecies);
        this.seedBank = new Float32Array(this.width * this.height * this.numSpecies);
        this.soilDepth = new Int32Array(this.width * this.height);

        this.initGrid();
        this.initKernels();
    }

    initGrid() {
        for (let i = 0; i < this.width * this.height; i++) {
            this.soilDepth[i] = Math.floor(Math.random() * 3);
        }
        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                const idx = (y * this.width + x) * this.numSpecies;
                for (let k = 0; k < this.numSpecies; k++) {
                    this.seedBank[idx + k] = CONFIG.initialSeed[k];
                    this.biomass[idx + k] = 0;
                }
                if (Math.random() < 0.5) { // 提高分布比例
                    this.biomass[idx + 0] = Math.random() * 0.2 + 0.1; // 提高初始盖度
                }
            }
        }
    }

    initKernels() {
        // Precompute Gaussian kernels for each species
        this.precomputedKernels = CONFIG.lambda.map(l => {
            const r = Math.ceil(l * 2);
            const size = r * 2 + 1;
            const data = new Float32Array(size * size);
            for (let dy = -r; dy <= r; dy++) {
                for (let dx = -r; dx <= r; dx++) {
                    const distSq = dx * dx + dy * dy;
                    data[(dy + r) * size + (dx + r)] = Math.exp(-distSq / (2 * l * l));
                }
            }
            return { r, size, data };
        });
    }

    step() {
        const climateMults = {
            '热带雨林气候': [1.0, 1.0, 1.0, 1.0, 1.0],
            '温带草原气候': [1.0, 1.0, 1.0, 0.2, 0.0],
            '寒带苔原气候': [1.0, 1.0, 0.1, 0.0, 0.0],
            '温带落叶林气候': [0.8, 0.8, 0.8, 0.8, 0.8],
            '荒漠气候': [1.0, 0.2, 0.05, 0.0, 0.0]
        };
        const currentMults = climateMults[this.currentClimate] || climateMults['热带雨林气候'];
        this.stepCPU(currentMults);
    }

    stepCPU(climateMults) {
        const nextB = new Float32Array(this.biomass.length);
        const nextS = new Float32Array(this.seedBank.length);
        const w = this.width;
        const h = this.height;
        const ns = this.numSpecies;

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = (y * w + x) * ns;
                let totalB = 0;
                for (let k = 0; k < ns; k++) totalB += this.biomass[idx + k];

                for (let k = 0; k < ns; k++) {
                    let sk = this.seedBank[idx + k] + CONFIG.bgSeed[k];
                    
                    // Optimized dispersal using precomputed kernel
                    const kernelInfo = this.precomputedKernels[k];
                    const r = kernelInfo.r;
                    const kSize = kernelInfo.size;
                    const kData = kernelInfo.data;
                    let dispersed = 0;
                    
                    for (let dy = -r; dy <= r; dy++) {
                        const ny = (y + dy + h) % h;
                        for (let dx = -r; dx <= r; dx++) {
                            const nx = (x + dx + w) % w;
                            const nIdx = (ny * w + nx) * ns;
                            const weight = kData[(dy + r) * kSize + (dx + r)];
                            dispersed += this.biomass[nIdx + k] * CONFIG.rho[k] * weight * 0.01;
                        }
                    }
                    sk += dispersed;

                    const depth = this.soilDepth[y * w + x];
                    const k_local = CONFIG.K_base[k] * CONFIG.soilMult[k][depth] * climateMults[k];
                    let compSum = 0;
                    for (let l = 0; l < ns; l++) compSum += CONFIG.alpha[k][l] * this.biomass[idx + l];
                    
                    let b_growth = this.biomass[idx + k] + CONFIG.r[k] * climateMults[k] * this.biomass[idx + k] * (1 - compSum / Math.max(0.01, k_local));
                    
                    // Step C: Germination with Succession Thresholds
                    // 极大幅提高门槛：先锋种必须积累足够生物量 (象征土壤有机质积累)
                    const thresholds = [0.0, 0.1, 0.4, 0.73, 0.85]; 
                    let g_eff = CONFIG.g[k] * Math.max(0, 1 - totalB);
                    if (totalB < thresholds[k]) g_eff = 0; // Succession barrier: blocked if soil not ready

                    const b_germ = Math.min(sk * g_eff * 0.1 * climateMults[k], Math.max(0, 1 - totalB));
                    
                    nextB[idx + k] = Math.max(0, b_growth + b_germ);
                    nextS[idx + k] = Math.max(0, (sk - b_germ) * CONFIG.s[k]);
                }

                let finalTotalB = 0;
                for (let k = 0; k < ns; k++) finalTotalB += nextB[idx + k];
                if (finalTotalB > 1.0) {
                    for (let k = 0; k < ns; k++) nextB[idx + k] /= finalTotalB;
                }
            }
        }
        this.biomass.set(nextB);
        this.seedBank.set(nextS);
    }

    applyDisturbance(centerX, centerY, radius, deathRates, seedDeath) {
        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                const distSq = (x - centerX) ** 2 + (y - centerY) ** 2;
                if (distSq <= radius * radius) {
                    const idx = (y * this.width + x) * this.numSpecies;
                    for (let k = 0; k < this.numSpecies; k++) {
                        this.biomass[idx + k] *= (1 - deathRates[k]);
                        this.seedBank[idx + k] *= (1 - seedDeath);
                    }
                }
            }
        }
    }
}

const { createApp, ref, onMounted, reactive } = Vue;

createApp({
    setup() {
        const simulator = ref(null);
        const canvasRef = ref(null);
        const chartRef = ref(null);
        let chart = null;

        const state = reactive({
            running: false,
            year: 0,
            speed: 5,
            climate: '热带雨林气候',
            hoverData: null
        });

        const climates = ['热带雨林气候', '温带草原气候', '寒带苔原气候', '温带落叶林气候', '荒漠气候'];

        const init = () => {
            simulator.value = new EcoSimulator();
            state.year = 0;
            state.running = false;
            Vue.nextTick(() => {
                initChart();
                render();
            });
        };

        const togglePlay = () => {
            state.running = !state.running;
            if (state.running) loop();
        };

        const reset = () => init();

        let lastTime = 0;
        const loop = (time) => {
            if (!state.running || !simulator.value) return;
            
            // Limit logic steps per frame for performance
            simulator.value.currentClimate = state.climate;
            const stepsPerFrame = Math.min(state.speed, 5);
            for(let i=0; i < stepsPerFrame; i++) {
                simulator.value.step();
                state.year++;
                updateChart();
            }
            
            render();
            requestAnimationFrame(loop);
        };

        const hexToRgb = (hex) => {
            const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
            return result ? {
                r: parseInt(result[1], 16),
                g: parseInt(result[2], 16),
                b: parseInt(result[3], 16)
            } : { r: 0, g: 0, b: 0 };
        };

        const speciesRgb = CONFIG.colors.map(hex => hexToRgb(hex));

        const render = () => {
            const canvas = canvasRef.value;
            if (!canvas || !simulator.value) return;
            const ctx = canvas.getContext('2d');
            
            const width = CONFIG.gridWidth;
            const height = CONFIG.gridHeight;
            const cw = canvas.width / width;
            const ch = canvas.height / height;

            const imageData = ctx.createImageData(canvas.width, canvas.height);
            const data = imageData.data;

            for (let y = 0; y < canvas.height; y++) {
                const gy = Math.floor(y / ch);
                for (let x = 0; x < canvas.width; x++) {
                    const gx = Math.floor(x / cw);
                    const idx = (gy * width + gx) * CONFIG.numSpecies;
                    const pixelIdx = (y * canvas.width + x) * 4;

                    let r = 17, g = 17, b = 17;
                    let totalB = 0;

                    for (let k = 0; k < CONFIG.numSpecies; k++) {
                        const biomass = simulator.value.biomass[idx + k];
                        if (biomass > 0.01) {
                            const c = speciesRgb[k];
                            r += (c.r - 17) * biomass;
                            g += (c.g - 17) * biomass;
                            b += (c.b - 17) * biomass;
                            totalB += biomass;
                        }
                    }

                    data[pixelIdx] = Math.min(255, r);
                    data[pixelIdx + 1] = Math.min(255, g);
                    data[pixelIdx + 2] = Math.min(255, b);
                    data[pixelIdx + 3] = 255;
                }
            }
            ctx.putImageData(imageData, 0, 0);
        };

        const handleCanvasClick = (e) => {
            if (!simulator.value) return;
            const rect = canvasRef.value.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) / (rect.width / CONFIG.gridWidth));
            const y = Math.floor((e.clientY - rect.top) / (rect.height / CONFIG.gridHeight));
            simulator.value.applyDisturbance(x, y, 5, [0, 0, 0.95, 0.6, 0.5], 0.5);
            render();
        };

        const handleMouseMove = (e) => {
            if (!simulator.value) return;
            const rect = canvasRef.value.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) / (rect.width / CONFIG.gridWidth));
            const y = Math.floor((e.clientY - rect.top) / (rect.height / CONFIG.gridHeight));
            
            if (x >= 0 && x < CONFIG.gridWidth && y >= 0 && y < CONFIG.gridHeight) {
                const idx = (y * CONFIG.gridWidth + x) * CONFIG.numSpecies;
                state.hoverData = {
                    x, y,
                    soil: ['浅', '中', '深'][simulator.value.soilDepth[y * CONFIG.gridWidth + x]],
                    species: CONFIG.speciesNames.map((name, i) => ({
                        name,
                        val: (simulator.value.biomass[idx + i] * 100).toFixed(1) + '%'
                    }))
                };
            } else state.hoverData = null;
        };

        const initChart = () => {
            const canvas = chartRef.value;
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (chart) chart.destroy();
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: CONFIG.speciesNames.map((name, i) => ({
                        label: name,
                        data: [],
                        borderColor: CONFIG.colors[i],
                        tension: 0.1,
                        pointRadius: 0
                    }))
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        y: { beginAtZero: true, max: 100, grid: { color: '#333' }, ticks: { color: '#888' } },
                        x: { grid: { display: false }, ticks: { color: '#888' } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        };

        const updateChart = () => {
            if (state.year % 10 !== 0 || !simulator.value || !chart) return;
            const totals = new Array(CONFIG.numSpecies).fill(0);
            const count = CONFIG.gridWidth * CONFIG.gridHeight;
            for (let i = 0; i < count; i++) {
                for (let k = 0; k < CONFIG.numSpecies; k++) {
                    totals[k] += simulator.value.biomass[i * CONFIG.numSpecies + k];
                }
            }
            chart.data.labels.push(state.year);
            totals.forEach((t, i) => chart.data.datasets[i].data.push((t / count) * 100));
            if (chart.data.labels.length > 50) {
                chart.data.labels.shift();
                chart.data.datasets.forEach(d => d.data.shift());
            }
            chart.update('none');
        };

        const applyDisturbance = (type) => {
            if (!simulator.value) return;
            const cx = CONFIG.gridWidth / 2;
            const cy = CONFIG.gridHeight / 2;
            if (type === 'fire') simulator.value.applyDisturbance(cx, cy, 15, [0, 0, 0.95, 0.6, 0.9], 0.5);
            else if (type === 'volcano') simulator.value.applyDisturbance(cx, cy, 20, [1, 1, 1, 1, 1], 1);
            else if (type === 'drought') simulator.value.applyDisturbance(cx, cy, 40, [0.1, 0.2, 0.5, 0.2, 0.6], 0.1);
            render();
        };

        onMounted(() => init());

        return {
            simulator, canvasRef, chartRef, state, climates,
            togglePlay, reset, handleCanvasClick, handleMouseMove, applyDisturbance,
            CONFIG
        };
    }
}).mount('#app');
