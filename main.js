/**
 * EcoSim - Ecological Succession Simulator
 * Core Logic & GPU Accelerators
 */

const CONFIG = {
    gridWidth: 20,              // 网格宽度 (格子数)
    gridHeight: 13,             // 网格高度 (格子数)
    numSpecies: 5,              // 功能型物种总数 (地衣, 苔藓, 草本, 灌木, 乔木)
    speciesNames: ['地衣', '苔藓', '草本', '灌木', '乔木'],
    colors: [                   // 各物种在地图上显示的颜色 (十六进制)
        '#6E7C88', // 地衣 (先锋种)
        '#599F2F', // 苔藓 (地被种)
        '#60e166', // 草本 (竞争种)
        '#468843', // 灌木 (过渡种)
        '#1B4332'  // 乔木 (顶极种)
    ],
    bgSeed: [0.002, 0.0000001, 0.000005, 0.0000001, 0.0000005], // 背景种子雨：每年随机落入格子的种子量 (模拟远距离传播)
    initialSeed: [0.01, 0.0, 0.0, 0.0, 0.0],                   // 初始种子库：t=0 时各格子的预置种子量
    
    // 物种生物学参数 (按 [地衣, 苔藓, 草本, 灌木, 乔木] 顺序排列)
    rho: [30.0, 100.0, 50.0, 200.0, 80.0],       // 繁殖力 (ρ)：单位盖度每年产生的种子数
    lambda: [0.5, 0.8, 1.5, 3.0, 6.0],           // 扩散距离 (λ)：种子扩散的标准差 (单位: 格子)
    R_max: 5.0,                                  // 最大扩散半径：计算扩散时的截断距离 (格子数)
    r: [0.3, 0.4, 0.3, 0.6, 0.2],                // 增长率 (r)：物种的最大年增长速度
    g: [0.15, 0.2, 0.6, 0.4, 0.2],               // 萌发率 (g)：种子在空白处成功萌发的概率
    s: [0.6, 0.6, 0.2, 0.6, 0.7],                // 种子库存活率 (s)：未萌发种子的年存活率
    nu: [0.045, 0.06, 0.18, 0.12, 0.06],         // 库萌发率 (ν)：种子库中种子后续萌发的概率
    K_base: [0.3, 0.4, 0.7, 0.8, 0.95],          // 基础承载力 (K)：物种能达到的最大理论盖度
    
    // 竞争矩阵 (alpha[k][l])：物种 l 对物种 k 的抑制系数
    // 横行受害者 k，纵列竞争者 l。数值越大抑制越强。
    alpha: [
        [1.0, 1.15, 2.0, 2.5, 3.3], // 地衣受其他物种抑制强 (遮荫效应)
        [0.4, 1.0, 1.5, 2.0, 2.8],  // 苔藓受更高阶物种抑制
        [0.2, 0.3, 1.0, 1.5, 2.3],  // 草本受灌木/乔木抑制
        [0.1, 0.1, 0.2, 1.0, 1.6],  // 灌木受乔木抑制
        [0.01, 0.01, 0.05, 0.1, 1.0] // 乔木 (顶极种) 几乎不受早期物种影响
    ],

    // 土壤响应乘子 (soilMult[k][depth])：不同土壤深度对 K 的修正系数
    // 深度索引: 0 (浅), 1 (中), 2 (深)
    soilMult: [
        [1.0, 0.8, 0.6],  // 地衣: 偏好浅土
        [0.9, 1.0, 0.8],  // 苔藓: 适应性广
        [0.4, 0.9, 1.0],  // 草本: 偏好深土
        [0.1, 0.7, 1.0],  // 灌木: 依赖深土
        [0.02, 0.4, 1.0]  // 乔木: 必须有深土
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
        // 环境随机性：模拟年度气候波动 (±15%)
        const yearlyFluctuation = 1.0 + (Math.random() * 0.3 - 0.15);
        
        const climateMults = {
            '热带雨林气候': [1.0, 1.0, 1.0, 1.0, 1.0],
            '温带草原气候': [1.0, 1.0, 1.0, 0.2, 0.0],
            '寒带苔原气候': [1.0, 1.0, 0.1, 0.0, 0.0],
            '温带落叶林气候': [0.8, 0.8, 0.8, 0.8, 0.8],
            '荒漠气候': [1.0, 0.2, 0.05, 0.0, 0.0]
        };
        const currentMults = (climateMults[this.currentClimate] || climateMults['热带雨林气候']).map(m => m * yearlyFluctuation);
        
        this.stepCPU(currentMults);

        // 随机小规模扰动：模拟自然林隙或局部灾害 (0.5% 概率)
        if (Math.random() < 0.005) {
            const rx = Math.floor(Math.random() * this.width);
            const ry = Math.floor(Math.random() * this.height);
            this.applyDisturbance(rx, ry, 3, [0.2, 0.3, 0.5, 0.7, 0.8], 0.2); // 越高级的物种受损越明显
        }
    }

    stepCPU(climateMults) {
        const nextB = new Float32Array(this.biomass.length);
        const nextS = new Float32Array(this.seedBank.length);
        const w = this.width;
        const h = this.height;
        const ns = this.numSpecies;

        // 预计算全局总生物量，用于全局扩散
        const globalBiomass = new Float32Array(ns);
        for (let i = 0; i < w * h * ns; i += ns) {
            for (let k = 0; k < ns; k++) globalBiomass[k] += this.biomass[i + k];
        }

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = (y * w + x) * ns;
                let totalB = 0;
                for (let k = 0; k < ns; k++) totalB += this.biomass[idx + k];

                for (let k = 0; k < ns; k++) {
                    // 随机背景种子输入 (±50% 波动)
                    const randRain = 0.5 + Math.random();
                    let sk = this.seedBank[idx + k] + CONFIG.bgSeed[k] * randRain;
                    
                    const kernelInfo = this.precomputedKernels[k];
                    let dispersed = 0;
                    
                    // 优化：对于扩散范围极大的物种（如乔木），使用全局平均扩散
                    if (kernelInfo.r >= Math.min(w, h) / 2) {
                        dispersed = (globalBiomass[k] * CONFIG.rho[k] * 0.01) / (w * h);
                    } else {
                        const r = kernelInfo.r;
                        const kSize = kernelInfo.size;
                        const kData = kernelInfo.data;
                        for (let dy = -r; dy <= r; dy++) {
                            const ny = (y + dy + h) % h;
                            for (let dx = -r; dx <= r; dx++) {
                                const nx = (x + dx + w) % w;
                                const nIdx = (ny * w + nx) * ns;
                                const sourceB = this.biomass[nIdx + k];
                                if (sourceB < 0.001) continue; // 剪枝：来源为空则跳过
                                
                                const weight = kData[(dy + r) * kSize + (dx + r)];
                                dispersed += sourceB * CONFIG.rho[k] * weight * 0.01;
                            }
                        }
                    }
                    sk += dispersed;

                    const depth = this.soilDepth[y * w + x];
                    const k_local = CONFIG.K_base[k] * CONFIG.soilMult[k][depth] * climateMults[k];
                    let compSum = 0;
                    for (let l = 0; l < ns; l++) compSum += CONFIG.alpha[k][l] * this.biomass[idx + l];
                    
                    let b_growth = this.biomass[idx + k] + CONFIG.r[k] * climateMults[k] * this.biomass[idx + k] * (1 - compSum / Math.max(0.01, k_local));
                    
                    const thresholds = [0.0, 0.1, 0.4, 0.73, 0.85]; 
                    let g_eff = CONFIG.g[k] * Math.max(0, 1 - totalB);
                    if (totalB < thresholds[k]) g_eff = 0;

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

            // 首先清空画布 (背景色 #111)
            ctx.fillStyle = '#111';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            for (let gy = 0; gy < height; gy++) {
                for (let gx = 0; gx < width; gx++) {
                    const idx = (gy * width + gx) * CONFIG.numSpecies;
                    
                    let r = 17, g = 17, b = 17; // 基准色 RGB(17, 17, 17)
                    let totalB = 0;

                    // 计算该格子的混合颜色
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

                    // 只有当生物量大于阈值时才绘制，减少 GPU 提交
                    if (totalB > 0.01) {
                        ctx.fillStyle = `rgb(${Math.min(255, r)}, ${Math.min(255, g)}, ${Math.min(255, b)})`;
                        // 直接绘制格子矩形
                        ctx.fillRect(gx * cw, gy * ch, cw, ch);
                        
                        // 绘制格子边框 (半透明，增加网格感)
                        ctx.strokeStyle = 'rgba(0,0,0,0.1)';
                        ctx.strokeRect(gx * cw, gy * ch, cw, ch);
                    }
                }
            }
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
            if (type === 'fire') simulator.value.applyDisturbance(cx, cy, 5, [0.2, 0.3, 0.95, 0.6, 0.9], 0.5);
            else if (type === 'volcano') simulator.value.applyDisturbance(cx, cy, 17, [1, 1, 1, 1, 1], 1);
            else if (type === 'drought') simulator.value.applyDisturbance(cx, cy, 40, [0.1, 0.2, 0.5, 0.2, 0.6], 0.1);
            render();
        };

        onMounted(() => {
            console.log("%c EcoSim v2.1 %c Built By LiuSonWood And YanYiLin ", "color: #fff; background: #1B4332; padding: 4px; border-radius: 4px 0 0 4px; font-weight: bold;", "color: #fff; background: #468843; padding: 4px; border-radius: 0 4px 4px 0;");
            init();
        });

        return {
            simulator, canvasRef, chartRef, state, climates,
            togglePlay, reset, handleCanvasClick, handleMouseMove, applyDisturbance,
            CONFIG
        };
    }
}).mount('#app');
