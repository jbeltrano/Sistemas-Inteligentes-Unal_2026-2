/*
 * JBVCAgent.js  (v2)
 * Agente de Gomoku (variante Swap 2) compatible con la clase Agent de gomoku.js.
 *
 * Motor:
 *   - Evaluación por ventanas (5 y 6 celdas) actualizada incrementalmente.
 *   - Detección O(1) de cuatros / casillas ganadoras.
 *   - Búsqueda alfa-beta (PVS) + tabla de transposición + LMR + búsqueda de quietud.
 *   - VCF / VCT (secuencias forzadas de cuatros y treses) para ataque y defensa.
 *   - Swap 2 completo: apertura equilibrada, decisión de color y opción de colocar 2 piedras.
 *   - Gestión del reloj ACUMULADO (el parámetro `time` es el tiempo total restante en ms)
 *     con tope por jugada < 10 s.
 *
 * Todo el cálculo es síncrono y de un solo hilo.
 */

var JBVC_WIN = 1000000000;
var JBVC_WIN_THRESHOLD = 500000000;
var JBVC_INF = 2000000000;
var JBVC_MAX_PLY = 64;
var JBVC_TT_BITS = 19;
var JBVC_TT_SIZE = 1 << JBVC_TT_BITS;
var JBVC_TT_MASK = JBVC_TT_SIZE - 1;
var JBVC_MAXM = 32;

/* ---- Tiempo ---- */
var JBVC_MAX_BUDGET = 8200;      /* ms máximos por jugada (límite del enunciado: < 10 s) */
var JBVC_EXPECTED_MOVES = 40;    /* jugadas propias esperadas por partida */
var JBVC_MIN_MOVES_LEFT = 12;    /* nunca gastar más de remaining/12 en una jugada */
var JBVC_QUICK_BELOW = 250;      /* por debajo de este reloj se juega sin búsqueda profunda */

/* ---- Evaluación ---- */
/* niveles: 1 una piedra en ventana de 5, 7 dos piedras, 2 tres piedras, 3 tres abierto (ventana de 6),
            4 cuatro, 5 cuatro abierto, 6 cinco */
var JBVC_LW = [0, 1, 25, 450, 5000, 60000, 0, 4];
var JBVC_THREE_TO_MOVE = 25000;
var JBVC_OPP_THREE1 = 3000;
var JBVC_OPP_THREE2 = 20000;
var JBVC_TEMPO = 10;

/* ---- Swap 2 (calibrado con autojuego, valor de búsqueda a profundidad 4, blancas a mover) ---- */
var JBVC_SWAP_DEPTH = 4;
var JBVC_SWAP_MU = -4;           /* valor ~ 50 % de victoria para las blancas */
var JBVC_SWAP_BLACK_BELOW = -12; /* estado 2: por debajo -> tomar negras */
var JBVC_SWAP_WHITE_ABOVE = -1;  /* estado 2: por encima -> jugar blancas directamente */
var JBVC_SWAP3_WHITE = -3;       /* estado 3: pasarse a blancas si valor >= esto */

/* Aperturas equilibradas medidas en autojuego (37-62 % para blancas), relativas al centro:
   [[bx1,by1],[bx2,by2],[wx,wy]] */
var JBVC_OPENINGS = [
    [[0, 0], [1, 0], [-2, -1]],
    [[0, 0], [2, 0], [1, 1]],
    [[0, 0], [1, 1], [-2, -1]],
    [[0, 0], [2, 1], [0, -2]],
    [[0, 0], [2, 2], [3, 0]],
    [[0, 0], [1, 1], [2, 0]],
    [[0, 0], [2, 0], [0, 2]]
];

/* pesos para ordenar jugadas (analyzeMove) */
var JBVC_AW = [0, 1, 20, 1500, 20000, 500000, 5000000];

var JBVC_DIRS = [[1, 0], [0, 1], [1, 1], [1, -1]];

function jbvcBuildPatterns() {
    var p5 = new Int8Array(243);
    var code, c, i, j, cells;
    for (code = 0; code < 243; code++) {
        c = code;
        cells = [];
        for (i = 4; i >= 0; i--) { cells[i] = c % 3; c = Math.floor(c / 3); }
        var opp = false, own = 0;
        for (j = 0; j < 5; j++) {
            if (cells[j] === 2) opp = true;
            else if (cells[j] === 1) own++;
        }
        var lv = 0;
        if (!opp) {
            if (own === 5) lv = 6;
            else if (own === 4) lv = 4;
            else if (own === 3) lv = 2;
            else if (own === 2) lv = 7;
            else if (own === 1) lv = 1;
        }
        p5[code] = lv;
    }
    var p6 = new Int8Array(729);
    var three = { '001110': 1, '011100': 1, '011010': 1, '010110': 1 };
    for (code = 0; code < 729; code++) {
        c = code;
        cells = [];
        for (i = 5; i >= 0; i--) { cells[i] = c % 3; c = Math.floor(c / 3); }
        var s = cells.join('');
        p6[code] = (s === '011110') ? 5 : (three[s] ? 3 : 0);
    }
    return { p5: p5, p6: p6 };
}

var JBVC_PATTERNS = jbvcBuildPatterns();
var JBVC_P5 = JBVC_PATTERNS.p5;
var JBVC_P6 = JBVC_PATTERNS.p6;

/* Base Agent: usa la clase Agent de gomoku.js si está disponible. */
var JBVC_BaseAgent = (typeof Agent !== 'undefined') ? Agent : class {
    constructor() {}
    compute(board, move_state, time) { return 0; }
};

class JBVCAgent extends JBVC_BaseAgent {
    constructor() {
        super();
        this.N = 0;
        this.cells = null;
        this.lines = null;
        this.cellLines = null;
        this.cellPos = null;
        this.near1 = null;
        this.near2 = null;
        this.neighbors1 = null;
        this.neighbors2 = null;
        this.z1 = null;
        this.z2 = null;
        this.zSide1 = 0;
        this.zSide2 = 0;
        this.centerBias = null;
        this.hash1 = 0;
        this.hash2 = 0;
        this.stoneCount = 0;
        this.nodes = 0;
        this.stopped = false;
        this.deadline = 0;
        this.vctDeadline = 0;
        this.vctAborted = false;
        this.vctNodes = 0;
        this.lastElapsed = 0;
        this.lastDepth = 0;
        this.lastNodes = 0;
        this.lastValue = 0;
        this.lastInfo = '';
        this._an = { score: 0, win: false, four: false, openThree: false };

        this.ttKey1 = new Int32Array(JBVC_TT_SIZE);
        this.ttKey2 = new Int32Array(JBVC_TT_SIZE);
        this.ttDepth = new Int8Array(JBVC_TT_SIZE);
        this.ttFlag = new Int8Array(JBVC_TT_SIZE);
        this.ttValue = new Int32Array(JBVC_TT_SIZE);
        this.ttMove = new Int16Array(JBVC_TT_SIZE);
        this.ttDepth.fill(-1);

        this.killers = new Int16Array(JBVC_MAX_PLY * 2);
        this.killers.fill(-1);
        this.history = null;

        this.mvC = [];
        this.mvS = [];
        this.mvF = [];
        for (var p = 0; p < JBVC_MAX_PLY; p++) {
            this.mvC.push(new Int16Array(JBVC_MAXM));
            this.mvS.push(new Float64Array(JBVC_MAXM));
            this.mvF.push(new Int8Array(JBVC_MAXM));
        }
        this.gbuf = new Int16Array(16);
        this.sbuf = new Int16Array(16);
        this.qbuf = new Int16Array(16);
        this.vbuf = new Int16Array(16);
        this.vbuf2 = new Int16Array(16);
        this.rootC = new Int16Array(64);
        this.rootV = new Float64Array(64);
        this.rootF = new Int8Array(64);
        this.rootN = 0;
        this.stamp = null;
        this.stampId = 0;
    }

    /* ------------------------------------------------------------------ */
    /* Estructuras del tablero                                            */
    /* ------------------------------------------------------------------ */

    setup(N) {
        this.N = N;
        var total = N * N;
        this.cells = new Int8Array(total);
        this.near1 = new Int16Array(total);
        this.near2 = new Int16Array(total);
        this.neighbors1 = new Array(total);
        this.neighbors2 = new Array(total);
        this.stamp = new Int32Array(total);
        this.stampId = 0;

        for (var y = 0; y < N; y++) {
            for (var x = 0; x < N; x++) {
                var idx = y * N + x;
                var n1 = [];
                var n2 = [];
                for (var dy = -2; dy <= 2; dy++) {
                    for (var dx = -2; dx <= 2; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        var yy = y + dy;
                        var xx = x + dx;
                        if (yy >= 0 && yy < N && xx >= 0 && xx < N) {
                            var ni = yy * N + xx;
                            n2.push(ni);
                            if (dx >= -1 && dx <= 1 && dy >= -1 && dy <= 1) n1.push(ni);
                        }
                    }
                }
                this.neighbors1[idx] = n1;
                this.neighbors2[idx] = n2;
            }
        }

        var built = this.buildLines(N);
        this.lines = built.lines;
        this.cellLines = built.cellLines;
        this.cellPos = built.cellPos;
        var nl = this.lines.length;
        this.base5 = new Int32Array(nl);
        this.base6 = new Int32Array(nl);
        var t5 = 0, t6 = 0;
        for (var l = 0; l < nl; l++) {
            var L = this.lines[l].length;
            this.base5[l] = t5;
            this.base6[l] = t6;
            t5 += L - 4;
            if (L >= 6) t6 += L - 5;
        }
        this.w5B = new Int8Array(t5);
        this.w5W = new Int8Array(t5);
        this.w6B = new Int8Array(t6 + 1);
        this.w6W = new Int8Array(t6 + 1);
        this.lcB = new Int16Array(nl * 8);
        this.lcW = new Int16Array(nl * 8);
        this.gB = new Int32Array(8);
        this.gW = new Int32Array(8);
        this.lf4 = new Int32Array(3);
        this.lf3 = new Int32Array(3);
        this.history = new Int32Array(total * 3);

        this.z1 = new Int32Array(total * 3);
        this.z2 = new Int32Array(total * 3);
        var seed = 0x9e3779b9;
        var rnd = function () {
            seed ^= seed << 13;
            seed ^= seed >>> 17;
            seed ^= seed << 5;
            return seed | 0;
        };
        for (var i = 0; i < total * 3; i++) {
            this.z1[i] = rnd();
            this.z2[i] = rnd();
        }
        this.zSide1 = rnd();
        this.zSide2 = rnd();

        this.centerBias = new Int32Array(total);
        var center = (N - 1) / 2;
        for (var cy = 0; cy < N; cy++) {
            for (var cx = 0; cx < N; cx++) {
                var d = Math.abs(cx - center) + Math.abs(cy - center);
                this.centerBias[cy * N + cx] = Math.max(0, Math.floor((N - d) * 2));
            }
        }
        /* la tabla de transposición depende del tamaño de tablero (hash distinto) */
        this.ttDepth.fill(-1);
    }

    buildLines(N) {
        var lines = [];
        var cellLines = new Array(N * N);
        var cellPos = new Array(N * N);
        for (var i = 0; i < N * N; i++) { cellLines[i] = []; cellPos[i] = []; }

        function addLine(cells) {
            if (cells.length < 5) return;
            var id = lines.length;
            lines.push(cells);
            for (var k = 0; k < cells.length; k++) {
                cellLines[cells[k]].push(id);
                cellPos[cells[k]].push(k);
            }
        }

        var y, x, cells, yy, xx;
        for (y = 0; y < N; y++) {
            cells = [];
            for (x = 0; x < N; x++) cells.push(y * N + x);
            addLine(cells);
        }
        for (x = 0; x < N; x++) {
            cells = [];
            for (y = 0; y < N; y++) cells.push(y * N + x);
            addLine(cells);
        }
        for (y = 0; y < N; y++) {
            for (x = 0; x < N; x++) {
                if (y > 0 && x > 0) continue;
                cells = [];
                yy = y; xx = x;
                while (yy < N && xx < N) { cells.push(yy * N + xx); yy++; xx++; }
                addLine(cells);
            }
        }
        for (y = 0; y < N; y++) {
            for (x = 0; x < N; x++) {
                if (y > 0 && x < N - 1) continue;
                cells = [];
                yy = y; xx = x;
                while (yy < N && xx >= 0) { cells.push(yy * N + xx); yy++; xx--; }
                addLine(cells);
            }
        }
        return { lines: lines, cellLines: cellLines, cellPos: cellPos };
    }

    loadBoard(board) {
        var N = board.length;
        if (N !== this.N) this.setup(N);

        this.cells.fill(0);
        this.near1.fill(0);
        this.near2.fill(0);
        this.stoneCount = 0;
        this.hash1 = 0;
        this.hash2 = 0;

        for (var y = 0; y < N; y++) {
            var row = board[y];
            for (var x = 0; x < N; x++) {
                var ch = row[x];
                if (ch === 'B' || ch === 'W') {
                    var cell = y * N + x;
                    var color = ch === 'B' ? 1 : 2;
                    this.cells[cell] = color;
                    this.stoneCount++;
                    this.hash1 ^= this.z1[cell * 3 + color];
                    this.hash2 ^= this.z2[cell * 3 + color];
                    var n1 = this.neighbors1[cell];
                    for (var i = 0; i < n1.length; i++) this.near1[n1[i]]++;
                    var n2 = this.neighbors2[cell];
                    for (var j = 0; j < n2.length; j++) this.near2[n2[j]]++;
                }
            }
        }
        this.windowInit();
        this.resetSearchState();
    }

    resetSearchState() {
        this.killers.fill(-1);
        if (this.history) this.history.fill(0);
        this.nodes = 0;
        this.stopped = false;
    }

    /* Calcula todas las ventanas desde cero. */
    windowInit() {
        this.lcB.fill(0); this.lcW.fill(0);
        this.gB.fill(0); this.gW.fill(0);
        this.lf4.fill(0); this.lf3.fill(0);
        var cells = this.cells;
        var nl = this.lines.length;
        for (var l = 0; l < nl; l++) {
            var line = this.lines[l];
            var L = line.length;
            var l8 = l * 8;
            var b5 = this.base5[l];
            var b6 = this.base6[l];
            var i, j, v, cb, cw, nb, nw;
            for (i = 0; i + 5 <= L; i++) {
                cb = 0; cw = 0;
                for (j = 0; j < 5; j++) {
                    v = cells[line[i + j]];
                    cb = cb * 3 + v;
                    cw = cw * 3 + (v === 0 ? 0 : 3 - v);
                }
                nb = JBVC_P5[cb]; nw = JBVC_P5[cw];
                this.w5B[b5 + i] = nb; this.w5W[b5 + i] = nw;
                if (nb > 0) { this.lcB[l8 + nb]++; this.gB[nb]++; }
                if (nw > 0) { this.lcW[l8 + nw]++; this.gW[nw]++; }
            }
            for (i = 0; i + 6 <= L; i++) {
                cb = 0; cw = 0;
                for (j = 0; j < 6; j++) {
                    v = cells[line[i + j]];
                    cb = cb * 3 + v;
                    cw = cw * 3 + (v === 0 ? 0 : 3 - v);
                }
                nb = JBVC_P6[cb]; nw = JBVC_P6[cw];
                this.w6B[b6 + i] = nb; this.w6W[b6 + i] = nw;
                if (nb > 0) { this.lcB[l8 + nb]++; this.gB[nb]++; }
                if (nw > 0) { this.lcW[l8 + nw]++; this.gW[nw]++; }
            }
            if (this.lcB[l8 + 4] > 0) this.lf4[1]++;
            if (this.lcW[l8 + 4] > 0) this.lf4[2]++;
            if (this.lcB[l8 + 3] > 0) this.lf3[1]++;
            if (this.lcW[l8 + 3] > 0) this.lf3[2]++;
        }
    }

    /* Recalcula solo las ventanas que contienen `cell`. */
    refresh(cell) {
        var cells = this.cells;
        var cls = this.cellLines[cell];
        var cps = this.cellPos[cell];
        var w5B = this.w5B, w5W = this.w5W, w6B = this.w6B, w6W = this.w6W;
        var lcB = this.lcB, lcW = this.lcW, gB = this.gB, gW = this.gW;
        for (var k = 0; k < cls.length; k++) {
            var l = cls[k];
            var p = cps[k];
            var line = this.lines[l];
            var L = line.length;
            var l8 = l * 8;
            var o4B = lcB[l8 + 4] > 0, o4W = lcW[l8 + 4] > 0;
            var o3B = lcB[l8 + 3] > 0, o3W = lcW[l8 + 3] > 0;
            var i, j, v, cb, cw, nb, nw, ob, ow, id;

            var lo = p - 4 < 0 ? 0 : p - 4;
            var hi = p > L - 5 ? L - 5 : p;
            var b5 = this.base5[l];
            for (i = lo; i <= hi; i++) {
                cb = 0; cw = 0;
                for (j = 0; j < 5; j++) {
                    v = cells[line[i + j]];
                    cb = cb * 3 + v;
                    cw = cw * 3 + (v === 0 ? 0 : 3 - v);
                }
                nb = JBVC_P5[cb]; nw = JBVC_P5[cw];
                id = b5 + i;
                ob = w5B[id];
                if (ob !== nb) {
                    w5B[id] = nb;
                    if (ob > 0) { lcB[l8 + ob]--; gB[ob]--; }
                    if (nb > 0) { lcB[l8 + nb]++; gB[nb]++; }
                }
                ow = w5W[id];
                if (ow !== nw) {
                    w5W[id] = nw;
                    if (ow > 0) { lcW[l8 + ow]--; gW[ow]--; }
                    if (nw > 0) { lcW[l8 + nw]++; gW[nw]++; }
                }
            }
            if (L >= 6) {
                lo = p - 5 < 0 ? 0 : p - 5;
                hi = p > L - 6 ? L - 6 : p;
                var b6 = this.base6[l];
                for (i = lo; i <= hi; i++) {
                    cb = 0; cw = 0;
                    for (j = 0; j < 6; j++) {
                        v = cells[line[i + j]];
                        cb = cb * 3 + v;
                        cw = cw * 3 + (v === 0 ? 0 : 3 - v);
                    }
                    nb = JBVC_P6[cb]; nw = JBVC_P6[cw];
                    id = b6 + i;
                    ob = w6B[id];
                    if (ob !== nb) {
                        w6B[id] = nb;
                        if (ob > 0) { lcB[l8 + ob]--; gB[ob]--; }
                        if (nb > 0) { lcB[l8 + nb]++; gB[nb]++; }
                    }
                    ow = w6W[id];
                    if (ow !== nw) {
                        w6W[id] = nw;
                        if (ow > 0) { lcW[l8 + ow]--; gW[ow]--; }
                        if (nw > 0) { lcW[l8 + nw]++; gW[nw]++; }
                    }
                }
            }
            var n4B = lcB[l8 + 4] > 0, n4W = lcW[l8 + 4] > 0;
            var n3B = lcB[l8 + 3] > 0, n3W = lcW[l8 + 3] > 0;
            if (n4B !== o4B) this.lf4[1] += n4B ? 1 : -1;
            if (n4W !== o4W) this.lf4[2] += n4W ? 1 : -1;
            if (n3B !== o3B) this.lf3[1] += n3B ? 1 : -1;
            if (n3W !== o3W) this.lf3[2] += n3W ? 1 : -1;
        }
    }

    makeMove(cell, color) {
        this.cells[cell] = color;
        this.hash1 ^= this.z1[cell * 3 + color];
        this.hash2 ^= this.z2[cell * 3 + color];
        this.refresh(cell);
        var n1 = this.neighbors1[cell];
        for (var j = 0; j < n1.length; j++) this.near1[n1[j]]++;
        var n2 = this.neighbors2[cell];
        for (var k = 0; k < n2.length; k++) this.near2[n2[k]]++;
        this.stoneCount++;
    }

    unmakeMove(cell, color) {
        this.cells[cell] = 0;
        this.hash1 ^= this.z1[cell * 3 + color];
        this.hash2 ^= this.z2[cell * 3 + color];
        this.refresh(cell);
        var n1 = this.neighbors1[cell];
        var i;
        for (i = 0; i < n1.length; i++) this.near1[n1[i]]--;
        var n2 = this.neighbors2[cell];
        for (i = 0; i < n2.length; i++) this.near2[n2[i]]--;
        this.stoneCount--;
    }

    isWinningAt(cell, color) {
        var N = this.N;
        var cells = this.cells;
        var y = (cell / N) | 0;
        var x = cell - y * N;
        for (var d = 0; d < 4; d++) {
            var dx = JBVC_DIRS[d][0];
            var dy = JBVC_DIRS[d][1];
            var count = 1;
            var yy = y + dy;
            var xx = x + dx;
            while (yy >= 0 && yy < N && xx >= 0 && xx < N && cells[yy * N + xx] === color) { count++; yy += dy; xx += dx; }
            yy = y - dy;
            xx = x - dx;
            while (yy >= 0 && yy < N && xx >= 0 && xx < N && cells[yy * N + xx] === color) { count++; yy -= dy; xx -= dx; }
            if (count >= 5) return true;
        }
        return false;
    }

    /* Casillas que completan cinco para `color`. Devuelve cuántas y las deja en `out`. */
    winCells(color, out) {
        if (this.lf4[color] === 0) return 0;
        var lc = color === 1 ? this.lcB : this.lcW;
        var w5 = color === 1 ? this.w5B : this.w5W;
        var cells = this.cells;
        var n = 0;
        var nl = this.lines.length;
        for (var l = 0; l < nl; l++) {
            if (lc[l * 8 + 4] === 0) continue;
            var line = this.lines[l];
            var L = line.length;
            var b5 = this.base5[l];
            for (var i = 0; i + 5 <= L; i++) {
                if (w5[b5 + i] !== 4) continue;
                for (var j = 0; j < 5; j++) {
                    var c = line[i + j];
                    if (cells[c] === 0) {
                        var dup = false;
                        for (var q = 0; q < n; q++) if (out[q] === c) { dup = true; break; }
                        if (!dup && n < 16) out[n++] = c;
                        break;
                    }
                }
            }
        }
        return n;
    }

    /* ------------------------------------------------------------------ */
    /* Evaluación                                                         */
    /* ------------------------------------------------------------------ */

    evaluate(color) {
        var gM = color === 1 ? this.gB : this.gW;
        var gO = color === 1 ? this.gW : this.gB;
        if (gM[6] > 0) return JBVC_WIN;
        if (gO[6] > 0) return -JBVC_WIN;
        var opp = 3 - color;
        var score = 0;
        for (var lv = 1; lv <= 7; lv++) score += (gM[lv] - gO[lv]) * JBVC_LW[lv];
        var t3m = this.lf3[color];
        var t3o = this.lf3[opp];
        if (t3m > 0) score += JBVC_THREE_TO_MOVE;
        if (t3o >= 2) score -= JBVC_OPP_THREE2;
        else if (t3o === 1) score -= JBVC_OPP_THREE1;
        return score + JBVC_TEMPO;
    }

    /* Valor estático estable (sin sesgo de turno) para ordenar y depurar */
    patternScore(color) {
        var g = color === 1 ? this.gB : this.gW;
        var s = 0;
        for (var lv = 1; lv <= 7; lv++) s += g[lv] * JBVC_LW[lv];
        return s;
    }

    cellToPos(cell) {
        var y = (cell / this.N) | 0;
        var x = cell - y * this.N;
        return [x, y];
    }

    analyzeMove(cell, color) {
        var N = this.N;
        var cells = this.cells;
        var y = (cell / N) | 0;
        var x = cell - y * N;
        var score = 0;
        var win = false;
        var four = false;
        var openThree = false;

        for (var d = 0; d < 4; d++) {
            var dx = JBVC_DIRS[d][0];
            var dy = JBVC_DIRS[d][1];
            var count = 1;
            var openEnds = 0;
            var gapCount = 0;

            var f = 0;
            var yy = y + dy;
            var xx = x + dx;
            while (f < 4 && yy >= 0 && yy < N && xx >= 0 && xx < N) {
                if (cells[yy * N + xx] === color) { count++; f++; yy += dy; xx += dx; } else break;
            }
            if (yy >= 0 && yy < N && xx >= 0 && xx < N && cells[yy * N + xx] === 0) {
                openEnds++;
                var yy2 = yy + dy;
                var xx2 = xx + dx;
                if (yy2 >= 0 && yy2 < N && xx2 >= 0 && xx2 < N && cells[yy2 * N + xx2] === color) {
                    var g = 0;
                    while (g < 3 && yy2 >= 0 && yy2 < N && xx2 >= 0 && xx2 < N && cells[yy2 * N + xx2] === color) {
                        g++; yy2 += dy; xx2 += dx;
                    }
                    gapCount += g;
                }
            }

            var b = 0;
            yy = y - dy;
            xx = x - dx;
            while (b < 4 && yy >= 0 && yy < N && xx >= 0 && xx < N) {
                if (cells[yy * N + xx] === color) { count++; b++; yy -= dy; xx -= dx; } else break;
            }
            if (yy >= 0 && yy < N && xx >= 0 && xx < N && cells[yy * N + xx] === 0) {
                openEnds++;
                var yy3 = yy - dy;
                var xx3 = xx - dx;
                if (yy3 >= 0 && yy3 < N && xx3 >= 0 && xx3 < N && cells[yy3 * N + xx3] === color) {
                    var g2 = 0;
                    while (g2 < 3 && yy3 >= 0 && yy3 < N && xx3 >= 0 && xx3 < N && cells[yy3 * N + xx3] === color) {
                        g2++; yy3 -= dy; xx3 -= dx;
                    }
                    gapCount += g2;
                }
            }

            var total = count + gapCount;
            var lvl = 0;
            if (count >= 5) { win = true; lvl = 6; }
            else if (count === 4 && openEnds >= 1) { four = true; lvl = 4; }
            else if (total >= 4 && gapCount > 0) { four = true; lvl = 4; }
            else if (count === 3 && openEnds >= 2) { openThree = true; lvl = 3; }
            else if (total >= 3 && gapCount > 0 && openEnds >= 1) { openThree = true; lvl = 3; }
            else if (count === 3 && openEnds >= 1) lvl = 2;
            else if (total >= 3) lvl = 2;
            else if (count === 2 && openEnds >= 2) lvl = 1;
            else if (count >= 1) lvl = 1;
            score += JBVC_AW[lvl];
            if (lvl >= 2) score += openEnds * 10;
        }
        var an = this._an;
        an.score = score;
        an.win = win;
        an.four = four;
        an.openThree = openThree;
        return an;
    }

    /* ------------------------------------------------------------------ */
    /* Generación de jugadas                                              */
    /* ------------------------------------------------------------------ */

    getBranchLimit(depth, ply) {
        var limit;
        if (ply === 0) limit = 24;
        else if (ply === 1) limit = 16;
        else if (depth >= 6) limit = 14;
        else if (depth >= 4) limit = 12;
        else limit = 8;
        if (this.N > 15) limit = Math.max(6, limit - 2);
        return limit;
    }

    /* Rellena this.mvC/mvS/mvF[ply] y devuelve el número de jugadas.
       flags: 1 = hace cuatro, 2 = hace tres abierto. */
    generateMoves(color, depth, ply, ttMove) {
        var N = this.N;
        var cells = this.cells;
        var opp = 3 - color;
        var P = ply < JBVC_MAX_PLY ? ply : JBVC_MAX_PLY - 1;
        var mc = this.mvC[P], ms = this.mvS[P], mf = this.mvF[P];

        /* bloqueo forzado */
        if (this.lf4[opp] > 0) {
            var nw = this.winCells(opp, this.gbuf);
            if (nw >= 1) { mc[0] = this.gbuf[0]; ms[0] = 0; mf[0] = 0; return 1; }
        }

        var n = 0;
        var K = this.getBranchLimit(depth, ply);
        var near = (ply <= 1) ? this.near2 : this.near1;
        var kIdx = Math.min(ply, JBVC_MAX_PLY - 1) * 2;
        var killer0 = this.killers[kIdx];
        var killer1 = this.killers[kIdx + 1];
        var histBase = color * N * N;
        var total = N * N;

        for (var cell = 0; cell < total; cell++) {
            if (cells[cell] !== 0 || near[cell] === 0) continue;
            var atk = this.analyzeMove(cell, color);
            var atkScore = atk.score;
            var atkFour = atk.four;
            var atkOpenThree = atk.openThree;
            var def = this.analyzeMove(cell, opp);
            var score = atkScore + def.score * 0.9;
            if (atkFour) score += 1000000;
            if (atkOpenThree) score += 50000;
            if (def.four) score += 200000;
            if (def.openThree) score += 20000;
            if (cell === ttMove) score += 5000000;
            if (cell === killer0 || cell === killer1) score += 300000;
            score += this.history[histBase + cell] * 0.05;
            score += this.centerBias[cell];

            var j;
            if (n < K) j = n++;
            else if (score > ms[K - 1]) j = K - 1;
            else continue;
            while (j > 0 && ms[j - 1] < score) {
                ms[j] = ms[j - 1]; mc[j] = mc[j - 1]; mf[j] = mf[j - 1];
                j--;
            }
            ms[j] = score;
            mc[j] = cell;
            mf[j] = (atkFour ? 1 : 0) | (atkOpenThree ? 2 : 0);
        }

        if (n === 0) {
            var fb = this.findFallbackMove();
            if (fb >= 0) { mc[0] = fb; ms[0] = 0; mf[0] = 0; n = 1; }
        }
        return n;
    }

    findFallbackMove() {
        var N = this.N;
        if (!N) return -1;
        var cells = this.cells;
        var center = (N - 1) / 2;
        var best = -1;
        var bestD = Infinity;
        for (var cell = 0; cell < N * N; cell++) {
            if (cells[cell] === 0) {
                var y = (cell / N) | 0;
                var x = cell - y * N;
                var d = Math.abs(x - center) + Math.abs(y - center);
                if (d < bestD) { bestD = d; best = cell; }
            }
        }
        return best;
    }

    /* ------------------------------------------------------------------ */
    /* Búsqueda                                                           */
    /* ------------------------------------------------------------------ */

    quiesce(color, ply, qd) {
        this.nodes++;
        if (this.lf4[color] > 0) return JBVC_WIN - ply - 1;
        var opp = 3 - color;
        if (this.lf4[opp] > 0) {
            var no = this.winCells(opp, this.qbuf);
            if (no >= 2) return -(JBVC_WIN - ply - 2);
            if (no === 1) {
                if (qd <= 0 || ply >= JBVC_MAX_PLY - 2) return this.evaluate(color);
                var c = this.qbuf[0];
                this.makeMove(c, color);
                var v = -this.quiesce(opp, ply + 1, qd - 1);
                this.unmakeMove(c, color);
                return v;
            }
        }
        return this.evaluate(color);
    }

    search(depth, alpha, beta, color, ply) {
        if (this.stopped) return 0;
        this.nodes++;
        if ((this.nodes & 127) === 0 && Date.now() >= this.deadline) {
            this.stopped = true;
            return 0;
        }

        var opp = 3 - color;
        if (this.lf4[color] > 0) return JBVC_WIN - ply - 1;
        var forced = false;
        if (this.lf4[opp] > 0) {
            var nw = this.winCells(opp, this.sbuf);
            if (nw >= 2) return -(JBVC_WIN - ply - 2);
            forced = true;
        }

        var alphaOrig = alpha;
        var betaOrig = beta;
        var key1 = this.hash1 ^ (color === 1 ? this.zSide1 : 0);
        var key2 = this.hash2 ^ (color === 1 ? this.zSide2 : 0);
        var idx = key1 & JBVC_TT_MASK;
        var ttMove = -1;

        if (this.ttKey1[idx] === key1 && this.ttKey2[idx] === key2) {
            if (this.ttDepth[idx] >= depth) {
                var ttVal = this.ttValue[idx];
                if (ttVal > JBVC_WIN_THRESHOLD) ttVal -= ply;
                else if (ttVal < -JBVC_WIN_THRESHOLD) ttVal += ply;
                var ttFlag = this.ttFlag[idx];
                if (ttFlag === 0) return ttVal;
                if (ttFlag === 1) { if (ttVal > alpha) alpha = ttVal; }
                else { if (ttVal < beta) beta = ttVal; }
                if (alpha >= beta) return ttVal;
            }
            ttMove = this.ttMove[idx];
        }

        if (ply >= JBVC_MAX_PLY - 3) return this.evaluate(color);
        if (depth <= 0) return this.quiesce(color, ply, 4);
        if (this.stoneCount >= this.N * this.N) return 0;

        var n = this.generateMoves(color, depth, ply, ttMove);
        if (n === 0) return this.evaluate(color);
        var mc = this.mvC[ply], mf = this.mvF[ply];

        var best = -JBVC_INF;
        var bestMove = mc[0];
        var kIdx = ply * 2;
        var total = this.N * this.N;

        for (var i = 0; i < n; i++) {
            if (this.stopped) return 0;
            var cell = mc[i];
            var fl = mf[i];
            this.makeMove(cell, color);
            var ext = 0;
            if (forced) { if (ply < 40) ext = 1; }
            else if ((fl & 1) && ply < 16 && depth <= 6) ext = 1;
            else if ((fl & 2) && ply < 8 && depth <= 3) ext = 1;
            var nd = depth - 1 + ext;
            var val;
            if (i === 0) {
                val = -this.search(nd, -beta, -alpha, opp, ply + 1);
            } else {
                var rd = nd;
                if (i >= 3 && depth >= 3 && ext === 0 && fl === 0 && !forced) rd = nd - 1;
                val = -this.search(rd, -alpha - 1, -alpha, opp, ply + 1);
                if (!this.stopped && val > alpha && rd < nd) val = -this.search(nd, -alpha - 1, -alpha, opp, ply + 1);
                if (!this.stopped && val > alpha && val < beta) val = -this.search(nd, -beta, -alpha, opp, ply + 1);
            }
            this.unmakeMove(cell, color);
            if (this.stopped) return 0;

            if (val > best) { best = val; bestMove = cell; }
            if (val > alpha) alpha = val;
            if (alpha >= beta) {
                if (this.killers[kIdx] !== cell) {
                    this.killers[kIdx + 1] = this.killers[kIdx];
                    this.killers[kIdx] = cell;
                }
                this.history[color * total + cell] += depth * depth;
                break;
            }
        }

        if (!this.stopped) {
            var flag = 0;
            if (best <= alphaOrig) flag = 2;
            else if (best >= betaOrig) flag = 1;
            var storeVal = best;
            if (best > JBVC_WIN_THRESHOLD) storeVal = best + ply;
            else if (best < -JBVC_WIN_THRESHOLD) storeVal = best - ply;
            this.ttKey1[idx] = key1;
            this.ttKey2[idx] = key2;
            this.ttDepth[idx] = depth > 100 ? 100 : depth;
            this.ttFlag[idx] = flag;
            this.ttValue[idx] = storeVal;
            this.ttMove[idx] = bestMove;
        }
        return best;
    }

    /* Prepara la lista de jugadas de la raíz (opcionalmente filtrada). */
    prepareRoot(color, allowed) {
        var n = this.generateMoves(color, 4, 0, -1);
        var mc = this.mvC[0], ms = this.mvS[0], mf = this.mvF[0];
        var m = 0;
        for (var i = 0; i < n; i++) {
            if (allowed && !allowed[mc[i]]) continue;
            this.rootC[m] = mc[i];
            this.rootF[m] = mf[i];
            this.rootV[m] = 0;
            m++;
        }
        if (m === 0) {
            for (var k = 0; k < n; k++) {
                this.rootC[k] = mc[k];
                this.rootF[k] = mf[k];
                this.rootV[k] = 0;
            }
            m = n;
        }
        this.rootN = m;
    }

    /* Una iteración a profundidad `depth`. Devuelve {cell, value, ok}. */
    searchRoot(depth, color) {
        var opp = 3 - color;
        var alpha = -JBVC_INF;
        var beta = JBVC_INF;
        var bestIdx = -1;
        var bestVal = -JBVC_INF;
        var n = this.rootN;
        for (var i = 0; i < n; i++) {
            var cell = this.rootC[i];
            var fl = this.rootF[i];
            this.makeMove(cell, color);
            var ext = 0;
            if ((fl & 1) && depth <= 6) ext = 1;
            else if ((fl & 2) && depth <= 3) ext = 1;
            var nd = depth - 1 + ext;
            var val;
            if (i === 0) {
                val = -this.search(nd, -beta, -alpha, opp, 1);
            } else {
                val = -this.search(nd, -alpha - 1, -alpha, opp, 1);
                if (!this.stopped && val > alpha && val < beta) val = -this.search(nd, -beta, -alpha, opp, 1);
            }
            this.unmakeMove(cell, color);
            if (this.stopped) break;
            this.rootV[i] = val;
            if (val > bestVal) { bestVal = val; bestIdx = i; }
            if (val > alpha) alpha = val;
        }
        if (bestIdx < 0) return { cell: -1, value: 0, ok: false };
        /* mover la mejor jugada al frente */
        var bc = this.rootC[bestIdx], bf = this.rootF[bestIdx], bv = this.rootV[bestIdx];
        for (var k = bestIdx; k > 0; k--) {
            this.rootC[k] = this.rootC[k - 1];
            this.rootF[k] = this.rootF[k - 1];
            this.rootV[k] = this.rootV[k - 1];
        }
        this.rootC[0] = bc; this.rootF[0] = bf; this.rootV[0] = bv;
        return { cell: bc, value: bestVal, ok: !this.stopped };
    }

    /* Deepening iterativo sobre la raíz ya preparada. */
    iterate(color, maxDepth) {
        var best = -1;
        var bestVal = 0;
        this.stopped = false;
        for (var depth = 1; depth <= maxDepth; depth++) {
            if (Date.now() >= this.deadline) break;
            this.stopped = false;
            var res = this.searchRoot(depth, color);
            if (res.cell >= 0) {
                best = res.cell;
                bestVal = res.value;
                this.lastDepth = depth;
            }
            if (this.stopped || !res.ok) break;
            if (bestVal > JBVC_WIN_THRESHOLD) break;
            if (bestVal < -JBVC_WIN_THRESHOLD) break;
            if (this.rootN === 1) break;
        }
        this.lastValue = bestVal;
        this.lastNodes = this.nodes;
        return best;
    }

    /* ------------------------------------------------------------------ */
    /* Búsqueda de amenazas (VCF / VCT)                                   */
    /* ------------------------------------------------------------------ */

    collectFourCells(color) {
        var lc = color === 1 ? this.lcB : this.lcW;
        var w5 = color === 1 ? this.w5B : this.w5W;
        var cells = this.cells;
        var res = [];
        this.stampId++;
        var sid = this.stampId, st = this.stamp;
        var nl = this.lines.length;
        for (var l = 0; l < nl; l++) {
            if (lc[l * 8 + 2] === 0) continue;
            var line = this.lines[l];
            var L = line.length;
            var b5 = this.base5[l];
            for (var i = 0; i + 5 <= L; i++) {
                if (w5[b5 + i] !== 2) continue;
                for (var j = 0; j < 5; j++) {
                    var c = line[i + j];
                    if (cells[c] === 0 && st[c] !== sid) { st[c] = sid; res.push(c); }
                }
            }
        }
        return res;
    }

    collectThreeCells(color) {
        var lc = color === 1 ? this.lcB : this.lcW;
        var w5 = color === 1 ? this.w5B : this.w5W;
        var cells = this.cells;
        var res = [];
        this.stampId++;
        var sid = this.stampId, st = this.stamp;
        var nl = this.lines.length;
        for (var l = 0; l < nl; l++) {
            if (lc[l * 8 + 7] === 0) continue;
            var line = this.lines[l];
            var L = line.length;
            var b5 = this.base5[l];
            for (var i = 0; i + 5 <= L; i++) {
                if (w5[b5 + i] !== 7) continue;
                for (var j = 0; j < 5; j++) {
                    var c = line[i + j];
                    if (cells[c] === 0 && st[c] !== sid) { st[c] = sid; res.push(c); }
                }
            }
        }
        return res;
    }

    /* Casillas con las que el defensor puede romper los tres abiertos de `me`
       (casillas vacías de las ventanas de 6 con tres abierto) + sus contra-cuatros. */
    defenseCells(me) {
        var opp = 3 - me;
        var lc = me === 1 ? this.lcB : this.lcW;
        var w6 = me === 1 ? this.w6B : this.w6W;
        var cells = this.cells;
        var res = [];
        this.stampId++;
        var sid = this.stampId, st = this.stamp;
        var nl = this.lines.length;
        var l, line, L, b6, i, j, c;
        for (l = 0; l < nl; l++) {
            if (lc[l * 8 + 3] === 0) continue;
            line = this.lines[l];
            L = line.length;
            b6 = this.base6[l];
            for (i = 0; i + 6 <= L; i++) {
                if (w6[b6 + i] !== 3) continue;
                for (j = 0; j < 6; j++) {
                    c = line[i + j];
                    if (cells[c] === 0 && st[c] !== sid) { st[c] = sid; res.push(c); }
                }
            }
        }
        /* contra-cuatros del defensor */
        var oc = this.collectFourCells(opp);
        for (i = 0; i < oc.length && i < 8; i++) {
            c = oc[i];
            if (st[c] !== sid) { st[c] = sid; res.push(c); }
        }
        return res;
    }

    /* `me` mueve. Devuelve la casilla que inicia una secuencia ganadora o -1. */
    vctAttack(me, d, three) {
        if (this.vctAborted) return -1;
        this.vctNodes++;
        if ((this.vctNodes & 63) === 0 && Date.now() >= this.vctDeadline) { this.vctAborted = true; return -1; }
        var opp = 3 - me;
        var buf = this.vbuf;
        if (this.winCells(me, buf) > 0) return buf[0];
        var no = this.winCells(opp, buf);
        if (no >= 2) return -1;
        if (no === 1) {
            var b = buf[0];
            if (d <= 0) return -1;
            this.makeMove(b, me);
            var r = this.vctDefend(me, d - 1, three);
            this.unmakeMove(b, me);
            return r ? b : -1;
        }
        if (d <= 0) return -1;

        var i, c, ok;
        var fours = this.collectFourCells(me);
        for (i = 0; i < fours.length; i++) {
            c = fours[i];
            this.makeMove(c, me);
            ok = this.lf4[me] > 0 && this.vctDefend(me, d - 1, three);
            this.unmakeMove(c, me);
            if (ok) return c;
            if (this.vctAborted) return -1;
        }
        if (three && d >= 3) {
            var cands = this.collectThreeCells(me);
            if (cands.length > 0) {
                var scored = [];
                for (i = 0; i < cands.length; i++) {
                    var an = this.analyzeMove(cands[i], me);
                    if (an.openThree) scored.push({ c: cands[i], s: an.score });
                }
                scored.sort(function (a, b2) { return b2.s - a.s; });
                var lim = Math.min(scored.length, 8);
                for (i = 0; i < lim; i++) {
                    c = scored[i].c;
                    var before = this.lf3[me];
                    this.makeMove(c, me);
                    ok = false;
                    if (this.lf3[me] > before && this.lf4[opp] === 0) ok = this.vctDefend(me, d - 1, three);
                    this.unmakeMove(c, me);
                    if (ok) return c;
                    if (this.vctAborted) return -1;
                }
            }
        }
        return -1;
    }

    /* El oponente de `me` mueve tras una amenaza de `me`. true = `me` gana por la fuerza. */
    vctDefend(me, d, three) {
        if (this.vctAborted) return false;
        var opp = 3 - me;
        var b1 = this.vbuf2;
        if (this.lf4[opp] > 0) return false; /* el defensor completa cinco */
        var nm = this.winCells(me, b1);
        var replies;
        if (nm >= 2) return true;
        if (nm === 1) replies = [b1[0]];
        else {
            if (this.lf3[me] === 0) return false;
            replies = this.defenseCells(me);
            if (replies.length === 0) return false;
        }
        if (d <= 0) return false;

        for (var i = 0; i < replies.length; i++) {
            var r = replies[i];
            this.makeMove(r, opp);
            var ok;
            if (this.lf4[opp] > 0) {
                if (this.lf4[me] > 0) ok = true;
                else {
                    var nw = this.winCells(opp, b1);
                    if (nw >= 2) ok = false;
                    else {
                        var bk = b1[0];
                        this.makeMove(bk, me);
                        ok = this.vctDefend(me, d - 1, three);
                        this.unmakeMove(bk, me);
                    }
                }
            } else {
                ok = this.vctAttack(me, d - 1, three) >= 0;
            }
            this.unmakeMove(r, opp);
            if (!ok) return false;
            if (this.vctAborted) return false;
        }
        return true;
    }

    runThreat(me, d, three, ms) {
        this.vctDeadline = Math.min(this.deadline, Date.now() + ms);
        this.vctAborted = false;
        this.vctNodes = 0;
        var r = this.vctAttack(me, d, three);
        if (this.vctAborted) return -1;
        return r;
    }

    /* ------------------------------------------------------------------ */
    /* Jugada normal                                                      */
    /* ------------------------------------------------------------------ */

    quickMove(color) {
        var n = this.generateMoves(color, 2, 1, -1);
        if (n > 0) return this.cellToPos(this.mvC[1][0]);
        var fb = this.findFallbackMove();
        return this.cellToPos(fb < 0 ? 0 : fb);
    }

    playNormal(color, quick) {
        var opp = 3 - color;
        var N = this.N;
        if (this.stoneCount === 0) return this.cellToPos(Math.floor(N / 2) * N + Math.floor(N / 2));

        if (this.winCells(color, this.sbuf) > 0) return this.cellToPos(this.sbuf[0]);
        if (this.winCells(opp, this.sbuf) > 0) return this.cellToPos(this.sbuf[0]);
        if (quick) return this.quickMove(color);

        var start = Date.now();
        var total = this.deadline - start;
        var found;

        /* 1. ataque: VCF y luego VCT */
        found = this.runThreat(color, 16, false, Math.max(20, total * 0.08));
        if (found >= 0 && this.cells[found] === 0) { this.lastInfo = 'vcf'; return this.cellToPos(found); }
        found = this.runThreat(color, 8, true, Math.max(30, total * 0.17));
        if (found >= 0 && this.cells[found] === 0) { this.lastInfo = 'vct'; return this.cellToPos(found); }

        /* 2. defensa: ¿el rival ganaría por la fuerza si jugara ahora? */
        var allowed = null;
        var oppWin = this.runThreat(opp, 8, true, Math.max(30, total * 0.10));
        if (oppWin >= 0) {
            var cands = [];
            var mc = this.mvC[0];
            var gm = this.generateMoves(color, 4, 0, -1);
            for (var i = 0; i < gm; i++) cands.push(mc[i]);
            if (cands.indexOf(oppWin) < 0 && this.cells[oppWin] === 0) cands.unshift(oppWin);
            allowed = new Uint8Array(N * N);
            var safeCount = 0;
            for (var k = 0; k < cands.length; k++) {
                if (Date.now() >= this.deadline - total * 0.3) break;
                var cm = cands[k];
                this.makeMove(cm, color);
                var w = this.runThreat(opp, 8, true, Math.max(15, total * 0.04));
                this.unmakeMove(cm, color);
                if (w < 0 && !this.vctAborted) { allowed[cm] = 1; safeCount++; }
            }
            if (safeCount === 0) allowed = null;
            this.lastInfo = 'def' + safeCount;
        }

        /* 3. búsqueda alfa-beta */
        this.stopped = false;
        this.prepareRoot(color, allowed);
        var best = this.rootN > 0 ? this.rootC[0] : -1;
        var it = this.iterate(color, 40);
        if (it >= 0) best = it;
        if (best < 0 || this.cells[best] !== 0) best = this.findFallbackMove();
        if (best < 0) best = 0;
        return this.cellToPos(best);
    }

    /* ------------------------------------------------------------------ */
    /* Swap 2                                                             */
    /* ------------------------------------------------------------------ */

    playOpening() {
        var N = this.N;
        var cx = Math.floor(N / 2), cy = Math.floor(N / 2);
        var tries = 0;
        while (tries++ < 20) {
            var o = JBVC_OPENINGS[Math.floor(Math.random() * JBVC_OPENINGS.length)];
            /* simetría aleatoria (8 isometrías) para no ser predecible */
            var s = Math.floor(Math.random() * 8);
            var pts = [];
            var ok = true;
            for (var i = 0; i < 3; i++) {
                var dx = o[i][0], dy = o[i][1], t;
                if (s & 1) dx = -dx;
                if (s & 2) dy = -dy;
                if (s & 4) { t = dx; dx = dy; dy = t; }
                var x = cx + dx, y = cy + dy;
                if (x < 0 || y < 0 || x >= N || y >= N) { ok = false; break; }
                pts.push([x, y]);
            }
            if (ok) return pts;
        }
        return null;
    }

    /* Valor, desde las blancas (que mueven), de la posición actual del tablero cargado. */
    whiteValue(ms, maxDepth) {
        this.deadline = Date.now() + ms;
        this.resetSearchState();
        this.prepareRoot(2, null);
        var cell = this.iterate(2, maxDepth || 14);
        if (this.lastDepth === 0 && cell < 0) return { value: 0, cell: this.findFallbackMove() };
        return { value: this.lastValue, cell: cell };
    }

    decideSwap(state, ms) {
        var t0 = Date.now();
        var wv = this.whiteValue(Math.max(50, ms * 0.3), JBVC_SWAP_DEPTH);
        var v = wv.value;
        var best = wv.cell;
        if (state === '3') {
            /* pasarse a blancas implica jugar ya esa piedra */
            if (v >= JBVC_SWAP3_WHITE) return this.cellToPos(best);
            return 'BLACK';
        }
        if (v < JBVC_SWAP_BLACK_BELOW) return 'BLACK';
        if (v > JBVC_SWAP_WHITE_ABOVE) return this.cellToPos(best);
        /* zona equilibrada: opción 3, colocar una blanca y una negra dejando al rival una posición pareja */
        var pair = this.balancePair(Math.max(100, ms - (Date.now() - t0)));
        if (pair) return pair;
        return 'BLACK';
    }

    /* Elige (blanca, negra) para que la posición resultante (blancas a mover) quede lo más
       cerca posible del equilibrio. */
    balancePair(ms) {
        var t0 = Date.now();
        var endAt = t0 + ms;
        var N = this.N;
        var wn = this.generateMoves(2, 4, 1, -1);
        var wc = [];
        for (var i = 0; i < wn && i < 8; i++) wc.push(this.mvC[1][i]);
        var bestPair = null;
        var bestGap = Infinity;
        var perEval = Math.max(10, ms / (wc.length * 8 + 1));
        for (var a = 0; a < wc.length; a++) {
            if (Date.now() > endAt) break;
            var w = wc[a];
            this.makeMove(w, 2);
            var bn = this.generateMoves(1, 4, 1, -1);
            var bc = [];
            for (var j = 0; j < bn && j < 8; j++) bc.push(this.mvC[1][j]);
            for (var b = 0; b < bc.length; b++) {
                if (Date.now() > endAt) break;
                var bm = bc[b];
                this.makeMove(bm, 1);
                var wv = this.whiteValue(perEval, JBVC_SWAP_DEPTH);
                this.unmakeMove(bm, 1);
                var gap = Math.abs(wv.value - JBVC_SWAP_MU);
                if (gap < bestGap) { bestGap = gap; bestPair = [this.cellToPos(w), this.cellToPos(bm)]; }
            }
            this.unmakeMove(w, 2);
        }
        return bestPair;
    }

    /* ------------------------------------------------------------------ */
    /* Punto de entrada                                                   */
    /* ------------------------------------------------------------------ */

    /* `time` es el reloj acumulado restante (ms) de este jugador. */
    allocateBudget(remaining) {
        var own = Math.floor(this.stoneCount / 2);
        var movesLeft = Math.max(JBVC_MIN_MOVES_LEFT, JBVC_EXPECTED_MOVES - own);
        var b = remaining / movesLeft;
        if (b > JBVC_MAX_BUDGET) b = JBVC_MAX_BUDGET;
        if (b > remaining - 150) b = remaining - 150;
        return b;
    }

    compute(board, move_state, time) {
        var start = Date.now();
        this.loadBoard(board);

        var remaining = 10000;
        if (typeof time === 'number' && isFinite(time)) remaining = time;
        else if (Array.isArray(time) && time.length > 0 && isFinite(time[0])) remaining = time[0];

        var state = String(move_state);
        var result;

        if (state === '1') {
            result = this.playOpening();
        } else if (state === '2' || state === '3') {
            var budget = Math.min(2500, remaining * 0.06);
            if (remaining < 600) result = (state === '2') ? 'BLACK' : null;
            else result = this.decideSwap(state, budget);
            if (result === null) result = this.cellToPos(this.findFallbackMove());
        } else {
            var color = state === 'W' ? 2 : 1;
            var quick = remaining < JBVC_QUICK_BELOW;
            var bud = this.allocateBudget(remaining);
            if (bud < 20) quick = true;
            this.deadline = start + Math.max(10, bud);
            result = this.playNormal(color, quick);
        }

        result = this.validateResult(result, board, state);
        this.lastElapsed = Date.now() - start;
        return result;
    }

    validateResult(result, board, state) {
        var N = board.length;

        function isEmpty(p) {
            if (!Array.isArray(p) || p.length < 2) return false;
            var x = p[0];
            var y = p[1];
            if (typeof x !== 'number' || typeof y !== 'number') return false;
            if (x % 1 !== 0 || y % 1 !== 0) return false;
            if (x < 0 || x >= N || y < 0 || y >= N) return false;
            return board[y][x] === ' ';
        }

        if (state === '1') {
            if (Array.isArray(result) && result.length === 3) {
                var used = {};
                var ok = true;
                for (var i = 0; i < 3; i++) {
                    if (!isEmpty(result[i])) { ok = false; break; }
                    var key = result[i][0] + ',' + result[i][1];
                    if (used[key]) { ok = false; break; }
                    used[key] = true;
                }
                if (ok) return result;
            }
            return this.fallbackTriple(board);
        }

        if (state === '2' || state === '3') {
            if (result === 'BLACK') return result;
            if (Array.isArray(result) && result.length === 2 && typeof result[0] === 'number' && isEmpty(result)) return result;
            if (state === '2' && Array.isArray(result) && result.length === 2 && Array.isArray(result[0]) && Array.isArray(result[1])) {
                if (isEmpty(result[0]) && isEmpty(result[1])) {
                    if (result[0][0] !== result[1][0] || result[0][1] !== result[1][1]) return result;
                }
            }
            return this.fallbackSingle(board);
        }

        if (Array.isArray(result) && result.length === 2 && typeof result[0] === 'number' && isEmpty(result)) return result;
        return this.fallbackSingle(board);
    }

    fallbackSingle(board) {
        var N = board.length;
        var center = Math.floor(N / 2);
        var best = null;
        var bestD = Infinity;
        for (var y = 0; y < N; y++) {
            for (var x = 0; x < N; x++) {
                if (board[y][x] === ' ') {
                    var d = Math.abs(x - center) + Math.abs(y - center);
                    if (d < bestD) { bestD = d; best = [x, y]; }
                }
            }
        }
        return best || [0, 0];
    }

    fallbackTriple(board) {
        var N = board.length;
        var center = Math.floor(N / 2);
        var list = [];
        for (var y = 0; y < N; y++) {
            for (var x = 0; x < N; x++) {
                if (board[y][x] === ' ') {
                    list.push({ x: x, y: y, d: Math.abs(x - center) + Math.abs(y - center) });
                }
            }
        }
        list.sort(function (a, b) { return a.d - b.d; });
        var res = [];
        for (var i = 0; i < list.length && res.length < 3; i++) res.push([list[i].x, list[i].y]);
        while (res.length < 3) res.push([0, 0]);
        return res;
    }
}

if (typeof module !== 'undefined' && module.exports) module.exports = JBVCAgent;