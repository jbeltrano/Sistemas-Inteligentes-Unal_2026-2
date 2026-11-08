/*
 * JBVCAgent.js
 * Agente de Gomoku (variante Swap 2) con heurísticas + minimax alpha-beta.
 * Interfaz compatible con class Agent de gomoku.js.
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

var JBVC_MAX_BUDGET = 8200;
var JBVC_SAFETY = 600;
var JBVC_MIN_BUDGET = 100;

var JBVC_LEVEL_NONE = 0;
var JBVC_LEVEL_WEAK = 1;
var JBVC_LEVEL_THREE = 2;
var JBVC_LEVEL_OPEN_THREE = 3;
var JBVC_LEVEL_FOUR = 4;
var JBVC_LEVEL_OPEN_FOUR = 5;
var JBVC_LEVEL_FIVE = 6;

var JBVC_LEVEL_WEIGHT = [0, 1, 20, 1500, 20000, 500000, 5000000];
var JBVC_DOUBLE_FOUR_BONUS = 400000;
var JBVC_DOUBLE_THREE_BONUS = 100000;
var JBVC_FOUR_THREE_BONUS = 250000;
var JBVC_TEMPO_BONUS = 5;

var JBVC_DIRS = [[1, 0], [0, 1], [1, 1], [1, -1]];

/* Tabla de niveles para ventanas de 5 celdas: 0 vacío, 1 propio, 2 rival. */
function jbvcBuildPattern5() {
    var arr = new Int8Array(243);
    for (var code = 0; code < 243; code++) {
        var c = code;
        var cells = [];
        for (var i = 4; i >= 0; i--) {
            cells[i] = c % 3;
            c = Math.floor(c / 3);
        }
        var opp = false;
        for (var j = 0; j < 5; j++) {
            if (cells[j] === 2) { opp = true; break; }
        }
        if (opp) { arr[code] = 0; continue; }
        var own = 0;
        for (var k = 0; k < 5; k++) if (cells[k] === 1) own++;
        var s = cells.join('');
        var level = 0;
        if (own === 5) level = 6;
        else if (own === 4) level = 4;
        else if (own === 3) level = (s === '01110' ? 3 : 2);
        else if (own === 2) level = 1;
        else if (own === 1) level = 1;
        arr[code] = level;
    }
    return arr;
}

var JBVC_PATTERN5 = jbvcBuildPattern5();

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
        this.lineLevelB = null;
        this.lineLevelW = null;
        this.countLevelB = null;
        this.countLevelW = null;
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
        this.vcfDeadline = 0;
        this.vcfAborted = false;
        this.vcfNodes = 0;
        this.prevTime = null;
        this.perMoveMode = null;
        this.moveCount = 0;
        this.color = null;
        this.lastElapsed = 0;
        this.lastDepth = 0;
        this.lastNodes = 0;
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
    }

    setup(N) {
        this.N = N;
        var total = N * N;
        this.cells = new Int8Array(total);
        this.near1 = new Int16Array(total);
        this.near2 = new Int16Array(total);
        this.neighbors1 = new Array(total);
        this.neighbors2 = new Array(total);

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
        this.lineLevelB = new Int8Array(this.lines.length);
        this.lineLevelW = new Int8Array(this.lines.length);
        this.countLevelB = new Int32Array(7);
        this.countLevelW = new Int32Array(7);
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
    }

    buildLines(N) {
        var lines = [];
        var cellLines = new Array(N * N);
        for (var i = 0; i < N * N; i++) cellLines[i] = [];

        function addLine(cells) {
            if (cells.length < 5) return;
            var id = lines.length;
            lines.push(cells);
            for (var k = 0; k < cells.length; k++) cellLines[cells[k]].push(id);
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
                yy = y;
                xx = x;
                while (yy < N && xx < N) {
                    cells.push(yy * N + xx);
                    yy++;
                    xx++;
                }
                addLine(cells);
            }
        }
        for (y = 0; y < N; y++) {
            for (x = 0; x < N; x++) {
                if (y > 0 && x < N - 1) continue;
                cells = [];
                yy = y;
                xx = x;
                while (yy < N && xx >= 0) {
                    cells.push(yy * N + xx);
                    yy++;
                    xx--;
                }
                addLine(cells);
            }
        }
        return { lines: lines, cellLines: cellLines };
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

        this.countLevelB.fill(0);
        this.countLevelW.fill(0);
        for (var l = 0; l < this.lines.length; l++) {
            this.recomputeLine(l);
            this.countLevelB[this.lineLevelB[l]]++;
            this.countLevelW[this.lineLevelW[l]]++;
        }

        this.killers.fill(-1);
        if (this.history) this.history.fill(0);
        this.nodes = 0;
        this.stopped = false;
    }

    recomputeLine(lineId) {
        var line = this.lines[lineId];
        var L = line.length;
        var maxB = 0;
        var maxW = 0;
        var i, j, v, cb, cw, lb, lw;

        for (i = 0; i + 5 <= L; i++) {
            cb = 0;
            cw = 0;
            for (j = 0; j < 5; j++) {
                v = this.cells[line[i + j]];
                cb = cb * 3 + v;
                cw = cw * 3 + (v === 0 ? 0 : (v === 1 ? 2 : 1));
            }
            lb = JBVC_PATTERN5[cb];
            lw = JBVC_PATTERN5[cw];
            if (lb > maxB) maxB = lb;
            if (lw > maxW) maxW = lw;
        }

        if (L >= 6) {
            for (i = 0; i + 6 <= L; i++) {
                cb = 0;
                cw = 0;
                for (j = 0; j < 6; j++) {
                    v = this.cells[line[i + j]];
                    cb = cb * 3 + v;
                    cw = cw * 3 + (v === 0 ? 0 : (v === 1 ? 2 : 1));
                }
                if (cb === 120) { if (maxB < 5) maxB = 5; } /* 011110 en base 3 */
                if (cw === 120) { if (maxW < 5) maxW = 5; }
            }
        }

        this.lineLevelB[lineId] = maxB;
        this.lineLevelW[lineId] = maxW;
    }

    makeMove(cell, color) {
        this.cells[cell] = color;
        this.hash1 ^= this.z1[cell * 3 + color];
        this.hash2 ^= this.z2[cell * 3 + color];

        var cls = this.cellLines[cell];
        for (var i = 0; i < cls.length; i++) {
            var id = cls[i];
            this.countLevelB[this.lineLevelB[id]]--;
            this.countLevelW[this.lineLevelW[id]]--;
            this.recomputeLine(id);
            this.countLevelB[this.lineLevelB[id]]++;
            this.countLevelW[this.lineLevelW[id]]++;
        }

        var n1 = this.neighbors1[cell];
        for (var j = 0; j < n1.length; j++) this.near1[n1[j]]++;
        var n2 = this.neighbors2[cell];
        for (var k = 0; k < n2.length; k++) this.near2[n2[k]]++;
        this.stoneCount++;
    }

    unmakeMove(cell, color) {
        var cls = this.cellLines[cell];
        var i, id;
        for (i = 0; i < cls.length; i++) {
            id = cls[i];
            this.countLevelB[this.lineLevelB[id]]--;
            this.countLevelW[this.lineLevelW[id]]--;
        }

        this.cells[cell] = 0;
        this.hash1 ^= this.z1[cell * 3 + color];
        this.hash2 ^= this.z2[cell * 3 + color];

        for (i = 0; i < cls.length; i++) {
            id = cls[i];
            this.recomputeLine(id);
            this.countLevelB[this.lineLevelB[id]]++;
            this.countLevelW[this.lineLevelW[id]]++;
        }

        var n1 = this.neighbors1[cell];
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
            while (yy >= 0 && yy < N && xx >= 0 && xx < N && cells[yy * N + xx] === color) {
                count++;
                yy += dy;
                xx += dx;
            }
            yy = y - dy;
            xx = x - dx;
            while (yy >= 0 && yy < N && xx >= 0 && xx < N && cells[yy * N + xx] === color) {
                count++;
                yy -= dy;
                xx -= dx;
            }
            if (count >= 5) return true;
        }
        return false;
    }

    patternScore(color) {
        var count = color === 1 ? this.countLevelB : this.countLevelW;
        var score = 0;
        for (var lvl = 1; lvl <= 6; lvl++) score += count[lvl] * JBVC_LEVEL_WEIGHT[lvl];
        var four = count[4] + count[5];
        var open3 = count[3];
        if (four >= 2) score += JBVC_DOUBLE_FOUR_BONUS;
        if (open3 >= 2) score += JBVC_DOUBLE_THREE_BONUS;
        if (four >= 1 && open3 >= 1) score += JBVC_FOUR_THREE_BONUS;
        return score;
    }

    evaluate(color) {
        var myB = color === 1;
        if (myB) {
            if (this.countLevelB[6] > 0) return JBVC_WIN;
            if (this.countLevelW[6] > 0) return -JBVC_WIN;
        } else {
            if (this.countLevelW[6] > 0) return JBVC_WIN;
            if (this.countLevelB[6] > 0) return -JBVC_WIN;
        }

        var score = 0;
        for (var lvl = 1; lvl <= 5; lvl++) {
            var w = JBVC_LEVEL_WEIGHT[lvl];
            if (myB) score += (this.countLevelB[lvl] - this.countLevelW[lvl]) * w;
            else score += (this.countLevelW[lvl] - this.countLevelB[lvl]) * w;
        }

        var myFour = myB ? (this.countLevelB[4] + this.countLevelB[5]) : (this.countLevelW[4] + this.countLevelW[5]);
        var oppFour = myB ? (this.countLevelW[4] + this.countLevelW[5]) : (this.countLevelB[4] + this.countLevelB[5]);
        var myOpen3 = myB ? this.countLevelB[3] : this.countLevelW[3];
        var oppOpen3 = myB ? this.countLevelW[3] : this.countLevelB[3];

        if (myFour >= 2) score += JBVC_DOUBLE_FOUR_BONUS;
        if (oppFour >= 2) score -= JBVC_DOUBLE_FOUR_BONUS;
        if (myOpen3 >= 2) score += JBVC_DOUBLE_THREE_BONUS;
        if (oppOpen3 >= 2) score -= JBVC_DOUBLE_THREE_BONUS;
        if (myFour >= 1 && myOpen3 >= 1) score += JBVC_FOUR_THREE_BONUS;
        if (oppFour >= 1 && oppOpen3 >= 1) score -= JBVC_FOUR_THREE_BONUS;

        return score + JBVC_TEMPO_BONUS;
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
                if (cells[yy * N + xx] === color) {
                    count++;
                    f++;
                    yy += dy;
                    xx += dx;
                } else break;
            }
            if (yy >= 0 && yy < N && xx >= 0 && xx < N && cells[yy * N + xx] === 0) {
                openEnds++;
                var yy2 = yy + dy;
                var xx2 = xx + dx;
                if (yy2 >= 0 && yy2 < N && xx2 >= 0 && xx2 < N && cells[yy2 * N + xx2] === color) {
                    var g = 0;
                    while (g < 3 && yy2 >= 0 && yy2 < N && xx2 >= 0 && xx2 < N && cells[yy2 * N + xx2] === color) {
                        g++;
                        yy2 += dy;
                        xx2 += dx;
                    }
                    gapCount += g;
                }
            }

            var b = 0;
            yy = y - dy;
            xx = x - dx;
            while (b < 4 && yy >= 0 && yy < N && xx >= 0 && xx < N) {
                if (cells[yy * N + xx] === color) {
                    count++;
                    b++;
                    yy -= dy;
                    xx -= dx;
                } else break;
            }
            if (yy >= 0 && yy < N && xx >= 0 && xx < N && cells[yy * N + xx] === 0) {
                openEnds++;
                var yy3 = yy - dy;
                var xx3 = xx - dx;
                if (yy3 >= 0 && yy3 < N && xx3 >= 0 && xx3 < N && cells[yy3 * N + xx3] === color) {
                    var g2 = 0;
                    while (g2 < 3 && yy3 >= 0 && yy3 < N && xx3 >= 0 && xx3 < N && cells[yy3 * N + xx3] === color) {
                        g2++;
                        yy3 -= dy;
                        xx3 -= dx;
                    }
                    gapCount += g2;
                }
            }

            var total = count + gapCount;
            var lvl = 0;
            if (count >= 5) {
                win = true;
                lvl = 6;
            } else if (count === 4 && openEnds >= 1) {
                four = true;
                lvl = 4;
            } else if (total >= 4 && gapCount > 0) {
                four = true;
                lvl = 4;
            } else if (count === 3 && openEnds >= 2) {
                openThree = true;
                lvl = 3;
            } else if (total >= 3 && gapCount > 0 && openEnds >= 1) {
                openThree = true;
                lvl = 3;
            } else if (count === 3 && openEnds >= 1) {
                lvl = 2;
            } else if (total >= 3) {
                lvl = 2;
            } else if (count === 2 && openEnds >= 2) {
                lvl = 1;
            } else if (count >= 1) {
                lvl = 1;
            }
            score += JBVC_LEVEL_WEIGHT[lvl];
            if (lvl >= 2) score += openEnds * 10;
        }
        var an = this._an;
        an.score = score;
        an.win = win;
        an.four = four;
        an.openThree = openThree;
        return an;
    }

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

    generateMoves(color, depth, ply, ttMove) {
        var N = this.N;
        var cells = this.cells;
        var opp = 3 - color;
        var moves = [];
        var useNear2 = (ply <= 1);
        var near = useNear2 ? this.near2 : this.near1;
        var kIdx = Math.min(ply, JBVC_MAX_PLY - 1) * 2;
        var killer0 = this.killers[kIdx];
        var killer1 = this.killers[kIdx + 1];
        var histBase = color * N * N;

        for (var cell = 0; cell < N * N; cell++) {
            if (cells[cell] !== 0 || near[cell] === 0) continue;
            var atk = this.analyzeMove(cell, color);
            if (atk.win) {
                return [{ cell: cell, score: JBVC_WIN, win: true, four: true, openThree: false }];
            }
            var atkScore = atk.score;
            var atkFour = atk.four;
            var atkOpenThree = atk.openThree;
            var def = this.analyzeMove(cell, opp);
            var score = atkScore + def.score * 0.9;
            if (def.win) score += 200000000;
            if (atkFour) score += 1000000;
            if (atkOpenThree) score += 50000;
            if (def.four) score += 200000;
            if (def.openThree) score += 20000;
            if (cell === ttMove) score += 5000000;
            if (cell === killer0 || cell === killer1) score += 300000;
            score += this.history[histBase + cell] * 0.05;
            score += this.centerBias[cell];
            moves.push({ cell: cell, score: score, win: false, four: atkFour, openThree: atkOpenThree });
        }

        if (moves.length === 0) {
            var fallback = this.findFallbackMove();
            if (fallback >= 0) moves.push({ cell: fallback, score: 0, win: false, four: false, openThree: false });
            return moves;
        }

        moves.sort(function (a, b) { return b.score - a.score; });
        var limit = this.getBranchLimit(depth, ply);
        if (moves.length > limit) moves.length = limit;
        return moves;
    }

    findWinningMove(color) {
        var N = this.N;
        var cells = this.cells;
        for (var cell = 0; cell < N * N; cell++) {
            if (cells[cell] === 0 && this.near2[cell] > 0 && this.isWinningAt(cell, color)) return cell;
        }
        return -1;
    }

    findWinningPoints(color) {
        var N = this.N;
        var cells = this.cells;
        var res = [];
        for (var cell = 0; cell < N * N; cell++) {
            if (cells[cell] === 0 && this.near2[cell] > 0 && this.isWinningAt(cell, color)) res.push(cell);
        }
        return res;
    }

    findFourMoves(color) {
        var N = this.N;
        var cells = this.cells;
        var res = [];
        for (var cell = 0; cell < N * N; cell++) {
            if (cells[cell] !== 0 || this.near2[cell] === 0) continue;
            var atk = this.analyzeMove(cell, color);
            if (atk.win || atk.four) res.push({ cell: cell, score: atk.score });
        }
        res.sort(function (a, b) { return b.score - a.score; });
        if (res.length > 24) res.length = 24;
        return res;
    }

    vcf(color, depth) {
        if (this.vcfAborted) return -1;
        this.vcfNodes++;
        if ((this.vcfNodes & 127) === 0 && Date.now() >= this.vcfDeadline) {
            this.vcfAborted = true;
            return -1;
        }
        var opp = 3 - color;
        var winMove = this.findWinningMove(color);
        if (winMove >= 0) return winMove;
        if (depth <= 0) return -1;

        var fours = this.findFourMoves(color);
        for (var i = 0; i < fours.length; i++) {
            var move = fours[i].cell;
            this.makeMove(move, color);
            if (this.isWinningAt(move, color)) {
                this.unmakeMove(move, color);
                return move;
            }
            var myWins = this.findWinningPoints(color);
            if (myWins.length === 0) {
                this.unmakeMove(move, color);
                continue;
            }
            var oppWins = this.findWinningPoints(opp);
            if (oppWins.length > 0) {
                this.unmakeMove(move, color);
                continue;
            }
            if (myWins.length >= 2) {
                this.unmakeMove(move, color);
                return move;
            }
            var block = myWins[0];
            this.makeMove(block, opp);
            var result = this.vcf(color, depth - 1);
            this.unmakeMove(block, opp);
            this.unmakeMove(move, color);
            if (result >= 0) return move;
            if (this.vcfAborted) return -1;
        }
        return -1;
    }

    search(depth, alpha, beta, color, ply) {
        if (this.stopped) return 0;
        this.nodes++;
        if ((this.nodes & 255) === 0 && Date.now() >= this.deadline) {
            this.stopped = true;
            return 0;
        }

        var opp = 3 - color;
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

        if (depth <= 0) return this.evaluate(color);
        if (this.stoneCount >= this.N * this.N) return 0;

        var moves = this.generateMoves(color, depth, ply, ttMove);
        if (moves.length === 0) return this.evaluate(color);
        if (moves[0].win) return JBVC_WIN - (ply + 1);

        var best = -JBVC_INF;
        var bestMove = moves[0].cell;
        var kIdx = Math.min(ply, JBVC_MAX_PLY - 1) * 2;

        for (var i = 0; i < moves.length; i++) {
            if (this.stopped) return 0;
            var mv = moves[i];
            this.makeMove(mv.cell, color);
            var val;
            {
                var ext = 0;
                if (mv.four && ply < 16 && depth <= 6) ext = 1;
                else if (mv.openThree && ply < 8 && depth <= 3) ext = 1;
                var newDepth = depth - 1 + ext;
                if (i === 0) {
                    val = -this.search(newDepth, -beta, -alpha, opp, ply + 1);
                } else {
                    val = -this.search(newDepth, -alpha - 1, -alpha, opp, ply + 1);
                    if (!this.stopped && val > alpha && val < beta) {
                        val = -this.search(newDepth, -beta, -alpha, opp, ply + 1);
                    }
                }
            }
            this.unmakeMove(mv.cell, color);
            if (this.stopped) return 0;

            if (val > best) { best = val; bestMove = mv.cell; }
            if (val > alpha) alpha = val;
            if (alpha >= beta) {
                if (this.killers[kIdx] !== mv.cell) {
                    this.killers[kIdx + 1] = this.killers[kIdx];
                    this.killers[kIdx] = mv.cell;
                }
                this.history[color * this.N * this.N + mv.cell] += depth * depth;
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
            this.ttDepth[idx] = depth;
            this.ttFlag[idx] = flag;
            this.ttValue[idx] = storeVal;
            this.ttMove[idx] = bestMove;
        }
        return best;
    }

    searchRoot(depth, color, alpha, beta) {
        var opp = 3 - color;
        var alphaOrig = alpha;
        var betaOrig = beta;
        var key1 = this.hash1 ^ (color === 1 ? this.zSide1 : 0);
        var key2 = this.hash2 ^ (color === 1 ? this.zSide2 : 0);
        var idx = key1 & JBVC_TT_MASK;
        var ttMove = -1;
        if (this.ttKey1[idx] === key1 && this.ttKey2[idx] === key2) ttMove = this.ttMove[idx];

        var moves = this.generateMoves(color, depth, 0, ttMove);
        if (moves.length === 0) return { move: -1, value: this.evaluate(color) };
        if (moves[0].win) return { move: moves[0].cell, value: JBVC_WIN - 1 };

        var bestMove = moves[0].cell;
        var bestValue = -JBVC_INF;

        for (var i = 0; i < moves.length; i++) {
            if (this.stopped) break;
            var mv = moves[i];
            this.makeMove(mv.cell, color);
            var val;
            {
                var ext = 0;
                if (mv.four && depth <= 6) ext = 1;
                else if (mv.openThree && depth <= 3) ext = 1;
                var newDepth = depth - 1 + ext;
                if (i === 0) {
                    val = -this.search(newDepth, -beta, -alpha, opp, 1);
                } else {
                    val = -this.search(newDepth, -alpha - 1, -alpha, opp, 1);
                    if (!this.stopped && val > alpha && val < beta) {
                        val = -this.search(newDepth, -beta, -alpha, opp, 1);
                    }
                }
            }
            this.unmakeMove(mv.cell, color);
            if (this.stopped) break;
            if (val > bestValue) { bestValue = val; bestMove = mv.cell; }
            if (val > alpha) alpha = val;
            if (alpha >= beta) break;
        }

        if (!this.stopped) {
            var flag = 0;
            if (bestValue <= alphaOrig) flag = 2;
            else if (bestValue >= betaOrig) flag = 1;
            this.ttKey1[idx] = key1;
            this.ttKey2[idx] = key2;
            this.ttDepth[idx] = depth;
            this.ttFlag[idx] = flag;
            this.ttValue[idx] = bestValue;
            this.ttMove[idx] = bestMove;
        }
        return { move: bestMove, value: bestValue };
    }

    playNormal(color) {
        var opp = 3 - color;
        var win = this.findWinningMove(color);
        if (win >= 0) return this.cellToPos(win);

        var oppWin = this.findWinningMove(opp);
        if (oppWin >= 0) return this.cellToPos(oppWin);

        var now = Date.now();
        var remain = this.deadline - now;
        if (remain > 50) {
            var vcfBudget = Math.min(1500, Math.max(50, remain * 0.3));
            this.vcfDeadline = now + vcfBudget;
            this.vcfAborted = false;
            this.vcfNodes = 0;
            var vcfMove = this.vcf(color, 12);
            if (vcfMove >= 0 && this.cells[vcfMove] === 0) return this.cellToPos(vcfMove);
        }

        this.stopped = false;
        var bestMove = -1;
        var bestValue = -JBVC_INF;
        var initMoves = this.generateMoves(color, 4, 0, -1);
        if (initMoves.length > 0) bestMove = initMoves[0].cell;

        for (var depth = 1; depth <= 32; depth++) {
            if (Date.now() >= this.deadline) break;
            this.stopped = false;
            var res = this.searchRoot(depth, color, -JBVC_INF, JBVC_INF);
            if (this.stopped) break;
            if (res.move >= 0) {
                bestMove = res.move;
                bestValue = res.value;
                this.lastDepth = depth;
            }
            if (bestValue > JBVC_WIN_THRESHOLD) break;
            if (Date.now() >= this.deadline) break;
        }
        this.lastNodes = this.nodes;

        if (bestMove < 0) bestMove = this.findFallbackMove();
        if (bestMove < 0) bestMove = 0;
        return this.cellToPos(bestMove);
    }

    decideSwap() {
        this.stopped = false;
        var bestRes = null;
        for (var depth = 1; depth <= 8; depth++) {
            if (Date.now() >= this.deadline) break;
            this.stopped = false;
            var res = this.searchRoot(depth, 2, -JBVC_INF, JBVC_INF);
            if (this.stopped) break;
            if (res.move >= 0) {
                bestRes = res;
                this.lastDepth = depth;
            }
            if (bestRes && bestRes.value > JBVC_WIN_THRESHOLD) break;
            if (bestRes && bestRes.value < -JBVC_WIN_THRESHOLD) break;
            if (Date.now() >= this.deadline) break;
        }

        if (!bestRes || bestRes.move < 0) {
            var moves = this.generateMoves(2, 4, 0, -1);
            if (moves.length > 0) bestRes = { move: moves[0].cell, value: 0 };
        }
        if (!bestRes || bestRes.move < 0) {
            var fallback = this.findFallbackMove();
            if (fallback < 0) return 'BLACK';
            return this.cellToPos(fallback);
        }
        if (bestRes.value < -100) return 'BLACK';
        return this.cellToPos(bestRes.move);
    }

    playOpening() {
        var N = this.N;
        var center = Math.floor(N / 2);
        var r = Math.min(2, Math.floor(N / 2));
        var cells = [];
        var dy, dx, y, x, cell;
        for (dy = -r; dy <= r; dy++) {
            for (dx = -r; dx <= r; dx++) {
                y = center + dy;
                x = center + dx;
                if (y >= 0 && y < N && x >= 0 && x < N) {
                    cell = y * N + x;
                    if (this.cells[cell] === 0) cells.push(cell);
                }
            }
        }

        var blackCells = [];
        for (var i = 0; i < cells.length; i++) {
            var c = cells[i];
            var cy = (c / N) | 0;
            var cx = c - cy * N;
            if (Math.abs(cy - center) <= 1 && Math.abs(cx - center) <= 1) blackCells.push(c);
        }

        var bestTriple = null;
        var bestBalance = Infinity;
        for (var a = 0; a < blackCells.length; a++) {
            for (var b = a + 1; b < blackCells.length; b++) {
                var b1 = blackCells[a];
                var b2 = blackCells[b];
                var y1 = (b1 / N) | 0;
                var x1 = b1 - y1 * N;
                var y2 = (b2 / N) | 0;
                var x2 = b2 - y2 * N;
                if (Math.abs(y1 - y2) > 1 || Math.abs(x1 - x2) > 1) continue;
                for (var w = 0; w < cells.length; w++) {
                    var wc = cells[w];
                    if (wc === b1 || wc === b2) continue;
                    this.makeMove(b1, 1);
                    this.makeMove(b2, 1);
                    this.makeMove(wc, 2);
                    var sb = this.patternScore(1);
                    var sw = this.patternScore(2);
                    var balance = Math.abs(sb - sw - JBVC_TEMPO_BONUS);
                    if (balance < bestBalance) {
                        bestBalance = balance;
                        bestTriple = [b1, b2, wc];
                    }
                    this.unmakeMove(wc, 2);
                    this.unmakeMove(b2, 1);
                    this.unmakeMove(b1, 1);
                }
            }
        }

        if (!bestTriple) {
            var empty = [];
            for (var e = 0; e < N * N; e++) if (this.cells[e] === 0) empty.push(e);
            var c0 = Math.floor(N / 2) * N + Math.floor(N / 2);
            var chosen = [];
            if (this.cells[c0] === 0) chosen.push(c0);
            for (var q = 0; q < empty.length && chosen.length < 3; q++) {
                if (empty[q] !== c0) chosen.push(empty[q]);
            }
            while (chosen.length < 3) chosen.push(0);
            bestTriple = [chosen[0], chosen[1], chosen[2]];
        }

        return [this.cellToPos(bestTriple[0]), this.cellToPos(bestTriple[1]), this.cellToPos(bestTriple[2])];
    }

    compute(board, move_state, time) {
        var start = Date.now();
        this.loadBoard(board);
        this.moveCount++;

        var remaining = 10000;
        if (typeof time === 'number' && isFinite(time)) remaining = time;
        else if (Array.isArray(time) && time.length > 0) remaining = time[0];
        if (remaining <= 0) remaining = 100;

        if (this.prevTime !== null) {
            var delta = this.prevTime - remaining;
            if (this.lastElapsed > 200) {
                if (delta > this.lastElapsed * 0.5) this.perMoveMode = false;
                else if (delta < 50) this.perMoveMode = true;
            }
        }
        this.prevTime = remaining;

        var state = String(move_state);
        var budget;
        var result;

        if (state === '1') {
            budget = Math.min(250, remaining - 50);
            if (budget < 20) budget = 20;
            this.deadline = start + budget;
            result = this.playOpening();
        } else if (state === '2' || state === '3') {
            if (this.perMoveMode === true) budget = Math.min(1500, remaining - JBVC_SAFETY);
            else if (this.perMoveMode === false) budget = Math.max(200, Math.min(1000, remaining * 0.12));
            else budget = Math.min(1000, remaining - JBVC_SAFETY);
            if (budget < 50) budget = 50;
            if (budget > remaining - 50) budget = Math.max(20, remaining - 50);
            this.deadline = start + budget;
            result = this.decideSwap();
        } else {
            var color = state === 'W' ? 2 : 1;
            this.color = color;
            if (this.perMoveMode === true) {
                budget = Math.min(JBVC_MAX_BUDGET, remaining - JBVC_SAFETY);
            } else if (this.perMoveMode === false) {
                var movesLeft = Math.max(6, 24 - this.moveCount);
                budget = Math.min(JBVC_MAX_BUDGET, Math.max(JBVC_MIN_BUDGET, remaining / movesLeft));
            } else {
                budget = Math.min(2000, remaining - JBVC_SAFETY);
            }
            if (budget < 50) budget = 50;
            if (budget > remaining - 50) budget = Math.max(20, remaining - 50);
            this.deadline = start + budget;
            result = this.playNormal(color);
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
            if (Array.isArray(result) && result.length === 2 && Array.isArray(result[0]) && Array.isArray(result[1])) {
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
                    var d = Math.abs(x - center) + Math.abs(y - center);
                    list.push({ x: x, y: y, d: d });
                }
            }
        }
        list.sort(function (a, b) { return a.d - b.d; });
        var res = [];
        for (var i = 0; i < list.length && res.length < 3; i++) res.push([list[i].x, list[i].y]);
        while (res.length < 3) res.push([0, 0]);
        return res;
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
}

if (typeof module !== 'undefined' && module.exports) module.exports = JBVCAgent;
